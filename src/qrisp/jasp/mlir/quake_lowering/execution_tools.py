"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

# ====================================================================== #
# CUSTOM INGESTION PIPELINE: Bridging External MLIR to CUDA-Q
# ====================================================================== #
# Rationale:
# CUDA-Q's execution backend (C++) strictly requires an MLIR module to 
# specify the host machine's exact memory architecture (llvm.data_layout) 
# and hardware target (llvm.target_triple) to successfully compile and 
# allocate memory. Currently, the CUDA-Q Python API lacks a native 
# mechanism to cleanly ingest externally compiled MLIR strings. Omitting 
# these hardware attributes results in fatal "missing data layout" 
# runtime crashes.
#
# Workflow & Mechanism:
# 1. Target Extraction: We define an empty Python function decorated with 
#    `@cudaq.kernel` to trigger the CUDA-Q compiler pipeline. This forces 
#    the underlying LLVM compiler to generate the exact, natively-matched 
#    layout and target triple for the host environment, which we then 
#    extract via regular expressions. If that fails
#    (e.g. in CI environments where str() doesn't trigger full LLVM
#    lowering), we fall back to well-known platform defaults derived from
#    the host's architecture and OS.
#
# 2. Interface Adaptation: We inject the extracted hardware specifications 
#    into the Qrisp-generated MLIR. Crucially, we also clone the primary 
#    entry function to create a required `.run` variant. During this cloning, 
#    we translate standard `func.return` instructions into `cc.log_output` 
#    operations. This structural change is required, as it is the exact 
#    mechanism CUDA-Q uses to capture and aggregate individual per-shot 
#    measurement data during simulation.
#
# 3. Re-Compilation: The fully adapted, hardware-aware MLIR string is fed 
#    back into CUDA-Q's internal compiler via `Module.parse()`. This 
#    re-compiles the string within the active MLIR context, resulting in a 
#    valid kernel object that the C++ backend can safely execute.
# ====================================================================== #

import platform
import re
import sys
import warnings

import cudaq
from cudaq import cudaq_runtime
from cudaq.mlir.ir import Module, NoneType
from cudaq.mlir.dialects import quake, cc


# ------------------------------------------------------------------ #
# Platform-aware LLVM attribute defaults
# ------------------------------------------------------------------ #

# These are the standard LLVM data layouts and target triples for
# platforms commonly used with CUDA-Q.  They describe the host's
# classical memory architecture and are deterministic per (arch, OS).
_PLATFORM_DEFAULTS = {
    # (machine, sys.platform prefix) → (data_layout, target_triple)
    ("x86_64", "linux"): (
        'e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128',
        'x86_64-unknown-linux-gnu',
    ),
    ("aarch64", "linux"): (
        'e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128',
        'aarch64-unknown-linux-gnu',
    ),
    ("arm64", "darwin"): (
        'e-m:o-i64:64-i128:128-n32:64-S128-Fn32',
        'arm64-apple-macosx14.0.0',
    ),
    ("x86_64", "darwin"): (
        'e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128',
        'x86_64-apple-macosx14.0.0',
    ),
}


def _detect_platform_key() -> tuple:
    """Return ``(machine, os_prefix)`` for the current host."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        os_key = "linux"
    elif sys.platform == "darwin":
        os_key = "darwin"
    else:
        os_key = sys.platform
    return (machine, os_key)


def _get_llvm_attributes() -> tuple:
    """Extract ``llvm.data_layout`` and ``llvm.target_triple`` strings.

    Strategy:
    1. Compile a dummy ``@cudaq.kernel`` and regex-search its MLIR string.
    2. If that fails, fall back to platform-based defaults.

    Returns
    -------
    (data_layout_attr, target_triple_attr)
        Each is a string like ``'llvm.data_layout = "..."'`` ready for
        insertion into a ``module attributes { }`` block.  The target
        triple may be ``None`` if not available.

    Raises
    ------
    RuntimeError
        If neither extraction nor platform defaults succeed.
    """
    data_layout_str = None
    target_triple_str = None

    # --- Method 1: Extract from a compiled dummy kernel ---
    try:
        @cudaq.kernel
        def _dummy_extractor():
            pass

        dummy_mlir = str(_dummy_extractor)
        dl_match = re.search(r'llvm\.data_layout\s*=\s*"([^"]+)"', dummy_mlir)
        tt_match = re.search(r'llvm\.target_triple\s*=\s*"([^"]+)"', dummy_mlir)

        if dl_match:
            data_layout_str = dl_match.group(1)
        if tt_match:
            target_triple_str = tt_match.group(1)
    except Exception:
        pass  # Fall through to platform defaults

    # --- Method 2: Platform-based defaults ---
    if data_layout_str is None:
        key = _detect_platform_key()
        defaults = _PLATFORM_DEFAULTS.get(key)

        if defaults is None:
            raise RuntimeError(
                f"Failed to extract llvm.data_layout from the active "
                f"CUDA-Q environment, and no platform default is available "
                f"for {key}.  Supported platforms: "
                f"{list(_PLATFORM_DEFAULTS.keys())}"
            )

        data_layout_str, target_triple_str = defaults
        warnings.warn(
            f"Could not extract llvm.data_layout from CUDA-Q; "
            f"using platform default for {key}.",
            stacklevel=3,
        )

    data_layout_attr = f'llvm.data_layout = "{data_layout_str}"'
    target_triple_attr = (
        f'llvm.target_triple = "{target_triple_str}"'
        if target_triple_str else None
    )
    return data_layout_attr, target_triple_attr


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #


def run_quake_mlir(mlir_str: str, shots: int = 100) -> list:
    """
    Executes a given MLIR string (representing a quantum kernel) using the CUDA-Q runtime. The input MLIR is expected to define a function with the 
    `cudaq-entrypoint` attribute. The function can return a value, which will be captured and returned as a list of measurement results.
    
    Parameters
    ----------
    mlir_str : str
        The MLIR code as a string. Must contain a function with the `cudaq-entrypoint` attribute.
    shots : int
        The number of times to execute the kernel for statistical sampling. Default is 100.

    Returns
    -------
    list
        A list of measurement results captured from the kernel execution.

    Examples
    --------

    Example usage of `run_quake_mlir`:

    ::

        from qrisp import QuantumVariable, h, cx, measure,x 
        from qrisp.jasp import make_jaspr
        from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake, run_quake_mlir

        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        # Qrisp → Jaspr → Quake MLIR → CUDA-Q execution
        module = jaspr_to_quake(make_jaspr(bell)())
        mlir_str = str(module)

        result = run_quake_mlir(mlir_str, shots=10)
        print(result) 
        # [0, 0, 3, 0, 3, 0, 3, 3, 0, 0]  

    """
    # ------------------------------------------------------------------ #
    # Step 1: Create a dummy PyKernel to obtain CUDA-Q's internal naming
    #         conventions, and extract native LLVM layouts from a fully
    #         compiled dummy kernel to prevent missing layout errors.
    # ------------------------------------------------------------------ #
    kernel = cudaq.make_kernel()
    func_name = kernel.funcName
    entry_point = kernel.funcNameEntryPoint
    uniq_name = func_name.replace("__nvqpp__mlirgen__", "")

    data_layout_attr, target_triple_attr = _get_llvm_attributes()

    # ------------------------------------------------------------------ #
    # Step 2: Extract the inner MLIR and separate @main from other funcs
    # ------------------------------------------------------------------ #
    func_start = mlir_str.find('func.func')
    last_brace = mlir_str.rfind('}')
    inner_mlir = mlir_str[func_start:last_brace].strip()

    # Locate the @main function signature specifically
    main_match = re.search(r'func\.func\s+(?:public\s+)?@main', inner_mlir)
    if not main_match:
        raise ValueError("Could not find @main function in MLIR string.")

    # Robustly find the function body, skipping optional `attributes { ... }`
    anchor = main_match.end()
    search_idx = anchor
    body_start_idx = -1
    
    while search_idx < len(inner_mlir):
        if inner_mlir[search_idx] == '{':
            text_before = inner_mlir[anchor:search_idx].strip()
            if text_before.endswith('attributes'):
                # This is an attributes block, skip it
                depth = 1
                search_idx += 1
                while search_idx < len(inner_mlir) and depth > 0:
                    if inner_mlir[search_idx] == '{': depth += 1
                    elif inner_mlir[search_idx] == '}': depth -= 1
                    search_idx += 1
                anchor = search_idx  # move anchor past the attributes block
                continue
            else:
                # Found the actual body
                body_start_idx = search_idx
                break
        search_idx += 1

    if body_start_idx == -1:
        raise ValueError("Could not find body for @main.")

    # Extract the exact block of @main using brace counting
    depth = 0
    end_idx = -1
    for i in range(body_start_idx, len(inner_mlir)):
        if inner_mlir[i] == '{': 
            depth += 1
        elif inner_mlir[i] == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
                
    main_func_body = inner_mlir[body_start_idx+1:end_idx].strip()
    
    # Keep any other functions (like @closed_call) perfectly intact
    other_functions = inner_mlir[:main_match.start()] + inner_mlir[end_idx+1:]

    # ------------------------------------------------------------------ #
    # Step 3: Determine the return type and build the ".run" variant.
    # ------------------------------------------------------------------ #
    return_match = re.search(r'func\.return\s+(%\w+)\s*:\s*(.+)', main_func_body)
    if return_match:
        return_type_str = return_match.group(2).strip()
        return_sig = f' -> {return_type_str}'
        run_body = re.sub(
            r'func\.return\s+(%\w+)\s*:\s*(.+)',
            r'cc.log_output \1 : \2\n    return',
            main_func_body
        )
    else:
        return_type_str = None
        return_sig = ''
        run_body = main_func_body

# ------------------------------------------------------------------ #
    # Step 4: Assemble the complete CUDA-Q-compatible MLIR module.
    # ------------------------------------------------------------------ #
    run_func_name = func_name + ".run"
    run_entry_name = func_name + ".run.entry"

    mod_attributes = [f'cc.python_uniqued = "{uniq_name}"']
    mod_attributes.append(data_layout_attr)
    if target_triple_attr:
        mod_attributes.append(target_triple_attr)

    mod_attributes.append(
        f'quake.mangled_name_map = {{\n'
        f'    {func_name} = "{entry_point}",\n'
        f'    {run_func_name} = "{run_entry_name}"\n'
        f'  }}'
    )
    attributes_str = ",\n  ".join(mod_attributes)

    adapted_mlir = f'module attributes {{\n  {attributes_str}\n}} {{\n'
    
    # 1. Inject all helper functions first (e.g. @closed_call)
    if other_functions.strip():
        adapted_mlir += f'  {other_functions.strip()}\n\n'
        
    # 2. Inject the main function and rename it to the CUDA-Q target name
    adapted_mlir += (
        f'  func.func @{func_name}(){return_sig} attributes '
        f'{{"cudaq-entrypoint", "cudaq-kernel"}} {{\n    {main_func_body}\n  }}\n'
    )

    # 3. Inject the .run variant for state measurements
    run_attrs = f'{{"cudaq-entrypoint", "cudaq-kernel", no_this'
    if return_type_str:
        run_attrs += f', quake.cudaq_run = [{return_type_str}]'
    run_attrs += '}'
    adapted_mlir += (
        f'  func.func @{run_func_name}() attributes {run_attrs} {{\n    {run_body}\n  }}\n'
    )

    # 4. Inject the dummy entry
    adapted_mlir += (
        f'  func.func @{run_entry_name}() attributes {{no_this}} {{\n    return\n  }}\n'
        f'}}\n'
    )

    print("Adapted MLIR for CUDA-Q execution:")
    print(adapted_mlir)
    # ------------------------------------------------------------------ #
    # Step 5: Parse the adapted MLIR
    # ------------------------------------------------------------------ #
    with kernel.ctx:
        try:
            new_module = Module.parse(adapted_mlir, kernel.ctx)
        except Exception:
            quake.register_dialect(context=kernel.ctx)
            cc.register_dialect(context=kernel.ctx)
            new_module = Module.parse(adapted_mlir, kernel.ctx)

        kernel.module = new_module
        ret_type = NoneType.get(context=kernel.ctx)

    # ------------------------------------------------------------------ #
    # Step 6: Execute via the CUDA-Q runtime.
    # ------------------------------------------------------------------ #
    return cudaq_runtime.run_impl(
        uniq_name + ".run",
        kernel.module,
        ret_type,
        shots,
        None,   # noise_model
        0       # qpu_id
    )
