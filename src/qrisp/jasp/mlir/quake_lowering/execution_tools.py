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
#    extract via regular expressions.
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

import re
import cudaq
from cudaq import cudaq_runtime
from cudaq.mlir.ir import Module, NoneType
from cudaq.mlir.dialects import quake, cc

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
    # 1a. Generate the execution context
    kernel = cudaq.make_kernel()
    func_name = kernel.funcName              
    entry_point = kernel.funcNameEntryPoint  
    uniq_name = func_name.replace("__nvqpp__mlirgen__", "")  

    # 1b. Force CUDA-Q to generate a valid hardware layout
    @cudaq.kernel
    def _dummy_extractor():
        pass
    
    dummy_mlir = str(_dummy_extractor)
    data_layout_match = re.search(r'llvm\.data_layout\s*=\s*"[^"]+"', dummy_mlir)
    target_triple_match = re.search(r'llvm\.target_triple\s*=\s*"[^"]+"', dummy_mlir)

    if not data_layout_match:
        raise RuntimeError("Failed to extract llvm.data_layout from the active CUDA-Q environment.")

    # ------------------------------------------------------------------ #
    # Step 2: Extract the function body from the input MLIR.
    # ------------------------------------------------------------------ #
    func_start = mlir_str.index('func.func')

    depth = 0
    brace_groups = []
    i = func_start
    while i < len(mlir_str):
        if mlir_str[i] == '{':
            if depth == 0: group_start = i
            depth += 1
        elif mlir_str[i] == '}':
            depth -= 1
            if depth == 0:
                brace_groups.append((group_start, i))
                if len(brace_groups) == 2: break
        i += 1

    body_start, body_end = brace_groups[1]
    func_body = mlir_str[body_start + 1 : body_end].strip()

    # ------------------------------------------------------------------ #
    # Step 3: Determine the return type and build the ".run" variant.
    # ------------------------------------------------------------------ #
    return_match = re.search(r'func\.return\s+(%\w+)\s*:\s*(.+)', func_body)
    if return_match:
        return_type_str = return_match.group(2).strip()
        return_sig = f' -> {return_type_str}'
        run_body = re.sub(
            r'func\.return\s+(%\w+)\s*:\s*(.+)',
            r'cc.log_output \1 : \2\n    return',
            func_body
        )
    else:
        return_type_str = None
        return_sig = ''
        run_body = func_body

    # ------------------------------------------------------------------ #
    # Step 4: Assemble the complete CUDA-Q-compatible MLIR module.
    # ------------------------------------------------------------------ #
    run_func_name = func_name + ".run"
    run_entry_name = func_name + ".run.entry"

    mod_attributes = [f'cc.python_uniqued = "{uniq_name}"']
    mod_attributes.append(data_layout_match.group(0))
    if target_triple_match:
        mod_attributes.append(target_triple_match.group(0))

    mod_attributes.append(
        f'quake.mangled_name_map = {{\n'
        f'    {func_name} = "{entry_point}",\n'
        f'    {run_func_name} = "{run_entry_name}"\n'
        f'  }}'
    )
    attributes_str = ",\n  ".join(mod_attributes)

    adapted_mlir = (
        f'module attributes {{\n  {attributes_str}\n}} {{\n'
        f'  func.func @{func_name}(){return_sig} attributes '
        f'{{"cudaq-entrypoint", "cudaq-kernel"}} {{\n    {func_body}\n  }}\n'
        f'  func.func @{run_func_name}() attributes '
        f'{{"cudaq-entrypoint", "cudaq-kernel", no_this'
    )

    if return_type_str:
        adapted_mlir += f', quake.cudaq_run = [{return_type_str}]'

    adapted_mlir += (
        f'}} {{\n    {run_body}\n  }}\n'
        f'  func.func @{run_entry_name}() attributes {{no_this}} {{\n    return\n  }}\n'
        f'}}\n'
    )

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
