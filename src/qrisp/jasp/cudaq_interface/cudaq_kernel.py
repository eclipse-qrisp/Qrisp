"""********************************************************************************
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
# Workflow:
# 1. Target Extraction: We define an empty Python function decorated with 
#    `@cudaq.kernel` to trigger the CUDA-Q compiler pipeline. This forces 
#    the underlying LLVM compiler to generate the exact, natively-matched 
#    layout and target triple for the host environment, which we then 
#    extract via regular expressions. If that fails
#    (e.g. in CI environments where str() doesn't trigger full LLVM
#    lowering), we fall back to well-known platform defaults derived from
#    the host's architecture and OS.
#
# 2. Interface Adaptation (cudaq_prep.py): 
#    We inject the extracted hardware specifications 
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

from collections.abc import Callable
from typing import Literal
import inspect
import platform
import re
import sys
import warnings

import cudaq
from cudaq.kernel.kernel_decorator import PyKernelDecorator
from cudaq.mlir.ir import Module, NoneType
from cudaq.mlir.dialects import quake as cudaq_quake_dialect, cc as cudaq_cc_dialect

from xdsl.dialects.builtin import ModuleOp

from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.mlir.quake_lowering.jaspr_to_quake import jaspr_to_quake_mlir
from qrisp.jasp.cudaq_interface.annotations import FixedShapeNDArray
from qrisp.jasp.cudaq_interface.cudaq_prep import prepare_module_for_cudaq


# ------------------------------------------------------------------ #
# Platform-aware LLVM attribute defaults
# ------------------------------------------------------------------ #

_PLATFORM_DEFAULTS = {
    ("x86_64", "linux"): (
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
        "x86_64-unknown-linux-gnu",
    ),
    ("aarch64", "linux"): (
        "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
        "aarch64-unknown-linux-gnu",
    ),
    ("arm64", "darwin"): (
        "e-m:o-i64:64-i128:128-n32:64-S128-Fn32",
        "arm64-apple-macosx14.0.0",
    ),
    ("x86_64", "darwin"): (
        "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
        "x86_64-apple-macosx14.0.0",
    ),
}


def _detect_platform_key() -> tuple:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        os_key = "linux"
    elif sys.platform == "darwin":
        os_key = "darwin"
    else:
        os_key = sys.platform
    return (machine, os_key)


def _get_llvm_attributes() -> tuple[str, str | None]:
    """Return (data_layout_str, target_triple_str | None)."""
    data_layout_str = None
    target_triple_str = None

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
        pass

    if data_layout_str is None:
        key = _detect_platform_key()
        defaults = _PLATFORM_DEFAULTS.get(key)
        if defaults is None:
            raise RuntimeError(
                f"Failed to extract llvm.data_layout from CUDA-Q, "
                f"no default for {key}. Supported: {list(_PLATFORM_DEFAULTS.keys())}"
            )
        data_layout_str, target_triple_str = defaults
        warnings.warn(
            f"Could not extract llvm.data_layout from CUDA-Q; using platform default for {key}.",
            stacklevel=3,
        )

    return data_layout_str, target_triple_str


# ------------------------------------------------------------------ #
# xDSL → CUDA-Q serialization normalization
# ------------------------------------------------------------------ #


def _normalize_xdsl_to_cudaq(mlir_str: str) -> str:
    """Normalize xDSL's generic printing format to what CUDA-Q's parser expects.

    This handles purely syntactic differences between xDSL's printer output
    and MLIR's standard format that CUDA-Q uses:

    1. `builtin.module` → `module`
    2. `func.return` → `return`  (bare return, not qualified)
    3. `-> (T)` → `-> T`  (single return type without parens)

    These are safe textual substitutions because:
    - `builtin.module` only appears at the top level
    - `func.return` is always a standalone op (never inside a string/attr)
    - Single-element return type parens are redundant in MLIR syntax
    """
    # 1. builtin.module → module
    mlir_str = mlir_str.replace("builtin.module", "module", 1)

    # 2. func.return → return
    mlir_str = mlir_str.replace("func.return", "return")

    # 3. -> (T) → -> T  (only for single return types, not tuples)
    # Match `-> (` followed by a type (no comma) followed by `)`
    # This regex is safe: it only matches single-type returns
    mlir_str = re.sub(
        r"->\s*\(([^,\)]+)\)",
        r"-> \1",
        mlir_str,
    )

    return mlir_str


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #


def cudaq_kernel_from_xdsl_module(
    xdsl_module: ModuleOp,
    execution_mode: Literal["run", "sample"] = "run",
) -> PyKernelDecorator:
    """
    Compiles an xDSL ModuleOp into a native PyKernelDecorator.

    Parameters
    ----------
    xdsl_module : ModuleOp
        An xDSL module representing the quantum computation in Quake and CC dialects.
        Must contain a @main function as entrypoint.
    execution_mode : "run" | "sample"
        Controls how the compiled kernel is structured for CUDA-Q's backend,
        determining which execution API the resulting kernel is compatible with.

        - ``"run"`` — Prepares the kernel for use with ``cudaq.run()``. This mode
          preserves the function's return values by synthesizing additional
          ``.run`` and ``.run.entry`` function variants. The ``.run`` variant
          replaces ``func.return`` operations with ``cc.log_output`` calls,
          which is the mechanism CUDA-Q uses to capture and aggregate per-shot
          measurement results across repeated executions. Use this mode when
          you need to retrieve computed classical values (e.g., expectation
          values, bit-strings with post-processing) from the quantum kernel.

        - ``"sample"`` — Prepares the kernel for use with ``cudaq.sample()``.
          This mode strips all return values from the kernel (making it
          void-returning), as ``cudaq.sample()`` collects measurement results
          implicitly from qubit measurements embedded in the circuit rather
          than from explicit return statements. Use this mode when you only
          need measurement count statistics (histograms) from the quantum
          circuit.

    Returns
    -------
    PyKernelDecorator
        A compiled, callable CUDA-Q kernel.
    """

    module = xdsl_module.clone()

    # Get CUDA-Q naming from a dummy kernel
    kernel = cudaq.make_kernel()
    func_name = kernel.funcName
    entry_point = kernel.funcNameEntryPoint
    uniq_name = func_name.replace("__nvqpp__mlirgen__", "")

    # Get platform LLVM attributes
    data_layout_str, target_triple_str = _get_llvm_attributes()

    # Apply all structural passes (in-place on the xDSL module)
    prepare_module_for_cudaq(
        module,
        func_name=func_name,
        entry_point=entry_point,
        uniq_name=uniq_name,
        data_layout=data_layout_str,
        target_triple=target_triple_str,
        execution_mode=execution_mode,
    )

    # Serialize to string exactly once, then normalize for CUDA-Q
    raw_mlir = str(module)
    adapted_mlir = _normalize_xdsl_to_cudaq(raw_mlir)

    # Parse into CUDA-Q
    with kernel.ctx:
        try:
            new_module = Module.parse(adapted_mlir, kernel.ctx)
        except Exception:
            cudaq_quake_dialect.register_dialect(context=kernel.ctx)
            cudaq_cc_dialect.register_dialect(context=kernel.ctx)
            new_module = Module.parse(adapted_mlir, kernel.ctx)

        kernel.module = new_module
        NoneType.get(context=kernel.ctx)

    return PyKernelDecorator(None, kernelName=uniq_name, module=kernel.module)


# ------------------------------------------------------------------ #
# @cudaq_kernel decorator
# ------------------------------------------------------------------ #

_ANNOTATION_TO_DUMMY = {
    int: 0,
    float: 0.0,
    bool: False,
}


def cudaq_kernel(
    func_arg: Callable | None = None,
    execution_mode: Literal["run", "sample"] = "run",
) -> PyKernelDecorator:
    """Decorator that compiles a Qrisp function to a native CUDA-Q kernel."""
    if func_arg is None:
        return lambda x: cudaq_kernel(x, execution_mode=execution_mode)

    sig = inspect.signature(func_arg)
    params = list(sig.parameters.values())
    _supported = list(_ANNOTATION_TO_DUMMY.keys()) + ["FixedShapeNDArray(dtype, size)"]

    dummy_args = []
    for p in params:
        if p.annotation is inspect.Parameter.empty:
            raise RuntimeError(
                f"@cudaq_kernel: parameter '{p.name}' of "
                f"'{func_arg.__name__}' requires a type annotation. "
                f"Supported: {_supported}."
            )
        if isinstance(p.annotation, FixedShapeNDArray):
            dummy_args.append(p.annotation.make_dummy())
        elif p.annotation in _ANNOTATION_TO_DUMMY:
            dummy_args.append(_ANNOTATION_TO_DUMMY[p.annotation])
        else:
            ann_name = getattr(p.annotation, "__name__", repr(p.annotation))
            raise RuntimeError(
                f"@cudaq_kernel: unsupported annotation "
                f"'{ann_name}' for parameter '{p.name}' of "
                f"'{func_arg.__name__}'. Supported: {_supported}."
            )

    try:
        mlir_module = jaspr_to_quake_mlir(make_jaspr(func_arg)(*dummy_args), execution_mode=execution_mode)
    except Exception as e:
        raise RuntimeError(f"Failed to compile Qrisp function '{func_arg.__name__}' to MLIR: {e}")

    return cudaq_kernel_from_xdsl_module(mlir_module, execution_mode=execution_mode)
