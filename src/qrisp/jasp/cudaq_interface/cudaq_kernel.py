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

"""
cudaq_kernel.py — refactored to use xDSL structural IR manipulation.

All string-based MLIR surgery is replaced by pass6_cudaq_prep which operates
on the xDSL ModuleOp directly.  Serialization happens exactly once, right
before handing to CUDA-Q's Module.parse.

A small normalization step converts xDSL's generic printing conventions
to the format CUDA-Q's parser expects (e.g. `builtin.module` → `module`,
`func.return` → `return`).
"""

import inspect
import platform
import re
import sys
from typing import Callable, Literal
import warnings

import numpy as np

import cudaq
from cudaq import cudaq_runtime
from cudaq.kernel.kernel_decorator import PyKernelDecorator
from cudaq.mlir.ir import Module, NoneType
from cudaq.mlir.dialects import quake as cudaq_quake_dialect, cc as cudaq_cc_dialect

from xdsl.dialects.builtin import ModuleOp

from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.mlir.quake_lowering.jaspr_to_quake import jaspr_to_quake_mlir
from qrisp.jasp.cudaq_interface.annotations import FixedShapeNDArray
from qrisp.jasp.cudaq_interface.pass6_cudaq_prep import prepare_module_for_cudaq


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


def cudaq_kernel_from_mlir(
    xdsl_module: str | ModuleOp,
    execution_mode: Literal["run", "sample"] = "run",
) -> PyKernelDecorator:
    """
    Compiles a Quake MLIR string (or xDSL ModuleOp) into a native PyKernelDecorator.

    Parameters
    ----------
    mlir_input : str | ModuleOp
        Either a Quake MLIR source string or an already-parsed xDSL ModuleOp.
        Must contain a @main function.
    execution_mode : "run" | "sample"
        "run" — synthesizes .run/.run.entry for cudaq.run.
        "sample" — void-return kernel for cudaq.sample.

    Returns
    -------
    PyKernelDecorator
        A compiled, callable CUDA-Q kernel.
    """
    # Accept either string or xDSL module
    if isinstance(xdsl_module, str):
        raise NotImplementedError(
            "String-based MLIR parsing is not yet implemented. Please provide an xDSL ModuleOp instead."
        )
    #    from xdsl.context import Context
    #    from xdsl.parser import Parser
    #    from xdsl.dialects import builtin, func, arith, math
    #    from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.quake_dialect import QuakeDialect
    #    from qrisp.jasp.mlir.quake_lowering.cc_dialect import CcDialect

    #    ctx = Context()
    #    # Load standard dialects that the MLIR string uses
    #    ctx.load_dialect(builtin.Builtin)
    #    ctx.load_dialect(func.Func)
    #    ctx.load_dialect(arith.Arith)
    #    ctx.load_dialect(math.Math)
    #    # Load custom dialects
    #    ctx.load_dialect(QuakeDialect)
    #    ctx.load_dialect(CcDialect)
    #    parser = Parser(ctx, mlir_input)
    #    module = parser.parse_module()
    # else:
    #    module = mlir_input

    # Get CUDA-Q naming from a dummy kernel
    kernel = cudaq.make_kernel()
    func_name = kernel.funcName
    entry_point = kernel.funcNameEntryPoint
    uniq_name = func_name.replace("__nvqpp__mlirgen__", "")

    # Get platform LLVM attributes
    data_layout_str, target_triple_str = _get_llvm_attributes()

    # Apply all structural passes (in-place on the xDSL module)
    prepare_module_for_cudaq(
        xdsl_module,
        func_name=func_name,
        entry_point=entry_point,
        uniq_name=uniq_name,
        data_layout=data_layout_str,
        target_triple=target_triple_str,
        execution_mode=execution_mode,
    )

    # Serialize to string exactly once, then normalize for CUDA-Q
    raw_mlir = str(xdsl_module)
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
# Convenience functions
# ------------------------------------------------------------------ #


def run_quake_mlir(xdsl_module: ModuleOp, shots: int = 100) -> list:
    """Execute a Quake MLIR string via cudaq.run."""
    pykd = cudaq_kernel_from_mlir(xdsl_module, execution_mode="run")
    return cudaq.run(pykd, shots_count=shots)


def sample_quake_mlir(xdsl_module: ModuleOp, shots: int = 100) -> dict[str, int]:
    """Execute a Quake MLIR string via cudaq.sample."""
    pykd = cudaq_kernel_from_mlir(xdsl_module, execution_mode="sample")
    result = cudaq.sample(pykd, shots_count=shots)
    return {key: value for key, value in result.items()}


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

    return cudaq_kernel_from_mlir(mlir_module, execution_mode=execution_mode)
