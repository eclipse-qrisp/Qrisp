# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Compile normalized xDSL MLIR into native CUDA-Q kernels."""

# Normalizing xDSL-printed MLIR and compiling it into a native CUDA-Q
# kernel (pipeline stage 3 – "Re-Compilation").
# =======================================================================

import re
from typing import Literal

import cudaq
from cudaq.kernel.kernel_decorator import PyKernelDecorator
from cudaq.mlir.dialects import cc as cudaq_cc_dialect
from cudaq.mlir.dialects import quake as cudaq_quake_dialect
from cudaq.mlir.ir import Module, NoneType
from xdsl.dialects.builtin import ModuleOp

from qrisp.jasp.cudaq_interface.cudaq_ingestion.cudaq_prep import (
    _CudaqPreparationConfig,
    _prepare_module_for_cudaq,
)
from qrisp.jasp.cudaq_interface.cudaq_ingestion.host_attributes import _get_llvm_attributes

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

# ``cudaq.make_kernel()`` derives its "unique" kernel name from ``id(self)``
# (the Python object's memory address, see CUDA-Q's ``PyKernel.__init__``).
# If the throwaway ``PyKernel`` instance created below is allowed to be
# garbage-collected, CPython can reuse its address for a later, unrelated
# ``cudaq.make_kernel()`` call, producing a duplicate kernel/symbol name and
# causing CUDA-Q's native runtime to crash (native abort) when launching a
# kernel whose name collides with a different, already-registered module.
# Keeping a permanent reference to every such kernel object prevents its
# ``id()`` from ever being reused, guaranteeing name uniqueness for the
# lifetime of the process.
_KERNEL_NAME_KEEPALIVE: list = []


def _cudaq_kernel_from_xdsl_module(
    xdsl_module: ModuleOp,
    execution_mode: Literal["run", "sample"] = "run",
) -> PyKernelDecorator:
    """Compiles an xDSL ModuleOp into a native PyKernelDecorator.

    The input MLIR must define a ``@main`` function with a
    ``cudaq-entrypoint`` attribute.  The function may optionally return a
    value (e.g. an ``i64`` measurement result).

    The MLIR is expected to already have array parameters in
    ``!cc.stdvec<T>`` form (as produced by ``array_to_stdvec``).

    The returned kernel is a first-class CUDA-Q kernel object and supports
    all standard CUDA-Q execution patterns:

    * ``kernel()`` — single-shot execution, returning the measurement result.
    * ``cudaq.run(kernel, shots_count=N)`` — multi-shot sampling.
    * ``cudaq.sample(kernel, shots_count=N)`` — histogram sampling
      (requires ``execution_mode="sample"``).

    Parameters
    ----------
    xdsl_module : ModuleOp
        An xDSL module representing the quantum computation in Quake and CC dialects.
        Must contain a ``@main`` function with the ``cudaq-entrypoint`` attribute.
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
    cudaq.kernel.kernel_decorator.PyKernelDecorator
        A compiled, callable CUDA-Q kernel.

    Examples
    --------
    ::

        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp import make_jaspr
        from qrisp.jasp.cudaq_interface.quake_lowering import jaspr_to_quake_mlir
        from qrisp.jasp.cudaq_interface import cudaq_kernel_from_xdsl_module
        import cudaq

        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        jaspr = make_jaspr(bell)()
        xdsl_module = jaspr.to_quake_mlir()
        kernel = cudaq_kernel_from_xdsl_module(xdsl_module)

        print(kernel())                          # single-shot, e.g. 0 or 3
        print(cudaq.run(kernel, shots_count=100))

    """
    module = xdsl_module.clone()

    # Get CUDA-Q naming from a dummy kernel. This kernel object must never be
    # garbage-collected (see _KERNEL_NAME_KEEPALIVE above), otherwise its
    # id()-derived name could later collide with another kernel's name.
    kernel = cudaq.make_kernel()
    _KERNEL_NAME_KEEPALIVE.append(kernel)
    func_name = kernel.funcName
    entry_point = kernel.funcNameEntryPoint
    uniq_name = func_name.replace("__nvqpp__mlirgen__", "")

    # Get platform LLVM attributes
    data_layout_str, target_triple_str = _get_llvm_attributes()

    # Apply all structural passes (in-place on the xDSL module)
    _prepare_module_for_cudaq(
        module,
        _CudaqPreparationConfig(
            func_name=func_name,
            entry_point=entry_point,
            unique_name=uniq_name,
            data_layout=data_layout_str,
            target_triple=target_triple_str,
            execution_mode=execution_mode,
        ),
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
