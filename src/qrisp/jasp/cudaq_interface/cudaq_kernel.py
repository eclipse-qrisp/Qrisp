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

# Custom ingestion pipeline: bridging external MLIR to CUDA-Q.
# ===============================================================
#
# Rationale
# ---------
# CUDA-Q's execution backend (C++) strictly requires an MLIR module to
# specify the host machine's exact memory architecture (llvm.data_layout)
# and hardware target (llvm.target_triple) to successfully compile and
# allocate memory. Currently, the CUDA-Q Python API lacks a native
# mechanism to cleanly ingest externally compiled MLIR strings. Omitting
# these hardware attributes results in fatal "missing data layout"
# runtime crashes.
#
# Approach
# --------
# 1. Target Extraction: We define an empty Python function decorated with
#    `@cudaq.kernel` to trigger the CUDA-Q compiler pipeline. This forces
#    the underlying LLVM compiler to generate the exact, natively-matched
#    layout and target triple for the host environment, which we then
#    extract via regular expressions. If that fails
#    (e.g. in CI environments where str() doesn't trigger full LLVM
#    lowering), we fall back to well-known platform defaults derived from
#    the host's architecture and OS.
#
# 2. Interface Adaptation (cudaq_ingestion/cudaq_prep.py):
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

from collections.abc import Callable
from typing import Literal
import inspect

from cudaq.kernel.kernel_decorator import PyKernelDecorator

from qrisp.jasp.evaluation_tools.profiler import profile_jaspr
from qrisp.jasp.interpreter_tools import jaspr_to_static_register_jaspr
from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.cudaq_interface.quake_lowering.jaspr_to_quake import jaspr_to_quake_mlir
from qrisp.jasp.cudaq_interface.annotations import FixedShapeNDArray
from qrisp.jasp.cudaq_interface.cudaq_ingestion import cudaq_kernel_from_xdsl_module


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
    register_size: int | None = None,
) -> PyKernelDecorator:
    """
    Decorator that compiles a Qrisp function to a native CUDA-Q kernel.

    Mirrors ``@cudaq.kernel`` exactly: the decorated name is bound directly
    to a ``PyKernelDecorator`` — compiled **eagerly at decoration time** —
    and can be passed to ``cudaq.run`` without calling it first.

    For functions with parameters, each parameter **must carry a type
    annotation**, just as ``@cudaq.kernel`` requires them.  The annotation
    is used to generate a correctly-typed dummy value for Jaspr tracing,
    producing parametric MLIR whose ``@main`` function retains the parameter
    in its signature.  The resulting kernel accepts runtime arguments via
    ``cudaq.run`` or direct calls.

    Supported annotations:

    * ``int``, ``float``, ``bool`` — scalar values passed directly.
    * :class:`FixedShapeNDArray` ``[dtype, size]`` — fixed-size NumPy array.
      Specify the element type (``float``, ``int``, or ``bool``) and the
      number of elements.  At runtime, pass a ``numpy.ndarray`` of the
      matching dtype and length.

    Parameters
    ----------
    func : callable, optional
        A Qrisp function that can be traced with ``make_jaspr``.  Parameters,
        if any, must be annotated with ``int``, ``float``, ``bool``, or
        :class:`FixedShapeNDArray`.
        The function may return ``int``, ``float``, ``bool``, or a tuple of those types.
        When ``None``, the decorator is used in its parameterised form
        (``@cudaq_kernel(execution_mode=...)``).
    execution_mode : Literal["run", "sample"], optional
        - ``"run"`` *(default)* — compile the kernel for use with ``cudaq.run``;
          measurement results are returned as classical values per shot.
        - ``"sample"`` — compile the kernel for use with ``cudaq.sample``;
          measurements are collected by the runtime across all shots and returned
          as a ``SampleResult`` histogram.
    register_size : int, optional
        If specified, the kernel will be compiled with a fixed register size.

    Returns
    -------
    PyKernelDecorator
        A compiled, callable CUDA-Q kernel bound to the decorated name.

    Raises
    ------
    RuntimeError
        If a parameter is missing a type annotation or has an unsupported
        annotation type.
    RuntimeError
        If tracing or lowering the function to CUDA-Q fails. This can
        happen when the kernel uses unsupported traced array arithmetic.

    Examples
    --------
    No-argument kernel — identical usage to ``@cudaq.kernel``::

        import cudaq
        from qrisp import *
        from qrisp import cudaq_kernel

        @cudaq_kernel
        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        print(bell())                            # single-shot, e.g. 0 or 3
        print(cudaq.run(bell, shots_count=100))  # multi-shot, no () needed

    Multiple returns are supported; they are returned as a single tuple::

        import cudaq
        from qrisp import *
        from qrisp import cudaq_kernel

        @cudaq_kernel
        def main():
            a = QuantumFloat(3)
            b = QuantumFloat(2)
            a[:] = 3
            h(b)
            a += b
            return measure(a), measure(b)

        print(cudaq.run(main, shots_count=5))
        # e.g. [(3.0, 0.0), (3.0, 0.0), (5.0, 2.0), (5.0, 2.0), (6.0, 3.0)]

    Parameterised kernel with scalar and array annotations::

        import cudaq
        import numpy as np
        from qrisp import *
        from qrisp import cudaq_kernel, FixedShapeNDArray

        @cudaq_kernel
        def circuit(k: int):
            qv = QuantumFloat(2)
            h(qv[0])
            return measure(qv[0]) + k

        print(circuit(3))
        print(cudaq.run(circuit, 3, shots_count=100))

        @cudaq_kernel
        def circuit_arr(angles: FixedShapeNDArray[float, 3]):
            qv = QuantumFloat(2)
            ry(angles[0], qv[0])
            return measure(qv[0])

        angles = np.array([1.57, 0.78, 0.39])
        print(circuit_arr(angles))
        print(cudaq.run(circuit_arr, angles, shots_count=100))

    Sample mode — use ``@cudaq_kernel(execution_mode="sample")`` for
    ``cudaq.sample`` (void-return kernel, measurements collected by runtime)::

        import cudaq
        from qrisp import *
        from qrisp import cudaq_kernel

        @cudaq_kernel(execution_mode="sample")
        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        print(cudaq.sample(bell, shots_count=100))

    """
    if func_arg is None:
        return lambda x: cudaq_kernel(x, execution_mode=execution_mode, register_size=register_size)

    sig = inspect.signature(func_arg)
    params = list(sig.parameters.values())
    _supported = list(_ANNOTATION_TO_DUMMY.keys()) + ["FixedShapeNDArray[dtype, size]"]

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

    jaspr = make_jaspr(func_arg)(*dummy_args)

    # NOTE: THe flowing code is commented out because:
    # 1. The conversion to static register will likely not be necessary with a future CUDA-Q version: https://github.com/NVIDIA/cuda-quantum/pull/4945
    # This can currently only be tested when installing CUDA-Q from source, as the latest release (0.15) does not include this change.
    # 2. Use of profiler for deciding whether to use static register allocation could break for certain edge cases.

    # try:
    #    qubits_dict = profile_jaspr(jaspr, "num_qubits", meas_behavior="0", max_allocations=1000)(*dummy_args)
    #    peak_allocations = qubits_dict.get("peak_allocations", 0)
    #    total_allocated = qubits_dict.get("total_allocated", 0)

    # Decide whether to use static register allocation based on peak vs total allocations.
    # If total allocated qubits exceed 110% of peak allocations, we use static register allocation to optimize memory usage.
    # Otherwise, we proceed with the original jaspr without static register allocation,
    # since CUDA-Q runtime is faster without static register reinterpretation.
    #    use_static_register = total_allocated > peak_allocations * 1.1
    # except ValueError:
    #    use_static_register = False

    # if use_static_register:
    #    static_reg_jaspr = jaspr_to_static_register_jaspr(jaspr, peak_allocations)
    #    new_jaspr = static_reg_jaspr
    # else:
    #    new_jaspr = jaspr

    if register_size is not None:
        new_jaspr = jaspr_to_static_register_jaspr(jaspr, register_size)
    else:
        new_jaspr = jaspr

    try:
        mlir_module = jaspr_to_quake_mlir(new_jaspr, execution_mode=execution_mode)
    except Exception as e:
        raise RuntimeError(f"Failed to compile Qrisp function '{func_arg.__name__}' to MLIR: {e}") from e

    return cudaq_kernel_from_xdsl_module(mlir_module, execution_mode=execution_mode)
