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
# qrisp.jasp.cudaq_interface.cudaq_kernel
# CUDA-Q backend implementation for Qrisp/Jasp.
# ====================================================================== #
# Bridges Qrisp-generated Quake MLIR to the CUDA-Q execution runtime.
#
# Responsibilities (all CUDA-Q-specific, not generic MLIR lowering):
#  - Injecting LLVM data-layout / target-triple attributes required by the
#    CUDA-Q C++ backend.
#  - Synthesising the .run / .run.entry function variants that CUDA-Q's
#    shot-based sampling infrastructure expects.
#  - Wrapping the compiled module in a genuine PyKernelDecorator so the
#    result behaves exactly like a @cudaq.kernel-decorated function.
#  - Providing the @qrisp_cudaq_kernel user-facing decorator.
#
# The array-param → stdvec rewriting is now handled by pass4_array_to_stdvec
# at the xDSL IR level, before the MLIR string is produced.
#
# Public API is re-exported via qrisp.jasp.cudaq_interface.__init__.
# cudaq is an optional dependency; an ImportError here is caught by callers.
# ====================================================================== #

import inspect
import platform
import re
import sys
import warnings

import numpy as np

import cudaq
from cudaq import cudaq_runtime
from cudaq.kernel.kernel_decorator import PyKernelDecorator
from cudaq.mlir.ir import Module, NoneType
from cudaq.mlir.dialects import quake, cc

# Use direct sub-module imports (not qrisp.jasp.*) to avoid circular
# imports: qrisp.jasp.__init__ imports evaluation_tools which optionally
# imports qrisp.jasp.cudaq_interface, so qrisp.jasp is not yet fully initialised here.
from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.mlir.quake_lowering.jaspr_to_quake import jaspr_to_quake
from qrisp.jasp.cudaq_interface.annotations import FixedShapeNDArray  

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
        f'llvm.target_triple = "{target_triple_str}"' if target_triple_str else None
    )
    return data_layout_attr, target_triple_attr


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #


def cudaq_kernel_from_mlir(mlir_str: str) -> PyKernelDecorator:
    """
    Compiles a Quake MLIR string into a native ``PyKernelDecorator``.

    The input MLIR must define a ``@main`` function with a
    ``cudaq-entrypoint`` attribute.  The function may optionally return a
    value (e.g. an ``i64`` measurement result).

    The MLIR is expected to already have array parameters in
    ``!cc.stdvec<T>`` form (as produced by pass4_array_to_stdvec).

    The returned kernel is a first-class CUDA-Q kernel object and supports
    all standard CUDA-Q execution patterns:

    * ``kernel()`` — single-shot execution, returning the measurement result.
    * ``cudaq.run(kernel, shots_count=N)`` — multi-shot sampling.

    Parameters
    ----------
    mlir_str : str
        Quake MLIR source string.  Must contain a ``@main`` function with
        the ``cudaq-entrypoint`` attribute.

    Returns
    -------
    cudaq.kernel.kernel_decorator.PyKernelDecorator
        A compiled, callable CUDA-Q kernel.

    Examples
    --------
    ::

        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp import make_jaspr
        from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake
        from qrisp.jasp.cudaq_interface import cudaq_kernel_from_mlir
        import cudaq

        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        mlir_str = str(jaspr_to_quake(make_jaspr(bell)()))
        kernel = cudaq_kernel_from_mlir(mlir_str)

        print(kernel())                          # single-shot, e.g. 0 or 3
        print(cudaq.run(kernel, shots_count=100))

    """
    kernel = cudaq.make_kernel()
    func_name = kernel.funcName
    entry_point = kernel.funcNameEntryPoint
    uniq_name = func_name.replace("__nvqpp__mlirgen__", "")

    data_layout_attr, target_triple_attr = _get_llvm_attributes()

    func_start = mlir_str.find("func.func")
    last_brace = mlir_str.rfind("}")
    inner_mlir = mlir_str[func_start:last_brace].strip()

    main_match = re.search(r"func\.func\s+(?:public\s+)?@main", inner_mlir)
    if not main_match:
        raise ValueError("Could not find @main function in MLIR string.")

    anchor = main_match.end()
    search_idx = anchor
    body_start_idx = -1

    while search_idx < len(inner_mlir):
        if inner_mlir[search_idx] == "{":
            text_before = inner_mlir[anchor:search_idx].strip()
            if text_before.endswith("attributes"):
                depth = 1
                search_idx += 1
                while search_idx < len(inner_mlir) and depth > 0:
                    if inner_mlir[search_idx] == "{":
                        depth += 1
                    elif inner_mlir[search_idx] == "}":
                        depth -= 1
                    search_idx += 1
                anchor = search_idx
                continue
            else:
                body_start_idx = search_idx
                break
        search_idx += 1

    if body_start_idx == -1:
        raise ValueError("Could not find body for @main.")

    depth = 0
    end_idx = -1
    for i in range(body_start_idx, len(inner_mlir)):
        if inner_mlir[i] == "{":
            depth += 1
        elif inner_mlir[i] == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    main_func_body = inner_mlir[body_start_idx + 1 : end_idx].strip()
    other_functions = inner_mlir[: main_match.start()] + inner_mlir[end_idx + 1 :]

    header_raw = inner_mlir[main_match.end() : body_start_idx]
    first_paren = header_raw.find("(")
    param_list = ""
    if first_paren != -1:
        depth = 0
        for i, ch in enumerate(header_raw[first_paren:], first_paren):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    param_list = header_raw[first_paren : i + 1]
                    break

    run_body = main_func_body

    return_match = re.search(r"func\.return\s+(%\w+)\s*:\s*(.+)", main_func_body)
    if return_match:
        return_type_str = return_match.group(2).strip()
        return_sig = f" -> {return_type_str}"
        run_body = re.sub(
            r"func\.return\s+(%\w+)\s*:\s*(.+)",
            r"cc.log_output \1 : \2\n    return",
            main_func_body,
        )
    else:
        return_type_str = None
        return_sig = ""

    run_func_name = func_name + ".run"
    run_entry_name = func_name + ".run.entry"

    mod_attributes = [f'cc.python_uniqued = "{uniq_name}"']
    mod_attributes.append(data_layout_attr)
    if target_triple_attr:
        mod_attributes.append(target_triple_attr)

    mod_attributes.append(
        f"quake.mangled_name_map = {{\n"
        f'    {func_name} = "{entry_point}",\n'
        f'    {run_func_name} = "{run_entry_name}"\n'
        f"  }}"
    )
    attributes_str = ",\n  ".join(mod_attributes)

    adapted_mlir = f"module attributes {{\n  {attributes_str}\n}} {{\n"

    if other_functions.strip():
        adapted_mlir += f"  {other_functions.strip()}\n\n"

    adapted_mlir += (
        f"  func.func @{func_name}{param_list}{return_sig} attributes "
        f'{{"cudaq-entrypoint", "cudaq-kernel"}} {{\n    {main_func_body}\n  }}\n'
    )

    run_attrs = f'{{"cudaq-entrypoint", "cudaq-kernel", no_this'
    if return_type_str:
        run_attrs += f", quake.cudaq_run = [{return_type_str}]"
    run_attrs += "}"
    adapted_mlir += f"  func.func @{run_func_name}{param_list} attributes {run_attrs} {{\n    {run_body}\n  }}\n"

    adapted_mlir += (
        f"  func.func @{run_entry_name}{param_list} attributes {{no_this}} {{\n    return\n  }}\n"
        f"}}\n"
    )

    with kernel.ctx:
        try:
            new_module = Module.parse(adapted_mlir, kernel.ctx)
        except Exception:
            quake.register_dialect(context=kernel.ctx)
            cc.register_dialect(context=kernel.ctx)
            new_module = Module.parse(adapted_mlir, kernel.ctx)

        kernel.module = new_module
        NoneType.get(context=kernel.ctx)

    pykd = PyKernelDecorator(None, kernelName=uniq_name, module=kernel.module)
    return pykd


def run_quake_mlir(mlir_str: str, shots: int = 100) -> list:
    """
    Executes a Quake MLIR string on the CUDA-Q runtime.

    Parameters
    ----------
    mlir_str : str
        Quake MLIR source.  Must contain a ``@main`` function with the
        ``cudaq-entrypoint`` attribute.
    shots : int
        Number of shots.  Default is 100.

    Returns
    -------
    list
        Measurement results from all shots.

    Examples
    --------
    ::

        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp import make_jaspr
        from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake
        from qrisp.jasp.cudaq_interface import run_quake_mlir

        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        result = run_quake_mlir(str(jaspr_to_quake(make_jaspr(bell)())), shots=10)
        print(result)  # e.g. [0, 0, 3, 0, 3, 0, 3, 3, 0, 0]

    """
    pykd = cudaq_kernel_from_mlir(mlir_str)
    return cudaq.run(pykd, shots_count=shots)


# ------------------------------------------------------------------ #
# @qrisp_cudaq_kernel decorator
# ------------------------------------------------------------------ #

_ANNOTATION_TO_DUMMY = {
    int: 0,
    float: 0.0,
    bool: False,
}


def qrisp_cudaq_kernel(func):
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
    * :class:`FixedShapeNDArray` ``(dtype, size)`` — fixed-size NumPy array.
      Specify the element type (``float``, ``int``, or ``bool``) and the
      number of elements.  At runtime, pass a ``numpy.ndarray`` of the
      matching dtype and length.

    Parameters
    ----------
    func : callable
        A Qrisp function that can be traced with ``make_jaspr``.  Parameters,
        if any, must be annotated with ``int``, ``float``, ``bool``, or
        :class:`FixedShapeNDArray`.

    Returns
    -------
    PyKernelDecorator
        A compiled, callable CUDA-Q kernel bound to the decorated name.

    Raises
    ------
    RuntimeError
        If a parameter is missing a type annotation or has an unsupported
        annotation type.

    Examples
    --------
    No-argument kernel — identical usage to ``@cudaq.kernel``::

        import cudaq
        from qrisp import *
        from qrisp.jasp.cudaq_interface import qrisp_cudaq_kernel

        @qrisp_cudaq_kernel
        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        print(bell())                            # single-shot, e.g. 0 or 3
        print(cudaq.run(bell, shots_count=100))  # multi-shot, no () needed

    Parameterised kernel with scalar and array annotations::

        import cudaq
        import numpy as np
        from qrisp import *
        from qrisp.jasp.cudaq_interface import qrisp_cudaq_kernel, FixedShapeNDArray

        @qrisp_cudaq_kernel
        def circuit(k: int):
            qv = QuantumFloat(2)
            h(qv[0])
            return measure(qv[0]) + k

        print(circuit(3))
        print(cudaq.run(circuit, 3, shots_count=100))

        @qrisp_cudaq_kernel
        def circuit_arr(angles: FixedShapeNDArray(float, 3)):
            qv = QuantumFloat(2)
            ry(angles[0], qv[0])
            return measure(qv[0])

        angles = np.array([1.57, 0.78, 0.39])
        print(circuit_arr(angles))
        print(cudaq.run(circuit_arr, angles, shots_count=100))

    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    _supported = list(_ANNOTATION_TO_DUMMY.keys()) + ["FixedShapeNDArray(dtype, size)"]

    dummy_args = []
    for p in params:
        if p.annotation is inspect.Parameter.empty:
            raise RuntimeError(
                f"@qrisp_cudaq_kernel: parameter '{p.name}' of "
                f"'{func.__name__}' requires a type annotation. "
                f"Supported: {_supported}."
            )
        if isinstance(p.annotation, FixedShapeNDArray):
            dummy_args.append(p.annotation.make_dummy())
        elif p.annotation in _ANNOTATION_TO_DUMMY:
            dummy_args.append(_ANNOTATION_TO_DUMMY[p.annotation])
        else:
            ann_name = getattr(p.annotation, "__name__", repr(p.annotation))
            raise RuntimeError(
                f"@qrisp_cudaq_kernel: unsupported annotation "
                f"'{ann_name}' for parameter '{p.name}' of "
                f"'{func.__name__}'. Supported: {_supported}."
            )

    try:
        mlir_str = str(jaspr_to_quake(make_jaspr(func)(*dummy_args)))
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile Qrisp function '{func.__name__}' to MLIR: {e}"
        )
    return cudaq_kernel_from_mlir(mlir_str)
