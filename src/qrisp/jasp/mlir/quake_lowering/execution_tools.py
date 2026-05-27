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

from qrisp.jasp import make_jaspr
from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake

# ------------------------------------------------------------------ #
# Platform-aware LLVM attribute defaults
# ------------------------------------------------------------------ #

# These are the standard LLVM data layouts and target triples for
# platforms commonly used with CUDA-Q.  They describe the host's
# classical memory architecture and are deterministic per (arch, OS).
_PLATFORM_DEFAULTS = {
    # (machine, sys.platform prefix) → (data_layout, target_triple)
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
        f'llvm.target_triple = "{target_triple_str}"' if target_triple_str else None
    )
    return data_layout_attr, target_triple_attr


# ------------------------------------------------------------------ #
# FixedShapeNDArray annotation
# ------------------------------------------------------------------ #


class FixedShapeNDArray:
    """Type annotation for a fixed-size numpy array parameter in
    ``@qrisp_cudaq_kernel``.

    Mirrors the role of ``list[float]`` / ``list[int]`` in ``@cudaq.kernel``
    but requires an explicit size, because JAX/Jaspr traces with concrete
    shapes.  The decorator uses the element type and size to generate a
    correctly-typed dummy value for tracing, producing parametric MLIR
    whose parameter type is then rewritten from the static
    ``!cc.ptr<!cc.array<T x N>>`` to the CUDA-Q-marshaling-compatible
    ``!cc.stdvec<T>``.

    Parameters
    ----------
    dtype : type
        Element type.  Supported: ``float`` (→ ``f64``), ``int`` (→ ``i64``),
        ``bool`` (→ ``i1``).
    size : int
        Number of elements.  Must match the runtime array length.

    Examples
    --------
    ::

        @qrisp_cudaq_kernel
        def circuit(angles: FixedShapeNDArray(float, 3)):
            qv = QuantumFloat(2)
            ry(angles[0], qv[0])
            return measure(qv[0])

        angles = np.array([1.57, 0.78, 0.39])
        print(cudaq.run(circuit, angles, shots_count=100))

    """

    #: Maps Python dtype to NumPy dtype and MLIR element type string.
    _DTYPE_MAP = {
        float: (np.float64, "f64"),
        int: (np.int64, "i64"),
        bool: (np.bool_, "i1"),
    }

    def __init__(self, dtype: type, size: int):
        if dtype not in self._DTYPE_MAP:
            raise TypeError(
                f"FixedShapeNDArray: unsupported dtype '{dtype}'. "
                f"Supported: {list(self._DTYPE_MAP.keys())}."
            )
        if not isinstance(size, int) or size <= 0:
            raise ValueError("FixedShapeNDArray: size must be a positive integer.")
        self.dtype = dtype
        self.size = size
        self._np_dtype, self.mlir_elem_type = self._DTYPE_MAP[dtype]

    def make_dummy(self) -> np.ndarray:
        """Return a zero-filled NumPy array of the correct dtype and size."""
        return np.zeros(self.size, dtype=self._np_dtype)

    def __repr__(self):
        return f"FixedShapeNDArray({self.dtype.__name__}, {self.size})"


# ------------------------------------------------------------------ #
# MLIR rewriting pass: static array params → stdvec
# ------------------------------------------------------------------ #


def _rewrite_array_params_to_stdvec(
    param_list: str, body: str
) -> tuple:
    """Rewrite ``!cc.ptr<!cc.array<T x N>>`` parameters to ``!cc.stdvec<T>``.

    Jaspr traces numpy arrays as fixed-size C-array pointers.  CUDA-Q's
    Python marshaling layer only understands ``!cc.stdvec<T>`` for
    list / array arguments passed from Python.  This pass:

    1. Replaces the parameter type in the function signature.
    2. Inserts a ``cc.stdvec_data`` instruction at the top of the body to
       obtain a raw ``!cc.ptr<!cc.array<T x ?>>`` from the stdvec.
    3. Rewrites all ``cc.compute_ptr`` references that used the argument
       directly to use the new raw-pointer SSA value, and updates their
       type annotations from the static ``T x N`` to the dynamic ``T x ?``.

    Parameters
    ----------
    param_list : str
        The ``(...)`` parameter list text extracted from the ``@main`` header.
    body : str
        The function body text (content between the outer braces).

    Returns
    -------
    (new_param_list, new_body)
        Rewritten strings ready to be injected into the adapted MLIR.
    """
    # Find all parameters with static array types.
    array_params = re.findall(
        r"(%arg\w+)\s*:\s*!cc\.ptr<!cc\.array<([a-z0-9]+)\s*x\s*(\d+)>>",
        param_list,
    )
    if not array_params:
        return param_list, body

    new_param_list = param_list
    preamble_lines = []
    new_body = body

    for arg_name, elem_type, size in array_params:
        static_type = f"!cc.ptr<!cc.array<{elem_type} x {size}>>"
        stdvec_type = f"!cc.stdvec<{elem_type}>"
        dyn_ptr_type = f"!cc.ptr<!cc.array<{elem_type} x ?>>"
        data_var = f"{arg_name}_ptr"

        # 1. Rewrite signature
        new_param_list = new_param_list.replace(
            f"{arg_name}: {static_type}",
            f"{arg_name}: {stdvec_type}",
        )

        # 2. Preamble: extract raw pointer from the stdvec
        preamble_lines.append(
            f"{data_var} = cc.stdvec_data {arg_name} "
            f": ({stdvec_type}) -> {dyn_ptr_type}"
        )

        # 3. Redirect compute_ptr from %argN to %argN_ptr
        new_body = re.sub(
            rf"cc\.compute_ptr\s+{re.escape(arg_name)}\[",
            f"cc.compute_ptr {data_var}[",
            new_body,
        )

        # 4. Update the static type annotations in compute_ptr calls
        #    (!cc.ptr<!cc.array<T x N>>, ...) → (!cc.ptr<!cc.array<T x ?>>, ...)
        new_body = new_body.replace(
            f"({static_type},", f"({dyn_ptr_type},"
        )
        new_body = new_body.replace(
            f"({static_type})", f"({dyn_ptr_type})"
        )

    if preamble_lines:
        preamble_str = "\n    ".join(preamble_lines)
        new_body = preamble_str + "\n    " + new_body

    return new_param_list, new_body


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #


def cudaq_kernel_from_mlir(mlir_str: str) -> PyKernelDecorator:
    """
    Compiles a Quake MLIR string into a native ``cudaq.kernel.kernel_decorator.PyKernelDecorator``.

    The input MLIR must define a ``@main`` function with a ``cudaq-entrypoint``
    attribute.  The function may optionally return a value (e.g. an ``i64``
    measurement result).

    The returned kernel is a first-class CUDA-Q kernel object and supports all
    standard CUDA-Q execution patterns:

    * ``kernel()`` — single-shot execution via
      ``cudaq_runtime.marshal_and_launch_module``, returning the measurement
      result directly.
    * ``cudaq.run(kernel, shots_count=N)`` — multi-shot sampling using
      CUDA-Q's native run infrastructure (requires the ``.run`` variant
      injected during compilation).

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
        from qrisp.jasp.mlir.quake_lowering import jaspr_to_quake, cudaq_kernel_from_mlir
        import cudaq

        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        mlir_str = str(jaspr_to_quake(make_jaspr(bell)()))
        kernel = cudaq_kernel_from_mlir(mlir_str)

        # Single-shot call
        print(kernel())            # e.g. 0 or 3

        # Multi-shot sampling
        print(cudaq.run(kernel, shots_count=100))

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
    func_start = mlir_str.find("func.func")
    last_brace = mlir_str.rfind("}")
    inner_mlir = mlir_str[func_start:last_brace].strip()

    # Locate the @main function signature specifically
    main_match = re.search(r"func\.func\s+(?:public\s+)?@main", inner_mlir)
    if not main_match:
        raise ValueError("Could not find @main function in MLIR string.")

    # Robustly find the function body, skipping optional `attributes { ... }`
    anchor = main_match.end()
    search_idx = anchor
    body_start_idx = -1

    while search_idx < len(inner_mlir):
        if inner_mlir[search_idx] == "{":
            text_before = inner_mlir[anchor:search_idx].strip()
            if text_before.endswith("attributes"):
                # This is an attributes block, skip it
                depth = 1
                search_idx += 1
                while search_idx < len(inner_mlir) and depth > 0:
                    if inner_mlir[search_idx] == "{":
                        depth += 1
                    elif inner_mlir[search_idx] == "}":
                        depth -= 1
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
        if inner_mlir[i] == "{":
            depth += 1
        elif inner_mlir[i] == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    main_func_body = inner_mlir[body_start_idx + 1 : end_idx].strip()

    # Keep any other functions (like @closed_call) perfectly intact
    other_functions = inner_mlir[: main_match.start()] + inner_mlir[end_idx + 1 :]

    # ------------------------------------------------------------------ #
    # Step 2b: Extract the parameter list from the @main header.
    # ------------------------------------------------------------------ #
    # The header text between @main and the body { contains the parameter
    # list, optional return type, and optional attributes block, e.g.:
    #   (%0: i64) -> (i64) attributes {cudaq.kernel = "true", ...}
    # We extract the first balanced-parentheses group as the param list.
    # It must be preserved in ALL three adapted variants so that:
    #  (a) the main function body's SSA values (%0 etc.) are valid, and
    #  (b) cudaq.run(kernel, arg, shots_count=N) can pass arguments through.
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

    # ------------------------------------------------------------------ #
    # Step 2c: Rewrite static array parameters to !cc.stdvec<T>.
    # ------------------------------------------------------------------ #
    # Jaspr traces numpy arrays as !cc.ptr<!cc.array<T x N>> (fixed-size
    # C-array pointers).  CUDA-Q's marshaling layer only understands
    # !cc.stdvec<T> for list/array arguments, so we rewrite the signature
    # and element-access instructions here before assembly.
    param_list, main_func_body = _rewrite_array_params_to_stdvec(
        param_list, main_func_body
    )
    run_body = main_func_body  # run_body is derived from main_func_body below

    # ------------------------------------------------------------------ #
    # Step 3: Determine the return type and build the ".run" variant.
    # ------------------------------------------------------------------ #
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
        # run_body already set to main_func_body in Step 2c

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
        f"quake.mangled_name_map = {{\n"
        f'    {func_name} = "{entry_point}",\n'
        f'    {run_func_name} = "{run_entry_name}"\n'
        f"  }}"
    )
    attributes_str = ",\n  ".join(mod_attributes)

    adapted_mlir = f"module attributes {{\n  {attributes_str}\n}} {{\n"

    # 1. Inject all helper functions first (e.g. @closed_call)
    if other_functions.strip():
        adapted_mlir += f"  {other_functions.strip()}\n\n"

    # 2. Inject the main function and rename it to the CUDA-Q target name
    adapted_mlir += (
        f"  func.func @{func_name}{param_list}{return_sig} attributes "
        f'{{"cudaq-entrypoint", "cudaq-kernel"}} {{\n    {main_func_body}\n  }}\n'
    )

    # 3. Inject the .run variant for state measurements
    run_attrs = f'{{"cudaq-entrypoint", "cudaq-kernel", no_this'
    if return_type_str:
        run_attrs += f", quake.cudaq_run = [{return_type_str}]"
    run_attrs += "}"
    adapted_mlir += f"  func.func @{run_func_name}{param_list} attributes {run_attrs} {{\n    {run_body}\n  }}\n"

    # 4. Inject the dummy entry (also parameterised to match the mangled_name_map)
    adapted_mlir += (
        f"  func.func @{run_entry_name}{param_list} attributes {{no_this}} {{\n    return\n  }}\n"
        f"}}\n"
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
    # result = cudaq_runtime.run_impl(
    #    uniq_name + ".run",
    #    kernel.module,
    #    ret_type,
    #    shots,
    #    None,   # noise_model
    #    0       # qpu_id
    # )

    # ------------------------------------------------------------------ #
    # Step 7: Wrap the compiled module in a genuine PyKernelDecorator.
    # ------------------------------------------------------------------ #
    # PyKernelDecorator(None, kernelName=..., module=...) uses the
    # `module` construction path: it skips Python AST compilation, stores
    # the pre-parsed MLIR directly, and parses the signature (including
    # return type) from the MLIR via KernelSignature.parse_from_mlir.
    # Calling the resulting kernel routes through
    # cudaq_runtime.marshal_and_launch_module — the correct dispatcher for
    # non-void entry points — instead of the raw PyKernel builder's
    # pyAltLaunchKernel path which requires a void entry.
    #
    # cudaq.run(pykd, shots_count=N) also works because the adapted MLIR
    # contains the .run variant with cc.log_output that cudaq.run expects.
    pykd = PyKernelDecorator(None, kernelName=uniq_name, module=kernel.module)
    return pykd


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
    pykd = cudaq_kernel_from_mlir(mlir_str)
    result = cudaq.run(pykd, shots_count=shots)
    return result


# ------------------------------------------------------------------ #
# Decorator for user-friendly API
# ------------------------------------------------------------------ #


# Maps Python type annotations to dummy argument values for Jaspr tracing.
# The concrete type of the dummy value determines the MLIR type in @main's
# parameter list (e.g. int → i64, float → f64, bool → i1).
# FixedShapeNDArray instances are handled separately via FixedShapeNDArray.make_dummy().
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
      number of elements.

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

        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp.mlir.quake_lowering import qrisp_cudaq_kernel
        import cudaq

        @qrisp_cudaq_kernel
        def bell():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        print(bell())                            # single-shot, e.g. 0 or 3
        print(cudaq.run(bell, shots_count=100))  # multi-shot, no () needed

    Parameterised kernel with scalar and array type annotations::

        @qrisp_cudaq_kernel
        def circuit(k: int):
            qv = QuantumFloat(2)
            h(qv[0])
            return measure(qv[0]) + k

        print(circuit(3))                             # single-shot
        print(cudaq.run(circuit, 3, shots_count=100)) # multi-shot

        @qrisp_cudaq_kernel
        def circuit_arr(angles: FixedShapeNDArray(float, 3)):
            qv = QuantumFloat(2)
            ry(angles[0], qv[0])
            return measure(qv[0])

        angles = np.array([1.57, 0.78, 0.39])
        print(circuit_arr(angles))                             # single-shot
        print(cudaq.run(circuit_arr, angles, shots_count=100)) # multi-shot

    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    _supported = list(_ANNOTATION_TO_DUMMY.keys()) + ["FixedShapeNDArray(dtype, size)"]

    # Validate annotations for all parameters and build dummy args
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
