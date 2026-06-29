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

"""
Tests for the Jasp → Quake (memory-semantics) lowering backend.

Coverage
--------
- Basic quantum circuit (alloc, gate, measure, dealloc).
- Parameterized gates (rz, rx, u3).
- Controlled gates (cx, ccx).
- Reset operation.
- SCF control-flow lowering (jrange loop).
- Interface invariant: no ``!jasp.*`` types in the output.
- Negative test: ``jasp.parity`` is left in place (not lowered).
- Negative test: unsupported gate emits a warning and is left in place.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    QuantumVariable,
    QuantumBool,
    QuantumFloat,
    h,
    x,
    cx,
    rx,
    ry,
    rz,
    rz,
    measure,
    control,
)
from qrisp.jasp import q_while_loop, q_cond, qache

try:
    import cudaq
    from qrisp.jasp.cudaq_interface import (
        cudaq_kernel,
        FixedShapeNDArray,
    )
except ImportError as exc:
    # Skip the entire test file if the import fails
    pytest.skip(f"cudaq unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test qrisp cudaq kernel decorator
# ---------------------------------------------------------------------------


def test_cudaq_kernel_run():
    """Test that a simple @cudaq_kernel compiles and runs, returning a valid measurement outcome."""

    @cudaq_kernel
    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    result = bell()
    assert result in {0, 3}

    results = cudaq.run(bell, shots_count=10)
    for outcome in results:
        assert outcome in {0, 3}


def test_cudaq_kernel_run_nested_function():
    """Test that a @cudaq_kernel with a nested function with return value compiles and runs, returning valid measurement outcomes."""

    @qache
    def inner(qv):
        x(qv[0])
        cx(qv[0], qv[1])
        return 1

    @cudaq_kernel
    def main():
        qv = QuantumVariable(2)
        val = inner(qv)
        with control(val > 0):
            x(qv[0])
        return measure(qv)

    results = cudaq.run(main, shots_count=10)
    for outcome in results:
        assert outcome == 2


def test_cudaq_kernel_sample():
    """Test that a simple @cudaq_kernel compiles and samples, returning a valid measurement outcome."""

    @cudaq_kernel(execution_mode="sample")
    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        measure(qv)

    results = cudaq.sample(bell, shots_count=10)
    for outcome in results:
        assert outcome in {"00", "11"}


def test_cudaq_kernel_sample_nested_function():
    """Test that a @cudaq_kernel with a nested function with return value compiles and samples, returning valid measurement outcomes."""

    @qache
    def inner(qv):
        x(qv[0])
        cx(qv[0], qv[1])
        return 1

    @cudaq_kernel(execution_mode="sample")
    def main():
        qv = QuantumVariable(2)
        val = inner(qv)
        with control(val > 0):
            x(qv[0])
        measure(qv)

    results = cudaq.sample(main, shots_count=10)
    for outcome in results:
        assert outcome == "01"


def test_cudaq_kernel_multiple_returns():
    """Test that a @cudaq_kernel can return multiple values of different types."""

    @cudaq_kernel
    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])

        qb = QuantumBool()

        return measure(qv), measure(qb), 1.5

    result = bell()
    assert result in {(0, False, 1.5), (0, True, 1.5), (3, False, 1.5), (3, True, 1.5)}


def test_cudaq_kernel_algorithm():
    """Test that a more complex algorithm can be expressed in a @cudaq_kernel and executed."""
    from qrisp.operators import X, Y, Z
    import networkx as nx

    @cudaq_kernel
    def main():

        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        H = sum(X(i) for i in G.nodes) + sum(Z(i) * Z(j) for i, j in G.edges)
        U = H.trotterization()

        a = QuantumFloat(4)
        b = QuantumFloat(4)

        a[:] = 5
        h(b)
        U(a, t=1.0, steps=10)

        # Real-time control flow based on measurement results
        c = measure(b) < 5
        with control(c):
            a += 1

        return measure(a)

    result = main()
    assert result is not None


# ---------------------------------------------------------------------------
# Scalar parameter type tests
# ---------------------------------------------------------------------------


def test_int_parameter():
    """Kernel with int parameter: result is measure + classical offset."""

    @cudaq_kernel
    def int_kernel(k: int):
        qv = QuantumFloat(2)
        h(qv[0])
        return measure(qv[0]) + k

    # ry(0) leaves qv[0] in |0>, but h puts it in superposition → 0 or 1; +5
    result = int_kernel(5)
    assert result in {5, 6}


def test_float_parameter():
    """Kernel with float rotation angle: ry(0.0) deterministically leaves qubit in |0>."""

    @cudaq_kernel
    def float_kernel(angle: float):
        qv = QuantumFloat(1)
        ry(angle, qv[0])
        return measure(qv[0])

    assert float_kernel(0.0) == 0


def test_bool_parameter():
    """Kernel with bool parameter: flips qubit if True, leaves in |0> if False."""

    @cudaq_kernel
    def bool_kernel(flag: bool):
        qv = QuantumVariable(1)
        with control(flag):
            x(qv[0])
        return measure(qv)

    assert bool_kernel(True) == 1
    assert bool_kernel(False) == 0


# ---------------------------------------------------------------------------
# FixedShapeNDArray parameter tests
# ---------------------------------------------------------------------------


def test_fixed_shape_ndarray_float():
    """FixedShapeNDArray(float, 3): zero angles leave qubit in |0> deterministically."""

    @cudaq_kernel
    def float_arr_kernel(angles: FixedShapeNDArray(float, 3)):
        qv = QuantumFloat(1)
        ry(angles[0], qv[0])
        return measure(qv[0])

    assert float_arr_kernel(np.zeros(3)) == 0


def test_fixed_shape_ndarray_int():
    """FixedShapeNDArray(int, 2): integer element added to measurement."""

    @cudaq_kernel
    def int_arr_kernel(offsets: FixedShapeNDArray(int, 2)):
        qv = QuantumFloat(2)
        h(qv[0])
        return measure(qv[0]) + offsets[0]

    result = int_arr_kernel(np.array([10, 20], dtype=np.int64))
    assert result in {10, 11}


# ---------------------------------------------------------------------------
# Multiple parameters / multi-shot tests
# ---------------------------------------------------------------------------


def test_multiple_scalar_params():
    """Kernel with mixed scalar parameters (float angle + int offset)."""

    @cudaq_kernel
    def multi_kernel(angle: float, k: int):
        qv = QuantumFloat(2)
        ry(angle, qv[0])
        return measure(qv[0]) + k

    # ry(0.0) → |0>, measure → 0, +7 → 7 deterministically
    assert multi_kernel(0.0, 7) == 7


def test_multishot_with_cudaq_run():
    """cudaq.run works directly on the kernel without calling it first."""

    @cudaq_kernel
    def bell2():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    counts = cudaq.run(bell2, shots_count=200)
    # Bell state: only |00>=0 and |11>=3 should appear
    for outcome in counts:
        assert outcome in {0, 3}


def test_multishot_array_param_with_cudaq_run():
    """cudaq.run works with a FixedShapeNDArray kernel and zero angles."""

    @cudaq_kernel
    def ry_kernel(angles: FixedShapeNDArray(float, 2)):
        qv = QuantumFloat(1)
        ry(angles[0], qv[0])
        return measure(qv[0])

    counts = cudaq.run(ry_kernel, np.zeros(2), shots_count=100)
    # ry(0.0) → always |0>
    for outcome in counts:
        assert outcome == 0


def test_dynamic_index_into_array_parameter():
    """Dynamic index into a FixedShapeNDArray parameter: qubit is |0⟩ so
    measure returns 0 and angles[0] = 0.0 is selected, leaving the qubit in |0⟩."""

    @cudaq_kernel
    def circuit(angles: FixedShapeNDArray(float, 3)):
        qv = QuantumVariable(1)
        ind = jnp.int32(measure(qv[0]))
        rx(angles[ind], qv[0])
        return measure(qv[0])

    angles = np.array([0.0, 1.57, 3.14])
    result = cudaq.run(circuit, angles, shots_count=10)
    assert result == 10 * [0]


def test_dynamic_index_into_array_parameter_in_nested_function():
    """Dynamic index into a FixedShapeNDArray parameter inside a nested function."""

    @qache
    def inner(arr, qv):
        rz(arr[0], qv[0])

    @cudaq_kernel
    def test_circuit(angles: FixedShapeNDArray(float, 5)):
        qv = QuantumFloat(5)

        inner(angles, qv)
        return measure(qv)

    angles = np.array([1.57, 0.78, 0.39, 0.25, 0.12])
    result = cudaq.run(test_circuit, angles, shots_count=10)
    assert result == 10 * [0]


def test_dynamic_index_into_array_parameter_in_loop():
    """Dynamic index into a FixedShapeNDArray parameter inside a loop."""

    @cudaq_kernel
    def test_circuit(angles: FixedShapeNDArray(float, 5)):
        qv = QuantumFloat(5)

        def cond_fun(val):
            i, qv = val
            return i < 5

        def body_fun(val):
            i, qv = val
            ry(angles[i], qv[i])
            return i + 1, qv

        q_while_loop(cond_fun, body_fun, (0, qv))

        return measure(qv)

    angles = np.array([1.57, 0.78, 0.39, 0.25, 0.12])
    result = cudaq.run(test_circuit, angles, shots_count=10)
    assert result is not None


def test_static_index_into_array_parameter_in_cond():
    """Static index into a FixedShapeNDArray parameter inside a q_cond."""

    @cudaq_kernel
    def test_circuit(angles: FixedShapeNDArray(float, 5)):
        qv = QuantumFloat(5)
        ind = jnp.int32(measure(qv[0]))

        def true_fun(qv, arr):
            rx(arr[0], qv[0])

        def false_fun(qv, arr):
            ry(arr[0], qv[0])

        q_cond(ind == 0, true_fun, false_fun, qv, angles)

        return measure(qv)

    angles = np.array([1.57, 0.78, 0.39, 0.25, 0.12])
    result = cudaq.run(test_circuit, angles, shots_count=10)
    assert result is not None


def test_dynamic_index_into_array_parameter_in_cond_in_loop():
    """Dynamic index into a FixedShapeNDArray parameter inside a q_cond in a loop."""

    @cudaq_kernel
    def main(angles: FixedShapeNDArray(float, 5)):
        qv = QuantumFloat(5)

        def true_fun(qv, angles, ind):
            ry(angles[ind], qv[0])

        def false_fun(qv, angles, ind):
            rz(angles[ind], qv[0])

        def cond_fun(val):
            i, qv = val
            return i < 5

        def body_fun(val):
            i, qv = val

            q_cond(angles[i] > 0.5, true_fun, false_fun, qv, angles, i)
            return i + 1, qv

        q_while_loop(cond_fun, body_fun, (0, qv))

        return measure(qv)

    angles = np.array([1.57, 0.78, 0.39, 0.25, 0.12])
    result = cudaq.run(main, angles, shots_count=10)
    assert result is not None


def test_static_index_into_array_parameter():
    """Static index into a FixedShapeNDArray parameter. Parametrized ansatz with 4 layers, each taking an angle from the array."""

    @cudaq_kernel
    def ansatz_kernel(params: FixedShapeNDArray(float, 4)):
        qv = QuantumVariable(4)
        for layer in range(4):
            beta = params[layer]
            for i in range(4):
                rx(2.0 * beta, qv[i])
        return measure(qv)

    results = cudaq.run(ansatz_kernel, np.array([1.57, 0.78, 0.39, 0.25]), shots_count=10)
    assert results is not None


# ---------------------------------------------------------------------------
# Negative tests: FixedShapeNDArray construction
# ---------------------------------------------------------------------------


def test_fixed_shape_ndarray_unsupported_dtype_complex():
    """FixedShapeNDArray with complex dtype raises TypeError."""
    with pytest.raises(TypeError, match="unsupported dtype"):
        FixedShapeNDArray(complex, 3)


def test_fixed_shape_ndarray_unsupported_dtype_str():
    """FixedShapeNDArray with str dtype raises TypeError."""
    with pytest.raises(TypeError, match="unsupported dtype"):
        FixedShapeNDArray(str, 3)


def test_fixed_shape_ndarray_zero_size():
    """FixedShapeNDArray with size=0 raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        FixedShapeNDArray(float, 0)


def test_fixed_shape_ndarray_negative_size():
    """FixedShapeNDArray with negative size raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        FixedShapeNDArray(float, -1)


def test_fixed_shape_ndarray_float_size():
    """FixedShapeNDArray with a float size raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        FixedShapeNDArray(float, 1.5)


# ---------------------------------------------------------------------------
# Negative tests: @cudaq_kernel decorator
# ---------------------------------------------------------------------------


def test_missing_annotation_raises():
    """@cudaq_kernel raises RuntimeError when a parameter lacks a type annotation."""
    with pytest.raises(RuntimeError, match="requires a type annotation"):

        @cudaq_kernel
        def missing_ann(k):
            qv = QuantumFloat(1)
            return measure(qv[0])


def test_unsupported_annotation_raises():
    """@cudaq_kernel raises RuntimeError for an unsupported annotation type."""
    with pytest.raises(RuntimeError, match="unsupported annotation"):

        @cudaq_kernel
        def bad_ann(k: str):
            qv = QuantumFloat(1)
            return measure(qv[0])


def test_traced_jax_array_arithmetic_triggers_helpful_safeguard_error():
    """Arithmetic on traced jax.numpy arrays should fail early with a helpful
    safeguard message for users."""

    with pytest.raises(RuntimeError, match="traced jax.numpy arrays"):

        @cudaq_kernel
        def bad_kernel(k: int):
            qv = QuantumFloat(2)
            arr1 = jnp.array([1.57, 0.78, 0.39])
            arr2 = jnp.array([1.57, 0.78, 0.39])
            arr3 = arr1 + arr2
            rx(arr3[0], qv[0])
            return measure(qv[0]) + k
