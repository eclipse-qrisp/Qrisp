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

import warnings
import jax
import jax.numpy as jnp
import numpy as np
import operator
import pytest
import re

from qrisp import QuantumVariable, QuantumBool, QuantumFloat, h, mcx, x, y, z, cp, cx, cy, cz, gphase, rx, ry, rz, rxx, rz, rzz, s, swap, sx, t, xxyy, measure, control, invert, conjugate
from qrisp.alg_primitives import amplitude_amplification, q_switch
from qrisp.jasp import make_jaspr, jrange, q_while_loop, q_cond, q_fori_loop, qache

try:
    from qrisp.jasp.mlir.quake_lowering import (
        jaspr_to_quake,
        validate_quake_mlir,
        run_quake_mlir,
        qrisp_cudaq_kernel,
        FixedShapeNDArray,
    )
except ImportError as exc:
    # Skip the entire test file if the import fails
    pytest.skip(f"quake_lowering unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test qrisp cudaq kernel decorator
# ---------------------------------------------------------------------------

def test_qrisp_cudaq_kernel():
    """Test that a simple @qrisp_cudaq_kernel compiles and runs, returning a valid measurement outcome."""
    from qrisp.jasp.mlir.quake_lowering import qrisp_cudaq_kernel

    @qrisp_cudaq_kernel
    def bell():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    result = bell()
    assert result in {0, 3}


def test_qrisp_cudaq_kernel_algorithm():
    """Test that a more complex algorithm can be expressed in a @qrisp_cudaq_kernel and executed."""
    from qrisp.operators import X, Y, Z
    import networkx as nx

    @qrisp_cudaq_kernel
    def main():

        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        H = sum(X(i) for i in G.nodes) + sum(Z(i)*Z(j) for i, j in G.edges)
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

    @qrisp_cudaq_kernel
    def int_kernel(k: int):
        qv = QuantumFloat(2)
        h(qv[0])
        return measure(qv[0]) + k

    # ry(0) leaves qv[0] in |0>, but h puts it in superposition → 0 or 1; +5
    result = int_kernel(5)
    assert result in {5, 6}


def test_float_parameter():
    """Kernel with float rotation angle: ry(0.0) deterministically leaves qubit in |0>."""

    @qrisp_cudaq_kernel
    def float_kernel(angle: float):
        qv = QuantumFloat(1)
        ry(angle, qv[0])
        return measure(qv[0])

    assert float_kernel(0.0) == 0


def test_bool_parameter():
    """Kernel with bool parameter: flips qubit if True, leaves in |0> if False."""

    @qrisp_cudaq_kernel
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

    @qrisp_cudaq_kernel
    def float_arr_kernel(angles: FixedShapeNDArray(float, 3)):
        qv = QuantumFloat(1)
        ry(angles[0], qv[0])
        return measure(qv[0])

    assert float_arr_kernel(np.zeros(3)) == 0


def test_fixed_shape_ndarray_int():
    """FixedShapeNDArray(int, 2): integer element added to measurement."""

    @qrisp_cudaq_kernel
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

    @qrisp_cudaq_kernel
    def multi_kernel(angle: float, k: int):
        qv = QuantumFloat(2)
        ry(angle, qv[0])
        return measure(qv[0]) + k

    # ry(0.0) → |0>, measure → 0, +7 → 7 deterministically
    assert multi_kernel(0.0, 7) == 7


def test_multishot_with_cudaq_run():
    """cudaq.run works directly on the kernel without calling it first."""
    import cudaq

    @qrisp_cudaq_kernel
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
    import cudaq

    @qrisp_cudaq_kernel
    def ry_kernel(angles: FixedShapeNDArray(float, 2)):
        qv = QuantumFloat(1)
        ry(angles[0], qv[0])
        return measure(qv[0])

    counts = cudaq.run(ry_kernel, np.zeros(2), shots_count=100)
    # ry(0.0) → always |0>
    for outcome in counts:
        assert outcome == 0


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
# Negative tests: @qrisp_cudaq_kernel decorator
# ---------------------------------------------------------------------------


def test_missing_annotation_raises():
    """@qrisp_cudaq_kernel raises RuntimeError when a parameter lacks a type annotation."""
    with pytest.raises(RuntimeError, match="requires a type annotation"):

        @qrisp_cudaq_kernel
        def missing_ann(k):
            qv = QuantumFloat(1)
            return measure(qv[0])


def test_unsupported_annotation_raises():
    """@qrisp_cudaq_kernel raises RuntimeError for an unsupported annotation type."""
    with pytest.raises(RuntimeError, match="unsupported annotation"):

        @qrisp_cudaq_kernel
        def bad_ann(k: str):
            qv = QuantumFloat(1)
            return measure(qv[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
