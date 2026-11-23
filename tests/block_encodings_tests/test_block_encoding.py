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

"""Tests for the BlockEncoding class in qrisp.block_encodings."""

import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    cx,
    h,
    jaspify,
    measure,
    multi_measurement,
    prepare,
    ry,
    swap,
    terminal_sampling,
    x,
)
from qrisp.block_encodings import BlockEncoding
from qrisp.misc.exceptions import QrispDeprecationWarning
from qrisp.operators import P1, A, C, X, Y, Z, a, c


def _operator_to_le_matrix(op, num_qubits: int) -> np.ndarray:
    """Returns the dense matrix of an operator in Qrisp's little-endian qubit order."""
    if hasattr(op, "to_qubit_operator"):
        op = op.to_qubit_operator()
    M_be = op.to_sparse_matrix(factor_amount=num_qubits).toarray()
    perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(1 << num_qubits)]
    return M_be[np.ix_(perm, perm)]


def _assert_block_encoding_matches_matrix(BE, M: np.ndarray, b: np.ndarray, atol: float = 1e-6):
    """Verifies BE |b> matches M |b> / ||M |b>|| (amplitudes and relative phases) via swap test."""
    n = int(np.log2(len(b)))
    expected = M @ b
    expected = expected / np.linalg.norm(expected)

    def operand_prep():
        operand = QuantumFloat(n)
        prepare(operand, b)
        return operand

    @terminal_sampling
    def main():
        operand = BE.apply_rus(operand_prep)()

        psi = QuantumFloat(n)
        prepare(psi, expected)

        qb = QuantumBool()
        h(qb)
        with control(qb):
            swap(operand, psi)
        h(qb)

        return qb

    res_dict = main()
    assert np.isclose(res_dict.get(False, 0), 1.0, atol=atol)


@pytest.mark.parametrize(
    ("matrix", "expected_hermitian"),
    [
        # Real symmetric (Hermitian)
        (
            np.array([[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=float),
            True,
        ),
        # Complex Hermitian
        (
            np.array(
                [[1, 1j, 0, 0.5], [-1j, 2, 0.3 - 0.4j, 0], [0, 0.3 + 0.4j, -1, 0], [0.5, 0, 0, 0]],
                dtype=complex,
            ),
            True,
        ),
        # Real non-Hermitian (asymmetric)
        (
            np.array([[0, 1.4, 0, 1.1], [0.2, 0, 1.4, 0], [0, 0.5, 0, 1.4], [1.4, 0, 0.3, 0]], dtype=float),
            False,
        ),
        # Complex non-Hermitian
        (
            np.array(
                [[0.5j, 1 + 0.5j, 0, 0.3], [0.2, -1, 0.7j, 0], [0, 0.4, 0.2 - 0.6j, 1], [1j, 0, 0, 0.8]],
                dtype=complex,
            ),
            False,
        ),
        # Single-Pauli complex matrix (0 ancillas)
        (
            np.array([[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]], dtype=complex),
            False,
        ),
        # Sparse non-Hermitian csr_matrix
        (
            csr_matrix(np.array([[0, 1.2 + 0.3j, 0, 0], [0, 0, 0.8j, 0], [0.5, 0, 0, 1], [0, 0, 0, -0.4j]])),
            False,
        ),
    ],
)
def test_block_encoding_from_array(matrix, expected_hermitian):
    """Verify BlockEncoding.from_array on Hermitian and non-Hermitian dense/sparse arrays."""
    B = BlockEncoding.from_array(matrix)
    assert B.is_hermitian is expected_hermitian

    dense = matrix.toarray() if isinstance(matrix, csr_matrix) else matrix
    b = np.array([1.0, 0.5, -0.5, 1.0])
    _assert_block_encoding_matches_matrix(B, dense, b)


def test_block_encoding_from_array_dagger():
    """Verify dagger() of a non-Hermitian BlockEncoding.from_array encodes the conjugate transpose."""
    M = np.array(
        [[0.5j, 1 + 0.5j, 0, 0.3], [0.2, -1, 0.7j, 0], [0, 0.4, 0.2 - 0.6j, 1], [1j, 0, 0, 0.8]],
        dtype=complex,
    )
    B = BlockEncoding.from_array(M)
    assert B.is_hermitian is False

    b = np.array([1.0, 0.5, -0.5, 1.0])
    _assert_block_encoding_matches_matrix(B.dagger(), M.conj().T, b)


@pytest.mark.parametrize(
    "bad_array",
    [
        np.array([1, 2, 3, 4]),  # 1-D
        np.zeros((2, 4)),  # non-square
        np.zeros((3, 3)),  # non-power-of-two
    ],
)
def test_block_encoding_from_array_value_error(bad_array):
    """Verify BlockEncoding.from_array rejects non-2D, non-square, and non-power-of-two arrays."""
    with pytest.raises(ValueError):
        BlockEncoding.from_array(bad_array)


def test_block_encoding_from_lcu():
    def f0(x):
        x -= 1

    def f1(x):
        x += 1

    BE = BlockEncoding.from_lcu(np.array([1.0, 1.0]), [f0, f1])

    @terminal_sampling
    def main():
        return BE.apply_rus(lambda: QuantumFloat(2))()

    res = main()
    assert res == {1.0: 0.5, 3.0: 0.5}


@pytest.mark.parametrize(
    ("operator", "expected_hermitian"),
    [
        # Hermitian QubitOperator
        (X(0) * X(1) + 0.2 * Y(0) * Y(1) - 0.5 * Z(0), True),
        # Non-Hermitian QubitOperator with complex Pauli coefficients
        ((1 + 0.5j) * X(0) * X(1) - 0.7j * Y(0) * Z(1) + 0.3 * Z(1), False),
        # Non-Hermitian QubitOperator with ladder/projector terms
        (A(0) + 0.5 * C(0) * A(1) + 0.3 * P1(1), False),
        # Single-term non-Hermitian QubitOperator (0 ancillas)
        ((1 + 1j) * X(0) * Z(1), False),
        # Hermitian FermionicOperator
        (c(0) * a(1) + c(1) * a(0), True),
        # Non-Hermitian FermionicOperator
        (a(0) + 0.5 * c(0) * a(1), False),
    ],
)
def test_block_encoding_from_operator(operator, expected_hermitian):
    """Verify BlockEncoding.from_operator on Hermitian and non-Hermitian qubit/fermionic operators."""
    B = BlockEncoding.from_operator(operator)
    assert B.is_hermitian is expected_hermitian

    M = _operator_to_le_matrix(operator, num_qubits=2)
    b = np.array([1.0, 0.5, -0.5, 1.0])
    _assert_block_encoding_matches_matrix(B, M, b)


def test_block_encoding_from_operator_dagger():
    """Verify dagger() of a non-Hermitian BlockEncoding.from_operator encodes the adjoint operator."""
    op = (1 + 0.5j) * X(0) * X(1) - 0.7j * Y(0) * Z(1) + A(0)
    B = BlockEncoding.from_operator(op)
    assert B.is_hermitian is False

    M_dag = _operator_to_le_matrix(op.adjoint(), num_qubits=2)
    b = np.array([1.0, 0.5, -0.5, 1.0])
    _assert_block_encoding_matches_matrix(B.dagger(), M_dag, b)


def test_block_encoding_from_eye():

    # k = 0: ones on the main diagonal
    BE1 = BlockEncoding.from_eye(diagonal_index=0)

    # k = -4: ones on the fourth lower subdiagonal
    # (non-cyclic) shift |x> -> |x+4>
    BE2 = BlockEncoding.from_eye(diagonal_index=-4)

    BE3 = BE1.kron(BE2)

    def operand_prep():
        operand1 = QuantumFloat(3)
        operand2 = QuantumFloat(3)
        h(operand1)
        cx(operand1, operand2)
        return operand1, operand2

    @terminal_sampling
    def main():
        operand1, operand2 = BE3.apply_rus(operand_prep)()
        return operand1, operand2

    res_dict = main()
    assert res_dict == pytest.approx({(0.0, 4.0): 0.25, (1.0, 5.0): 0.25, (2.0, 6.0): 0.25, (3.0, 7.0): 0.25})


def test_block_encoding_from_projector():

    # Define projector P = |1><+|
    P = BlockEncoding.from_projector(1, lambda qv: h(qv))

    # Prepare operand in superposition state
    def operand_prep():
        operand = QuantumFloat(2)
        h(operand)
        return operand

    @terminal_sampling
    def main():
        return P.apply_rus(operand_prep)()

    res = main()
    assert res == {1.0: 1.0}


def test_block_encoding_from_projector_value_error():

    with pytest.raises(ValueError) as excinfo:
        P = BlockEncoding.from_projector(1, (2, 3))

    assert "Size mismatch: left has 1 elements, but right has 2" in str(excinfo.value)


def test_block_encoding_apply():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    operand = QuantumFloat(2)
    ancillas = BE.apply(operand)

    res_dict = multi_measurement(ancillas + [operand])
    assert res_dict == {(0, 0): 0.25, (0, 3): 0.25, (1, 0): 0.25, (1, 3): 0.25}


def test_block_encoding_apply_value_error():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    wrong_operands = (QuantumFloat(2), QuantumFloat(2))
    with pytest.raises(ValueError) as excinfo:
        ancillas = BE.apply(*wrong_operands)

    assert "Operation expected 1 operands, but got 2" in str(excinfo.value)


def test_block_encoding_apply_rus_type_error():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    wrong_operands = (QuantumFloat(2), QuantumFloat(2))
    with pytest.raises(TypeError) as excinfo:
        operands = BE.apply_rus(wrong_operands)()

    assert "Expected 'operand_prep' to be a callable, but got" in str(excinfo.value)


def test_block_encoding_apply_rus_value_error():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    def wrong_operand_prep():
        return QuantumFloat(2), QuantumFloat(2)

    with pytest.raises(ValueError) as excinfo:
        operands = BE.apply_rus(wrong_operand_prep)()

    assert "Operation expected 1 operands, but got 2" in str(excinfo.value)


def test_block_encoding_ev():
    H = X(0) * X(1) + 0.5 * Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    def operand_prep(phi):
        qv = QuantumFloat(2)
        ry(phi, qv[0])
        return qv

    @jaspify(terminal_sampling=True)
    def main(BE):
        ev = BE.expectation_value(operand_prep, shots=1_000_000)(np.pi / 4)
        return ev

    # Jasp mode (Dynamic Execution): Hadamard test
    be_ev_jasp = main(BE)

    # Standard mode (Static Execution): Hadamard test
    be_ev = BE.expectation_value(operand_prep, shots=1_000_000)(np.pi / 4)

    # Standard mode (Static Execution): Pauli measurements
    op_ev = H.expectation_value(operand_prep, precision=0.001)(np.pi / 4)

    assert np.abs(be_ev_jasp - be_ev) < 1e-2
    assert np.abs(be_ev - op_ev) < 1e-2


def test_block_encoding_ev_type_error():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    wrong_operands = (QuantumFloat(2), QuantumFloat(2))
    with pytest.raises(TypeError) as excinfo:
        ev = BE.expectation_value(wrong_operands)()

    assert "Expected 'operand_prep' to be a callable, but got" in str(excinfo.value)


def test_block_encoding_ev_value_error():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    def wrong_operand_prep():
        return QuantumFloat(2), QuantumFloat(2)

    with pytest.raises(ValueError) as excinfo:
        ev = BE.expectation_value(wrong_operand_prep)()

    assert "Operation expected 1 operands, but got 2" in str(excinfo.value)


def test_block_encoding_resources():
    H = X(0) * X(1) + 0.5 * Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    res_dict = BE.resources(QuantumFloat(2))
    # {'gate counts': {'x': 3, 'cz': 2, 'u3': 2, 'cx': 4, 'gphase': 2},
    # 'depth': 12, 'qubits': 4}
    assert isinstance(res_dict["gate counts"], dict)
    assert isinstance(res_dict["depth"], int)
    assert isinstance(res_dict["qubits"], int)


def test_block_encoding_resources_max_allocations_deprecated():
    """``max_allocations`` is deprecated: it warns only when passed and does not change the result."""
    H = X(0) * X(1) + 0.5 * Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    with warnings.catch_warnings():
        warnings.simplefilter("error", QrispDeprecationWarning)
        res_default = BE.resources(QuantumFloat(2))

    with pytest.warns(QrispDeprecationWarning, match="max_allocations"):
        res_deprecated = BE.resources(QuantumFloat(2), max_allocations=1)

    assert res_deprecated == res_default


def test_block_encoding_resources_value_error():
    H = X(0) * X(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H)

    wrong_operands = (QuantumFloat(2), QuantumFloat(2))
    with pytest.raises(ValueError) as excinfo:
        res_dict = BE.resources(*wrong_operands)

    assert "Operation expected 1 operands, but got 2" in str(excinfo.value)


def test_block_encoding_alpha_dynamic():

    @terminal_sampling
    def main():

        a = QuantumFloat(2)
        x(a)
        # b is dynamic
        b = measure(a)

        def U(qv):
            x(qv)

        BE1 = BlockEncoding(b, [], U)
        BE2 = BlockEncoding(1, [], U)
        BE = BE1 + BE2

        return BE.apply_rus(lambda: QuantumVariable(2))()

    res = main()
    assert res == {3: 1.0}

    @terminal_sampling
    def main():

        a = QuantumFloat(1)
        x(a)
        # b is dynamic
        b = measure(a)

        def U1(qv):
            x(qv)

        def U2(qv):
            pass

        BE1 = BlockEncoding(b, [], U1)
        BE2 = BlockEncoding(1, [], U2)
        BE = BE1 + BE2

        return BE.apply_rus(lambda: QuantumVariable(2))()

    res = main()
    assert res == {0: 0.5, 3: 0.5}


def test_block_encoding_dagger():
    """This test verifies that the dagger of a block-encoding correctly implements the Hermitian conjugate of the operator.
    We construct a block-encoding for a non-Hermitian matrix A, apply the dagger to get a block-encoding for A^†,
    and then use it to perform state transformation and compare with classical results.
    """
    N = 8
    A = np.eye(N, k=1) + 1 * np.eye(N)
    A[N - 1, 0] = 1

    b = np.array([0, 1, 1, 0, 1, 0, 0, 1])

    # Define identity and shift unitaries for LCU construction
    def id(qv):
        pass

    def U(qv):
        qv -= 1

    BA = BlockEncoding.from_lcu(np.array([1.0, 1.0]), [id, U])
    BA_dg = BA.dagger()

    def operand_prep():
        operand = QuantumFloat(3)
        prepare(operand, b)
        return operand

    @terminal_sampling
    def main():
        operand = BA_dg.apply_rus(operand_prep)()
        return operand

    res_dict = main()
    amps = np.sqrt([res_dict.get(i, 0) for i in range(len(b))])
    expected = A.conj().T @ b / np.linalg.norm(A.conj().T @ b)
    assert np.allclose(amps, expected, atol=1e-6)
