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

import itertools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from scipy.linalg import expm

from qrisp import QuantumVariable
from qrisp.operators import X, Y, Z
from qrisp.operators.qubit.pauli_network import PauliSet
from qrisp.operators.qubit.pauli_network.common import (
    CHUNKS,
    CX,
    Gate,
    PauliNetwork,
    Rotation,
    conjugate_by_chunk,
    leading_identities,
    place,
    pop_rotation,
    score,
)

FACTORS = {(True, False): Z, (False, True): X, (True, True): Y}  # keyed by (z, x)


def pauli_matrix(pauli_set: PauliSet, k: int) -> npt.NDArray[np.complex128]:
    """Dense matrix of string k, with its sign, in Qrisp's qubit convention."""
    operator = 1
    for q in np.flatnonzero(pauli_set.support_mask(k)):
        operator = operator * FACTORS[bool(pauli_set.z[q, k]), bool(pauli_set.x[q, k])](int(q))
    sign = -1 if pauli_set.signs[k] else 1
    if isinstance(operator, int):
        return sign * np.eye(2**pauli_set.num_qubits)
    return sign * operator.to_array(pauli_set.num_qubits)


def unitary_of(build: Callable[[QuantumVariable], object], num_qubits: int) -> npt.NDArray[np.complex128]:
    qv = QuantumVariable(num_qubits)
    build(qv)
    return qv.qs.compile().get_unitary()


def all_two_qubit_paulis() -> PauliSet:
    bits = np.array(list(itertools.product([False, True], repeat=4))).T
    return PauliSet(bits[:2], bits[2:])


class TestGate:
    @pytest.mark.parametrize("gate", [Gate("h", (0,)), Gate("s", (1,)), Gate("sx", (0,)), CX(0, 1), CX(1, 0)])
    def test_emitted_gate_conjugates_like_the_pauli_set(self, gate):
        # PauliSet's row updates must describe the gate that Gate.apply puts in the circuit.
        G = unitary_of(gate.apply, 2)
        paulis = all_two_qubit_paulis()
        before = [pauli_matrix(paulis, k) for k in range(paulis.num_paulis)]
        getattr(paulis, gate.name)(*gate.qubits)
        for k, P in enumerate(before):
            assert np.allclose(pauli_matrix(paulis, k), G @ P @ G.conj().T)

    @pytest.mark.parametrize("gate", [Gate("h", (0,)), Gate("s", (1,)), Gate("sx", (0,)), CX(1, 0)])
    def test_inverse_gate_undoes_gate(self, gate):
        U = unitary_of(lambda qv: (gate.apply(qv), gate.apply(qv, inverse=True)), 2)
        assert np.allclose(U, np.eye(4))

    def test_cx_exposes_control_and_target(self):
        gate = CX(3, 1)
        assert (gate.name, gate.qubits, gate.control, gate.target) == ("cx", (3, 1), 3, 1)


class TestChunks:
    def test_chunks_are_the_18_of_the_paper(self):
        # Sec. 3: CNOT . (U1 x U2) with U1 in {I, H, sqrt(X)} on the control, U2 in {I, H, S} on the target.
        assert len(set(CHUNKS)) == 18
        for chunk in CHUNKS:
            *singles, cx = chunk
            assert isinstance(cx, CX) and len(singles) <= 2
            for gate in singles:
                allowed = {"h", "sx"} if gate.qubits == (cx.control,) else {"h", "s"}
                assert gate.name in allowed and gate.qubits in {(cx.control,), (cx.target,)}

    def test_every_reducible_pair_has_a_chunk(self):
        # Sec. 3: a 2-qubit Pauli that is non-identity on both qubits can be reduced by one chunk.
        paulis = all_two_qubit_paulis()
        reducible = np.zeros(paulis.num_paulis, dtype=bool)
        for chunk in CHUNKS:
            reducible |= conjugate_by_chunk(paulis, chunk, 0, 1).support_sizes() == 1
        assert reducible[paulis.support_sizes() == 2].all()

    def test_place_maps_local_qubits(self):
        chunk = (Gate("h", (1,)), CX(1, 0))
        assert place(chunk, 3, 5) == [Gate("h", (5,)), CX(5, 3)]

    def test_conjugate_by_chunk_returns_a_conjugated_copy(self):
        paulis = all_two_qubit_paulis()
        before = paulis.copy()
        chunk = CHUNKS[7]
        conjugated = conjugate_by_chunk(paulis, chunk, 1, 0)
        expected = paulis.copy()
        for gate in place(chunk, 1, 0):
            getattr(expected, gate.name)(*gate.qubits)
        assert (conjugated.z == expected.z).all() and (conjugated.x == expected.x).all()
        assert (conjugated.signs == expected.signs).all()
        assert (paulis.z == before.z).all() and (paulis.x == before.x).all()


class TestLeadingIdentitiesAndScore:
    def test_leading_identities(self):
        # Strings: Z0 Z1, X1, Z0
        pauli_set = PauliSet([[1, 0, 1], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]])
        assert leading_identities(pauli_set, 0) == 0
        assert leading_identities(pauli_set, 1) == 0
        assert leading_identities(pauli_set[:, 1:], 0) == 1
        assert leading_identities(pauli_set[:, 1], 0) == 1  # no string acts on qubit 0

    def test_score_counts_strings_cleared_from_a_qubit(self):
        # Strings: Z0 Z1, X1. CX(0, 1) maps Z0 Z1 to Z1 and leaves X1, so both become identity on qubit 0.
        pauli_set = PauliSet([[1, 0], [1, 0]], [[0, 0], [0, 1]])
        assert score(pauli_set, (CX(0, 1),), 0, 1) == 2
        # CX(1, 0) maps Z0 Z1 to Z0 and X1 to X0 X1: only the first string is cleared, from qubit 1.
        assert score(pauli_set, (CX(1, 0),), 0, 1) == 1


class TestRotationAndPauliNetwork:
    def test_pop_rotation_reads_axis_qubit_and_sign(self):
        # One string: -Y on qubit 2 of 3, at column 0; reported as input term 7.
        pauli_set = PauliSet([[0], [0], [1]], [[0], [0], [1]])
        pauli_set.signs[0] = True
        assert pop_rotation(pauli_set, 0, 7, 4) == Rotation(term_index=7, qubit=2, axis="Y", negated=True, position=4)

    @pytest.mark.parametrize("axis, negated", list(itertools.product("XYZ", [False, True])))
    def test_rotation_applies_exp_of_signed_pauli(self, axis, negated):
        theta = 0.37
        rotation = Rotation(term_index=0, qubit=1, axis=axis, negated=negated, position=0)
        pauli = {"X": X, "Y": Y, "Z": Z}[axis](1).to_array(2)
        expected = expm(-1j * theta * (-1 if negated else 1) * pauli)
        assert np.allclose(unitary_of(lambda qv: rotation.apply(qv, theta), 2), expected)

    def test_network_iteration_puts_each_rotation_after_its_position_gates(self):
        gates = [CX(0, 1), Gate("h", (0,)), CX(1, 0)]
        rotations = [
            Rotation(0, 0, "Z", False, 0),
            Rotation(1, 1, "X", False, 1),
            Rotation(2, 0, "Z", True, 1),
            Rotation(3, 1, "Y", False, 3),
        ]
        network = PauliNetwork(gates, rotations)
        assert list(network) == [rotations[0], gates[0], rotations[1], rotations[2], gates[1], gates[2], rotations[3]]
