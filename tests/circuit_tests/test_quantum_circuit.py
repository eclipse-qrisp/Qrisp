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

import os
import tempfile

import numpy as np
import pytest
import sympy
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from sympy import symbols

from qrisp.circuit import Clbit, Instruction, Qubit
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.standard_operations import XGate


class TestQuantumCircuitInitialization:
    """Tests for QuantumCircuit initialization."""

    def test_initialization(self):
        """Test basic initialization with qubit and clbit counts."""

        qc = QuantumCircuit(num_qubits=3, num_clbits=2)
        assert len(qc.qubits) == 3
        assert len(qc.clbits) == 2

    def test_error_wrong_type_initialization(self):
        """Test that wrong argument types raise TypeError."""

        with pytest.raises(TypeError, match="type str for num_qubits"):
            QuantumCircuit(num_qubits="3")

        with pytest.raises(TypeError, match="type float for num_clbits"):
            QuantumCircuit(num_clbits=2.5)

    def test_fan_out_example(self):
        """Test the fan-out example from the QuantumCircuit docstring."""

        qc_0 = QuantumCircuit(4)
        qc_0.cx(0, range(1, 4))
        assert len(qc_0.data) == 3
        assert all(str(i)[:2] == "cx" for i in qc_0.data)

        qc_1 = QuantumCircuit(4)
        qc_1.h(0)
        qc_1.append(qc_0.to_gate(name="fan-out"), qc_1.qubits)
        assert len(qc_1.data) == 2
        assert str(qc_1.data[0])[:1] == "h"
        assert str(qc_1.data[1])[:7] == "fan-out"

        qc_1.measure(qc_1.qubits)
        assert len(qc_1.data) == 6
        assert all(str(qc_1.data[i])[:7] == "measure" for i in range(2, 6))

    def test_from_qiskit_fan_out_example(self):
        """Test QuantumCircuit.from_qiskit conversion."""

        qc_2 = QiskitQuantumCircuit(4)
        qc_2.cx(0, range(1, 4))

        qrisp_qc = QuantumCircuit.from_qiskit(qc_2)
        assert len(qrisp_qc.qubits) == 4
        assert len(qrisp_qc.clbits) == 0
        assert len(qrisp_qc.data) == 3
        assert all(str(i)[:2] == "cx" for i in qrisp_qc.data)

    def test_abstract_parameters_docstring_example(self):
        """Test the abstract parameters example from the docstring."""

        qc = QuantumCircuit(3)
        abstract_parameters = symbols("a b c")
        for i in range(3):
            qc.p(abstract_parameters[i], i)

        assert len(qc.data) == 3
        assert qc.abstract_params == set(abstract_parameters)

        bound_qc = qc.bind_parameters({abstract_parameters[i]: i for i in range(3)})
        assert len(bound_qc.data) == 3
        assert bound_qc.abstract_params == set()


class TestQuantumCircuitMethods:
    """Tests for QuantumCircuit methods."""

    # ------------------------------------------------------------------ #
    # add_qubit / add_clbit                                              #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "method, container",
        [
            ("add_qubit", "qubits"),
            ("add_clbit", "clbits"),
        ],
    )
    def test_add_bit_default(self, method, container):
        """Adding a bit without providing an object creates and appends a new one."""

        qc = QuantumCircuit()
        initial_count = len(getattr(qc, container))

        added = getattr(qc, method)()

        assert len(getattr(qc, container)) == initial_count + 1
        assert added in getattr(qc, container)
        assert added is getattr(qc, container)[-1]

    @pytest.mark.parametrize(
        "method,container,bit_cls,name",
        [
            ("add_qubit", "qubits", Qubit, "custom_qb"),
            ("add_clbit", "clbits", Clbit, "custom_cb"),
        ],
    )
    def test_add_bit_with_object(self, method, container, bit_cls, name):
        """Adding a bit by providing a typed object registers it correctly."""

        qc = QuantumCircuit()
        custom_bit = bit_cls(name)

        added = getattr(qc, method)(custom_bit)

        assert added is custom_bit
        assert custom_bit in getattr(qc, container)

    @pytest.mark.parametrize(
        "init_args,method,counter_attr",
        [
            ((2, 0), "add_qubit", "qubit_index_counter"),
            ((0, 2), "add_clbit", "clbit_index_counter"),
        ],
    )
    def test_add_bit_increments_counter(self, init_args, method, counter_attr):
        """Adding a bit increments the corresponding index counter."""

        qc = QuantumCircuit(*init_args)
        initial = qc.__dict__.get(counter_attr, getattr(qc, counter_attr))[0]
        getattr(qc, method)()
        assert getattr(qc, counter_attr)[0] == initial + 1

    @pytest.mark.parametrize(
        "method,match",
        [
            ("add_qubit", "Tried to add type .* as a qubit"),
            ("add_clbit", "Tried to add type .* as a classical bit"),
        ],
    )
    def test_add_bit_wrong_type_raises(self, method, match):
        """Passing a non-bit object raises TypeError."""
        qc = QuantumCircuit()
        with pytest.raises(TypeError, match=match):
            getattr(qc, method)("not_a_bit")

    @pytest.mark.parametrize(
        "method,bit_cls,match",
        [
            ("add_qubit", Qubit, "Qubit name .* already exists"),
            ("add_clbit", Clbit, "Clbit name .* already exists"),
        ],
    )
    def test_add_bit_duplicate_raises(self, method, bit_cls, match):
        """Adding two bits with the same name raises ValueError."""
        qc = QuantumCircuit()
        getattr(qc, method)(bit_cls("dup"))
        with pytest.raises(ValueError, match=match):
            getattr(qc, method)(bit_cls("dup"))

    # ------------------------------------------------------------------ #
    # to_op / to_gate                                                    #
    # ------------------------------------------------------------------ #

    def test_to_op_basic(self):
        """to_op produces an Operation with the expected qubit/clbit counts."""

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(1)
        op = qc.to_op(name="test_op")
        assert op.name == "test_op"
        assert op.num_qubits == 2
        assert op.num_clbits == 0
        assert op.definition is not None

    def test_to_op_default_name(self):
        """to_op generates a unique default name when none is supplied."""

        op = QuantumCircuit(1).to_op()
        assert op.name.startswith("circuit")

    def test_to_op_multiple_calls_unique_names(self):
        """Successive default-named to_op calls produce distinct names."""

        op1 = QuantumCircuit(1).to_op()
        op2 = QuantumCircuit(1).to_op()
        assert op1.name != op2.name

    def test_to_op_with_classical_bits(self):
        """to_op preserves classical bit count."""

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        op = qc.to_op(name="qc_with_clbits")
        assert op.num_qubits == 2
        assert op.num_clbits == 2

    def test_to_op_removes_allocation_instructions(self):
        """to_op strips qb_alloc and qb_dealloc from the definition."""

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.data.append(Instruction(XGate(), [qc.qubits[1]]))
        op = qc.to_op(name="filtered_op")
        assert all(
            instr.op.name not in {"qb_alloc", "qb_dealloc"}
            for instr in op.definition.data
        )

    def test_to_op_definition_is_copy(self):
        """Mutating the source circuit after to_op does not change the definition."""

        qc = QuantumCircuit(1)
        qc.h(0)
        op = qc.to_op(name="copy_test")
        original_len = len(op.definition.data)
        qc.x(0)
        assert len(op.definition.data) == original_len

    def test_to_op_preserves_gate_sequence(self):
        """The definition inside the Operation mirrors the source circuit."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(1)
        op = qc.to_op(name="seq_op")
        assert len(op.definition.data) == 3
        assert len(op.definition.qubits) == 2

    def test_to_gate_basic(self):
        """to_gate wraps a circuit into a gate with the correct qubit count."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        gate = qc.to_gate(name="my_gate")
        assert gate.name == "my_gate"
        assert gate.num_qubits == 2
        assert gate.num_clbits == 0

    def test_to_gate_default_name(self):
        """to_gate generates a unique default name when none is supplied."""

        gate = QuantumCircuit(1).to_gate()
        assert gate.name.startswith("circuit")

    def test_to_gate_error_if_clbits_present(self):
        """to_gate raises ValueError when the circuit contains classical bits."""

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        with pytest.raises(ValueError, match="classical bits"):
            qc.to_gate(name="invalid_gate")

    # ------------------------------------------------------------------ #
    # extend                                                             #
    # ------------------------------------------------------------------ #

    def test_extend_docstring_example(self):
        """Test the extend docstring example with a reversed qubit mapping."""

        ext = QuantumCircuit(4)
        ext.cx(0, 1)
        ext.cy(0, 2)
        ext.cz(0, 3)

        qc = QuantumCircuit(4)
        qc.extend(ext, {ext.qubits[i]: qc.qubits[-1 - i] for i in range(4)})
        assert len(qc.data) == 3
        assert str(qc.data[0])[:2] == "cx"

    def test_extend_with_default_translation(self):
        """extend without a translation dictionary appends in-place."""

        ext = QuantumCircuit(2)
        ext.x(0)
        ext.y(1)

        qc = QuantumCircuit.copy(ext)
        qc.extend(ext)
        assert len(qc.data) == 4

    def test_extend_with_classical_bits(self):
        """extend correctly handles circuits that include classical bits."""

        ext = QuantumCircuit(2, 1)
        ext.x(0)
        ext.measure(0, 0)

        qc = QuantumCircuit(2, 1)
        qc.h(1)
        qc.extend(
            ext,
            {
                ext.qubits[0]: qc.qubits[0],
                ext.qubits[1]: qc.qubits[1],
                ext.clbits[0]: qc.clbits[0],
            },
        )
        assert len(qc.data) == 3

    def test_extend_empty_circuit(self):
        """Extending with an empty circuit leaves the target unchanged."""

        qc = QuantumCircuit(2)
        qc.x(0)
        before = len(qc.data)
        qc.extend(QuantumCircuit(2))
        assert len(qc.data) == before

    def test_extend_raises_for_invalid_qubit_reference(self):
        """extend raises when the translation maps to a qubit absent from the target."""

        ext = QuantumCircuit(2)
        ext.x(0)
        qc = QuantumCircuit(1)
        with pytest.raises(
            ValueError, match="Instruction Qubits .* not present in circuit"
        ):
            qc.extend(ext, {ext.qubits[0]: ext.qubits[1]})

    def test_extend_raises_for_invalid_clbit_reference(self):
        """extend raises when the translation maps to a clbit absent from the target."""

        ext = QuantumCircuit(1, 2)
        ext.measure(0, 0)
        qc = QuantumCircuit(1, 1)
        with pytest.raises(
            ValueError, match="Instruction Clbits not present in circuit"
        ):
            qc.extend(
                ext,
                {
                    ext.qubits[0]: qc.qubits[0],
                    ext.clbits[0]: ext.clbits[1],
                },
            )

    # ------------------------------------------------------------------ #
    # copy / clearcopy                                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "method,keeps_data",
        [
            ("copy", True),
            ("clearcopy", False),
        ],
    )
    def test_copy_variants_data(self, method, keeps_data):
        """copy preserves instructions; clearcopy discards them."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = getattr(qc, method)()
        assert len(result.data) == (2 if keeps_data else 0)

    @pytest.mark.parametrize("method", ["copy", "clearcopy"])
    def test_copy_variants_preserve_structure(self, method):
        """Both copy variants preserve qubit and clbit identity."""

        qc = QuantumCircuit(3, 2)
        qc.h(0)
        result = getattr(qc, method)()
        assert len(result.qubits) == 3
        assert len(result.clbits) == 2
        for i in range(3):
            assert result.qubits[i] is qc.qubits[i]
        for i in range(2):
            assert result.clbits[i] is qc.clbits[i]

    @pytest.mark.parametrize("method", ["copy", "clearcopy"])
    def test_copy_variants_independence(self, method):
        """Mutating the result does not affect the original."""

        qc = QuantumCircuit(2)
        qc.h(0)
        result = getattr(qc, method)()
        result.x(0)
        assert len(qc.data) == 1

    @pytest.mark.parametrize("method", ["copy", "clearcopy"])
    def test_copy_variants_abstract_params(self, method):
        """Both variants copy abstract_params as an equal but distinct set."""

        qc = QuantumCircuit(1)
        a, b = symbols("a b")
        qc.abstract_params = {a, b}
        result = getattr(qc, method)()
        assert result.abstract_params == qc.abstract_params
        assert result.abstract_params is not qc.abstract_params

    def test_copy_shares_instruction_objects(self):
        """copy shares the same Instruction objects (shallow copy of data)."""

        qc = QuantumCircuit(1)
        qc.h(0)
        assert qc.copy().data[0] is qc.data[0]

    # ------------------------------------------------------------------ #
    # compare_unitary                                                    #
    # ------------------------------------------------------------------ #

    def test_compare_unitary_commuting_gates(self):
        """Circuits differing only by commuting gate order are equal."""

        qc_0 = QuantumCircuit(2)
        qc_0.z(0)
        qc_0.cx(0, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.cx(0, 1)
        qc_1.z(0)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_different_circuits(self):
        """Genuinely different circuits compare as unequal."""

        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.cx(0, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.x(0)
        qc_1.y(1)

        assert qc_0.compare_unitary(qc_1) is False

    def test_compare_unitary_different_qubit_counts(self):
        """Circuits with different qubit counts compare as unequal."""

        qc_0 = QuantumCircuit(2)
        qc_1 = QuantumCircuit(3)
        assert qc_0.compare_unitary(qc_1) is False

    def test_compare_unitary_empty_circuits(self):
        """Two empty circuits (both identity) compare as equal."""

        assert QuantumCircuit(2).compare_unitary(QuantumCircuit(2)) is True

    def test_compare_unitary_x_squared_is_identity(self):
        """X² = I: a double-X circuit equals an identity circuit."""

        qc_0 = QuantumCircuit(1)
        qc_0.x(0)
        qc_0.x(0)
        assert qc_0.compare_unitary(QuantumCircuit(1)) is True

    @pytest.mark.parametrize("precision", [4, 10])
    def test_compare_unitary_precision_parameter(self, precision):
        """Identical circuits compare as equal regardless of precision."""

        qc = QuantumCircuit(1)
        qc.h(0)
        assert qc.compare_unitary(qc, precision=precision) is True

    def test_compare_unitary_parametric_angles(self):
        """Same angle → equal; different angle → unequal."""

        qc_same_0 = QuantumCircuit(1)
        qc_same_0.rx(np.pi / 4, 0)
        qc_same_1 = QuantumCircuit(1)
        qc_same_1.rx(np.pi / 4, 0)
        assert qc_same_0.compare_unitary(qc_same_1) is True

        qc_diff = QuantumCircuit(1)
        qc_diff.rx(np.pi / 3, 0)
        assert qc_same_0.compare_unitary(qc_diff) is False

    def test_compare_unitary_global_phase(self):
        """Circuits that differ only by a global phase scalar are equal iff ignore_gphase=True."""

        # RX(2π) = -I  (global phase of -1 relative to the identity)
        qc_0 = QuantumCircuit(1)
        qc_0.rx(2 * np.pi, 0)

        # identity
        qc_1 = QuantumCircuit(1)

        assert qc_0.compare_unitary(qc_1, ignore_gphase=False) is False
        assert qc_0.compare_unitary(qc_1, ignore_gphase=True) is True

    def test_compare_unitary_cx_cx_cx_is_swap(self):
        """SWAP(0,1) equals the three-CX decomposition CX(0,1)·CX(1,0)·CX(0,1)."""

        qc_0 = QuantumCircuit(2)
        qc_0.cx(0, 1)
        qc_0.cx(1, 0)
        qc_0.cx(0, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.swap(0, 1)

        assert qc_0.compare_unitary(qc_1) is True

    # ------------------------------------------------------------------ #
    # inverse                                                            #
    # ------------------------------------------------------------------ #

    def test_inverse_reverses_gate_order(self):
        """Inverse reverses the sequence of gates in the circuit."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(1)

        inv_qc = qc.inverse()

        assert len(inv_qc.data) == 3
        assert str(inv_qc.data[0])[:1] == "z"
        assert str(inv_qc.data[1])[:2] == "cx"
        assert str(inv_qc.data[2])[:1] == "h"

    def test_inverse_empty_circuit(self):
        """Inverse of an empty circuit is an empty circuit with the same structure."""

        qc = QuantumCircuit(3, 2)
        inv_qc = qc.inverse()
        assert len(inv_qc.data) == 0
        assert len(inv_qc.qubits) == 3
        assert len(inv_qc.clbits) == 2

    def test_inverse_returns_new_circuit(self):
        """inverse() returns a new object without mutating the original."""

        qc = QuantumCircuit(1)
        qc.h(0)
        inv_qc = qc.inverse()
        assert inv_qc is not qc
        assert len(qc.data) == 1  # original unchanged

    def test_inverse_double_inverse_is_original(self):
        """Applying inverse twice recovers the original unitary."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.s(1)
        assert qc.compare_unitary(qc.inverse().inverse()) is True

    @pytest.mark.parametrize("gate_name", ["x", "y", "z", "h"])
    def test_inverse_self_adjoint_gates(self, gate_name):
        """Self-adjoint gates (X, Y, Z, H) are their own inverse."""

        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(0)
        assert qc.compare_unitary(qc.inverse()) is True

    @pytest.mark.parametrize(
        "gate_name,angle",
        [
            ("rx", np.pi / 5),
            ("ry", np.pi / 7),
            ("rz", np.pi / 3),
            ("p", np.pi / 3),
        ],
    )
    def test_inverse_parametric_gate_cancels(self, gate_name, angle):
        """A parametric gate composed with its inverse yields the identity."""

        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(angle, 0)
        combined = qc.clearcopy()
        combined.extend(qc)
        combined.extend(qc.inverse())
        assert combined.compare_unitary(QuantumCircuit(1)) is True

    @pytest.mark.parametrize(
        "build",
        [
            lambda qc: (qc.h(0), qc.cx(0, 1), qc.ry(np.pi / 4, 1)),
            lambda qc: (qc.x(0), qc.z(0), qc.y(0)),
            lambda qc: (qc.h(0), qc.cx(0, 1), qc.cx(1, 0)),
        ],
    )
    def test_inverse_combined_with_original_is_identity(self, build):
        """Any circuit composed with its inverse yields the identity unitary."""

        qc = QuantumCircuit(2)
        build(qc)
        combined = qc.clearcopy()
        combined.extend(qc)
        combined.extend(qc.inverse())
        assert combined.compare_unitary(QuantumCircuit(2)) is True

    # ------------------------------------------------------------------ #
    # get_unitary — single-qubit standard gates                          #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "gate_name,expected",
        [
            ("x", np.array([[0, 1], [1, 0]])),
            ("y", np.array([[0, -1j], [1j, 0]])),
            ("z", np.array([[1, 0], [0, -1]])),
            ("h", np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
            ("s", np.array([[1, 0], [0, 1j]])),
            ("t", np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])),
        ],
    )
    def test_get_unitary_standard_single_qubit_gates(self, gate_name, expected):
        """get_unitary returns the correct matrix for each standard single-qubit gate."""

        qc = QuantumCircuit(1)
        getattr(qc, gate_name)(0)
        unitary = qc.get_unitary()
        assert unitary.shape == (2, 2)
        assert np.allclose(unitary, np.array(expected, dtype=np.complex64), atol=1e-5)

    # ------------------------------------------------------------------ #
    # get_unitary — general properties                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("num_qubits", [1, 2, 3])
    def test_get_unitary_empty_circuit_is_identity(self, num_qubits):
        """Empty circuit of n qubits has the 2ⁿ × 2ⁿ identity as its unitary."""

        qc = QuantumCircuit(num_qubits)
        dim = 2**num_qubits
        assert np.allclose(qc.get_unitary(), np.eye(dim))

    @pytest.mark.parametrize(
        "num_qubits,build",
        [
            (2, lambda qc: (qc.h(0), qc.cx(0, 1))),
            (2, lambda qc: qc.swap(0, 1)),
            (3, lambda qc: (qc.h(0), qc.cx(0, 1), qc.cx(1, 2), qc.ry(np.pi / 6, 2))),
        ],
    )
    def test_get_unitary_is_unitary(self, num_qubits, build):
        """get_unitary always returns a proper unitary matrix (U†U = I)."""

        qc = QuantumCircuit(num_qubits)
        build(qc)
        u = qc.get_unitary()
        dim = 2**num_qubits
        assert np.allclose(u @ u.conj().T, np.eye(dim), atol=1e-5)

    def test_get_unitary_hadamard_is_hermitian(self):
        """The Hadamard gate is its own inverse: H = H†."""

        qc = QuantumCircuit(1)
        qc.h(0)
        u = qc.get_unitary()
        assert np.allclose(u, u.conj().T)

    # ------------------------------------------------------------------ #
    # get_unitary — docstring examples                                   #
    # ------------------------------------------------------------------ #

    def test_get_unitary_docstring_example_numeric(self):
        """Reproduce the numeric controlled-phase docstring example."""

        qc = QuantumCircuit(2)
        phi = np.pi
        qc.p(phi / 2, 0)
        qc.p(phi / 2, 1)
        qc.cx(0, 1)
        qc.p(-phi / 2, 1)
        qc.cx(0, 1)

        expected = np.diag([1, 1, 1, -1]).astype(np.complex64)
        assert np.allclose(qc.get_unitary(decimals=4), expected, atol=1e-4)

    def test_get_unitary_docstring_example_symbolic(self):
        """Reproduce the symbolic controlled-phase docstring example."""

        qc = QuantumCircuit(2)
        phi = sympy.Symbol("phi")
        qc.p(phi / 2, 0)
        qc.p(phi / 2, 1)
        qc.cx(0, 1)
        qc.p(-phi / 2, 1)
        qc.cx(0, 1)

        u = qc.get_unitary(decimals=4)
        assert u.dtype == np.dtype("O")
        assert u.shape == (4, 4)
        assert u[0, 0] == 1
        assert u[3, 3].has(phi)

    # ------------------------------------------------------------------ #
    # get_unitary — rounding and symbolic handling                       #
    # ------------------------------------------------------------------ #

    def test_get_unitary_phase_gate_numeric(self):
        """Phase gate P(π/2) yields the expected diagonal matrix."""

        qc = QuantumCircuit(1)
        qc.p(np.pi / 2, 0)
        expected = np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]], dtype=np.complex64)
        assert np.allclose(qc.get_unitary(), expected)

    def test_get_unitary_decimals_snapping(self):
        """Values within 10⁻ᵈ of 1 in symbolic arrays are snapped to exactly 1."""

        qc = QuantumCircuit(1)
        qc.p(sympy.Symbol("a"), 0)
        u = qc.get_unitary(decimals=4)
        assert u.dtype == np.dtype("O")
        assert u[0, 0] == 1

    @pytest.mark.parametrize(
        "symbols_str,num_qubits,build",
        [
            ("theta", 1, lambda qc, s: qc.rx(s[0], 0)),
            ("a b", 2, lambda qc, s: (qc.rx(s[0], 0), qc.ry(s[1], 1))),
        ],
    )
    def test_get_unitary_symbolic_parameters(self, symbols_str, num_qubits, build):
        """get_unitary with symbolic parameters returns a dtype=object array."""

        syms = sympy.symbols(symbols_str)
        syms = (syms,) if not isinstance(syms, (list, tuple)) else syms
        qc = QuantumCircuit(num_qubits)
        build(qc, syms)
        u = qc.get_unitary()
        assert u.dtype == np.dtype("O")
        assert u.shape == (2**num_qubits,) * 2

    # ------------------------------------------------------------------ #
    # get_depth_dic                                                      #
    # ------------------------------------------------------------------ #

    def test_get_depth_dic_sequential_gates(self):
        """Gates applied sequentially on the same qubit accumulate depth."""

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        depth_dic = qc.get_depth_dic()
        assert depth_dic[qc.qubits[0]] == 3

    def test_get_depth_dic_parallel_gates(self):
        """Gates on different qubits with no dependency share the same depth level."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        depth_dic = qc.get_depth_dic()
        assert depth_dic[qc.qubits[0]] == 1
        assert depth_dic[qc.qubits[1]] == 1

    def test_get_depth_dic_entangling_gate(self):
        """A two-qubit gate after single-qubit gates raises both qubits to the same depth."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(2)
        depth_dic = qc.get_depth_dic()
        assert depth_dic[qc.qubits[0]] == 2
        assert depth_dic[qc.qubits[1]] == 2
        assert depth_dic[qc.qubits[2]] == 1

    def test_get_depth_dic_idle_qubits_are_zero(self):
        """Qubits with no operations have depth 0."""

        qc = QuantumCircuit(3)
        qc.h(0)
        depth_dic = qc.get_depth_dic()
        assert depth_dic[qc.qubits[1]] == 0
        assert depth_dic[qc.qubits[2]] == 0

    def test_get_depth_dic_empty_circuit(self):
        """An empty circuit with qubits returns 0 for every qubit."""

        qc = QuantumCircuit(3)
        depth_dic = qc.get_depth_dic()
        assert list(depth_dic.values()) == [0, 0, 0]

    def test_get_depth_dic_zero_qubits_returns_empty(self):
        """A circuit with no qubits returns an empty dictionary."""

        assert QuantumCircuit(0).get_depth_dic() == {}

    def test_get_depth_dic_max_equals_depth(self):
        """The maximum value in get_depth_dic equals depth()."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        depth_dic = qc.get_depth_dic()
        assert max(depth_dic.values()) == qc.depth()

    def test_get_depth_dic_keys_are_qubits(self):
        """Keys of the returned dictionary are the circuit's Qubit objects."""

        qc = QuantumCircuit(2)
        qc.h(0)
        depth_dic = qc.get_depth_dic()
        assert set(depth_dic.keys()) == set(qc.qubits)

    # ------------------------------------------------------------------ #
    # depth                                                              #
    # ------------------------------------------------------------------ #

    def test_depth_docstring_example(self):
        """Docstring example: H on q0, CX(q0,q1), CX(q1,q2) gives depth 3."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        assert qc.depth() == 3

    def test_depth_empty_circuit(self):
        """An empty circuit has depth 0."""

        assert QuantumCircuit(3).depth() == 0

    def test_depth_parallel_gates(self):
        """Gates on independent qubits execute in parallel — depth is 1."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.z(2)
        assert qc.depth() == 1

    def test_depth_sequential_gates(self):
        """Gates applied sequentially on the same qubit accumulate depth."""

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        assert qc.depth() == 3

    def test_depth_custom_indicator_counts_only_cx(self):
        """A depth_indicator that returns 0 for non-CX gates counts only CX layers."""

        qc = QuantumCircuit(2)
        qc.h(0)  # ignored by indicator
        qc.cx(0, 1)  # depth 1
        qc.h(1)  # ignored
        qc.cx(0, 1)  # depth 2

        cx_only = lambda op: 1 if op.name == "cx" else 0
        assert qc.depth(depth_indicator=cx_only) == 2

    def test_depth_transpile_false_counts_composite_as_one(self):
        """With transpile=False a composite gate counts as a single layer."""

        inner = QuantumCircuit(2)
        inner.cx(0, 1)
        inner.cx(1, 0)
        gate = inner.to_gate(name="two_cx")

        qc = QuantumCircuit(2)
        qc.append(gate, qc.qubits)

        # Without transpilation the composite gate is one layer
        assert qc.depth(transpile=False) == 1
        # With transpilation the two CX gates are visible
        assert qc.depth(transpile=True) == 2

    # ------------------------------------------------------------------ #
    # t_depth                                                            #
    # ------------------------------------------------------------------ #

    def test_t_depth_docstring_example_with_epsilon(self):
        """Docstring example: T + CX + RX(2π·3/2⁴) with ε=2⁻⁵ gives T-depth 16."""

        qc = QuantumCircuit(2)
        qc.t(0)
        qc.cx(0, 1)
        qc.rx(2 * np.pi * 3 / 2**4, 1)
        assert qc.t_depth(epsilon=2**-5) == 16

    def test_t_depth_docstring_example_auto_epsilon(self):
        """Docstring example: same circuit without explicit ε gives T-depth 22."""

        qc = QuantumCircuit(2)
        qc.t(0)
        qc.cx(0, 1)
        qc.rx(2 * np.pi * 3 / 2**4, 1)
        assert qc.t_depth() == 22

    def test_t_depth_empty_circuit(self):
        """An empty circuit has T-depth 0."""

        assert QuantumCircuit(2).t_depth() == 0

    def test_t_depth_only_t_gate(self):
        """A single T gate has T-depth 1."""

        qc = QuantumCircuit(1)
        qc.t(0)
        assert qc.t_depth(epsilon=1e-10) == 1

    def test_t_depth_only_cx_gate(self):
        """A single CX gate (no T gates) has T-depth 0."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        assert qc.t_depth(epsilon=1e-10) == 0

    def test_t_depth_parallel_t_gates(self):
        """T gates on independent qubits are parallel — T-depth is 1."""

        qc = QuantumCircuit(2)
        qc.t(0)
        qc.t(1)
        assert qc.t_depth(epsilon=1e-10) == 1

    def test_t_depth_sequential_t_gates(self):
        """T gates applied sequentially on the same qubit accumulate T-depth."""

        qc = QuantumCircuit(1)
        qc.t(0)
        qc.t(0)
        qc.t(0)
        assert qc.t_depth(epsilon=1e-10) == 3

    # ------------------------------------------------------------------ #
    # cnot_depth                                                         #
    # ------------------------------------------------------------------ #

    def test_cnot_depth_docstring_example(self):
        """Docstring example: 4-qubit circuit with mixed CX and single-qubit gates gives CNOT depth 3."""

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.x(1)
        qc.cx(1, 2)
        qc.y(2)
        qc.cx(2, 3)
        qc.cx(1, 0)
        assert qc.cnot_depth() == 3

    def test_cnot_depth_empty_circuit(self):
        """An empty circuit has CNOT depth 0."""

        assert QuantumCircuit(2).cnot_depth() == 0

    def test_cnot_depth_no_cx_gates(self):
        """A circuit with only single-qubit gates has CNOT depth 0."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.z(0)
        assert qc.cnot_depth() == 0

    def test_cnot_depth_parallel_cx_gates(self):
        """CX gates on independent qubit pairs are parallel — CNOT depth is 1."""

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        assert qc.cnot_depth() == 1

    def test_cnot_depth_sequential_cx_gates(self):
        """CX gates sharing a qubit are sequential — CNOT depth equals the chain length."""

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        assert qc.cnot_depth() == 2

    # ------------------------------------------------------------------ #
    # num_qubits                                                         #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("n", [0, 1, 4, 5, 10])
    def test_num_qubits_various_sizes(self, n):
        """num_qubits returns the correct count for circuits of various sizes."""

        assert QuantumCircuit(n).num_qubits() == n

    def test_num_qubits_after_add_qubit(self):
        """num_qubits updates correctly after add_qubit is called."""

        qc = QuantumCircuit(2)
        assert qc.num_qubits() == 2
        qc.add_qubit()
        assert qc.num_qubits() == 3

    def test_num_qubits_does_not_count_clbits(self):
        """Classical bits are not included in the num_qubits count."""

        qc = QuantumCircuit(3, 5)
        assert qc.num_qubits() == 3

    # ------------------------------------------------------------------ #
    # cnot_count                                                         #
    # ------------------------------------------------------------------ #

    def test_cnot_count_basic_cx(self):
        """CX gates are counted correctly."""

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        assert qc.cnot_count() == 2

    def test_cnot_count_includes_cy_and_cz(self):
        """CX, CY, and CZ are all counted as CNOT-equivalent gates."""

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cy(0, 1)
        qc.cz(0, 1)
        assert qc.cnot_count() == 3

    def test_cnot_count_ignores_single_qubit_gates(self):
        """Single-qubit gates (H, X, Z, …) are not counted."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.z(0)
        qc.cx(0, 1)
        assert qc.cnot_count() == 1

    def test_cnot_count_empty_circuit(self):
        """An empty circuit has a CNOT count of 0."""

        assert QuantumCircuit(2).cnot_count() == 0

    def test_cnot_count_transpiles_composite_gates(self):
        """CX gates inside a composite gate are counted after transpilation."""

        inner = QuantumCircuit(2)
        inner.cx(0, 1)
        inner.cx(1, 0)
        gate = inner.to_gate(name="two_cx")

        qc = QuantumCircuit(2)
        qc.append(gate, qc.qubits)
        assert qc.cnot_count() == 2

    # ------------------------------------------------------------------ #
    # transpile                                                          #
    # ------------------------------------------------------------------ #

    def test_transpile_level_zero_is_noop(self):
        """transpile(level=0) leaves composite gates untouched."""

        inner = QuantumCircuit(1)
        inner.h(0)
        outer = QuantumCircuit(1)
        outer.append(inner.to_gate(name="inner"), [outer.qubits[0]])

        result = outer.transpile(transpilation_level=0)
        assert result.data[0].op.name == outer.data[0].op.name

    def test_transpile_level_one_decomposes_one_level(self):
        """transpile(level=1) unwraps exactly one layer of nesting."""

        inner = QuantumCircuit(1)
        inner.h(0)
        outer = QuantumCircuit(1)
        outer.append(inner.to_gate(name="inner"), [outer.qubits[0]])
        wrapped = QuantumCircuit(1)
        wrapped.append(outer.to_gate(name="outer"), [wrapped.qubits[0]])

        result = wrapped.transpile(transpilation_level=1)
        assert result.data[0].op.name == "inner"

    def test_transpile_full_decomposes_to_primitives(self):
        """transpile() with default level decomposes all nested gates to primitives."""

        inner = QuantumCircuit(1)
        inner.h(0)
        outer = QuantumCircuit(1)
        outer.append(inner.to_gate(name="inner"), [outer.qubits[0]])
        wrapped = QuantumCircuit(1)
        wrapped.append(outer.to_gate(name="outer"), [wrapped.qubits[0]])

        result = wrapped.transpile()
        assert all(instr.op.definition is None for instr in result.data)

    def test_transpile_does_not_mutate_original(self):
        """transpile() returns a new circuit and does not modify the source."""

        inner = QuantumCircuit(1)
        inner.h(0)
        qc = QuantumCircuit(1)
        qc.append(inner.to_gate(name="inner"), [qc.qubits[0]])

        original_name = qc.data[0].op.name
        qc.transpile()
        assert qc.data[0].op.name == original_name

    def test_transpile_preserves_unitary(self):
        """The unitary is unchanged by full transpilation."""

        inner = QuantumCircuit(1)
        inner.h(0)
        qc = QuantumCircuit(1)
        qc.append(inner.to_gate(name="wrap"), [qc.qubits[0]])

        assert qc.compare_unitary(qc.transpile()) is True

    # ------------------------------------------------------------------ #
    # count_ops                                                          #
    # ------------------------------------------------------------------ #

    def test_count_ops_docstring_example(self):
        """Reproduce the count_ops docstring example exactly."""

        qc = QuantumCircuit(5)
        qc.x(qc.qubits)
        qc.cx(0, range(1, 5))
        qc.z(1)
        qc.t(range(5))
        assert qc.count_ops() == {"x": 5, "cx": 4, "z": 1, "t": 5}

    def test_count_ops_empty_circuit(self):
        """count_ops on an empty circuit returns an empty dict."""

        assert QuantumCircuit(2).count_ops() == {}

    def test_count_ops_excludes_alloc_instructions(self):
        """qb_alloc and qb_dealloc are not included in the result."""

        qc = QuantumCircuit(2)
        qc.h(0)
        ops = qc.count_ops()
        assert "qb_alloc" not in ops
        assert "qb_dealloc" not in ops

    def test_count_ops_multiple_same_gate(self):
        """Multiple instances of the same gate are accumulated into one entry."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        assert qc.count_ops()["h"] == 3

    def test_count_ops_mixed_gates(self):
        """Different gate types each get their own counter."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(1)
        ops = qc.count_ops()
        assert ops == {"h": 1, "cx": 1, "x": 1}

    # ------------------------------------------------------------------ #
    # control                                                            #
    # ------------------------------------------------------------------ #

    def test_control_returns_operation_with_extra_qubit(self):
        """control(n) returns an Operation with n additional control qubits."""

        qc = QuantumCircuit(1)
        qc.x(0)
        ctrl_op = qc.control(1)
        assert ctrl_op.num_qubits == 2

    def test_control_two_controls(self):
        """control(2) on a single-qubit circuit produces a 3-qubit operation."""

        qc = QuantumCircuit(1)
        qc.x(0)
        ctrl_op = qc.control(2)
        assert ctrl_op.num_qubits == 3

    def test_control_single_x_equals_cx_unitary(self):
        """Controlled-X equals the standard CX unitary."""

        qc = QuantumCircuit(1)
        qc.x(0)
        ctrl_op = qc.control(1)

        test_qc = QuantumCircuit(2)
        test_qc.append(ctrl_op, test_qc.qubits)

        cx_qc = QuantumCircuit(2)
        cx_qc.cx(0, 1)

        assert test_qc.compare_unitary(cx_qc) is True

    # ------------------------------------------------------------------ #
    # compose                                                            #
    # ------------------------------------------------------------------ #

    def test_compose_inplace_returns_none_and_modifies_self(self):
        """compose(inplace=True) returns None and appends to self."""

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc2 = QuantumCircuit(2)
        qc2.cx(0, 1)

        result = qc1.compose(qc2, qubits=qc1.qubits)
        assert result is None
        assert len(qc1.data) == 2  # h + composed gate

    def test_compose_not_inplace_returns_new_circuit(self):
        """compose(inplace=False) returns a new circuit without mutating self."""

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc2 = QuantumCircuit(2)
        qc2.cx(0, 1)

        result = qc1.compose(qc2, qubits=qc1.qubits, inplace=False)

        assert result is not None
        assert result is not qc1
        assert len(qc1.data) == 1  # original unchanged
        assert len(result.data) == 2

    def test_compose_preserves_unitary(self):
        """The composed circuit has the same unitary as building the circuit manually."""

        qc_a = QuantumCircuit(2)
        qc_a.h(0)
        qc_b = QuantumCircuit(2)
        qc_b.cx(0, 1)

        composed = qc_a.compose(qc_b, qubits=qc_a.qubits, inplace=False)

        manual = QuantumCircuit(2)
        manual.h(0)
        manual.cx(0, 1)

        assert composed.compare_unitary(manual) is True

    # ------------------------------------------------------------------ #
    # bind_parameters                                                    #
    # ------------------------------------------------------------------ #

    def test_bind_parameters_removes_abstract_params(self):
        """After binding all parameters, abstract_params is empty."""

        a, b = symbols("a b")
        qc = QuantumCircuit(2)
        qc.p(a, 0)
        qc.p(b, 1)
        bound = qc.bind_parameters({a: np.pi / 2, b: np.pi})
        assert bound.abstract_params == set()

    def test_bind_parameters_does_not_mutate_original(self):
        """bind_parameters returns a new circuit without modifying self."""

        a = symbols("a")
        qc = QuantumCircuit(1)
        qc.p(a, 0)
        qc.bind_parameters({a: 1.0})
        assert a in qc.abstract_params

    def test_bind_parameters_partial_raises(self):
        """Binding only a subset of parameters raises a KeyError for the missing symbol."""

        a, b = symbols("a b")
        qc = QuantumCircuit(2)
        qc.p(a, 0)
        qc.p(b, 1)
        with pytest.raises(KeyError):
            qc.bind_parameters({a: 1.0})

    def test_bind_parameters_produces_correct_unitary(self):
        """Binding φ=π to a phase gate P(φ) gives the Pauli-Z unitary."""

        phi = symbols("phi")
        qc = QuantumCircuit(1)
        qc.p(phi, 0)
        bound = qc.bind_parameters({phi: np.pi})

        z_qc = QuantumCircuit(1)
        z_qc.z(0)
        assert bound.compare_unitary(z_qc) is True

    # ------------------------------------------------------------------ #
    # to_latex                                                           #
    # ------------------------------------------------------------------ #

    @pytest.mark.skip(reason="The 'pylatexenc' library might not be installed.")
    def test_to_latex_returns_string(self):
        """to_latex returns a non-empty string."""

        qc = QuantumCircuit(1)
        qc.h(0)
        result = qc.to_latex()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skip(reason="The 'pylatexenc' library might not be installed.")
    def test_to_latex_contains_latex_commands(self):
        """The returned string contains standard LaTeX document commands."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = qc.to_latex()
        assert "\\documentclass" in result

    # ------------------------------------------------------------------ #
    # to_qasm2                                                           #
    # ------------------------------------------------------------------ #

    def test_to_qasm2_header(self):
        """to_qasm2 output begins with the OpenQASM 2.0 header."""

        qc = QuantumCircuit(1)
        qc.h(0)
        assert qc.to_qasm2().startswith("OPENQASM 2.0")

    def test_to_qasm2_contains_gate_name(self):
        """to_qasm2 output contains the applied gate name."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qasm = qc.to_qasm2()
        assert "h " in qasm
        assert "cx " in qasm

    def test_to_qasm2_roundtrip(self):
        """A circuit exported to QASM 2.0 and re-imported has the same unitary."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        reloaded = QuantumCircuit.from_qasm_str(qc.to_qasm2())
        assert qc.compare_unitary(reloaded) is True

    # ------------------------------------------------------------------ #
    # to_qasm3                                                           #
    # ------------------------------------------------------------------ #

    def test_to_qasm3_header(self):
        """to_qasm3 output begins with the OpenQASM 3.0 header."""

        qc = QuantumCircuit(1)
        qc.h(0)
        assert qc.to_qasm3().startswith("OPENQASM 3.0")

    def test_to_qasm3_contains_gate_name(self):
        """to_qasm3 output contains the applied gate name."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qasm = qc.to_qasm3()
        assert "h " in qasm
        assert "cx " in qasm

    def test_to_qasm3_writes_file(self):
        """to_qasm3 with filename writes the same content to disk."""

        qc = QuantumCircuit(1)
        qc.h(0)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".qasm", delete=False) as tmp:
            fname = tmp.name
        try:
            qasm_str = qc.to_qasm3(filename=fname)
            assert open(fname).read() == qasm_str
        finally:
            os.unlink(fname)

    # ------------------------------------------------------------------ #
    # qasm                                                               #
    # ------------------------------------------------------------------ #

    def test_qasm_alias_for_to_qasm2(self):
        """qasm() produces the same output as to_qasm2()."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert qc.qasm() == qc.to_qasm2()

    # ------------------------------------------------------------------ #
    # measure                                                            #
    # ------------------------------------------------------------------ #

    def test_measure_single_int_auto_allocates_one_clbit(self):
        """measure(int) allocates exactly one new classical bit."""

        qc = QuantumCircuit(2)
        qc.measure(0)
        assert len(qc.clbits) == 1

    def test_measure_single_qubit_object_auto_allocates_one_clbit(self):
        """measure(Qubit) allocates exactly one new classical bit."""

        qc = QuantumCircuit(2)
        qc.measure(qc.qubits[1])
        assert len(qc.clbits) == 1

    def test_measure_list_auto_allocates_one_clbit_per_qubit(self):
        """measure(list) allocates one classical bit per qubit in the list."""

        qc = QuantumCircuit(3)
        qc.measure([0, 1, 2])
        assert len(qc.clbits) == 3

    def test_measure_range_auto_allocates_one_clbit_per_qubit(self):
        """measure(range) allocates one classical bit per qubit in the range."""

        qc = QuantumCircuit(4)
        qc.measure(range(4))
        assert len(qc.clbits) == 4

    def test_measure_qubits_list_auto_allocates_matching_clbits(self):
        """measure(qc.qubits) allocates a classical bit for every qubit."""

        qc = QuantumCircuit(5)
        qc.measure(qc.qubits)
        assert len(qc.clbits) == len(qc.qubits)

    def test_measure_explicit_clbit_no_new_allocation(self):
        """Providing an explicit Clbit does not allocate additional classical bits."""

        qc = QuantumCircuit(2)
        cb = qc.add_clbit()
        clbits_before = len(qc.clbits)
        qc.measure(0, cb)
        assert len(qc.clbits) == clbits_before

    def test_measure_explicit_clbit_list_no_new_allocation(self):
        """Providing an explicit list of Clbits does not allocate additional bits."""

        qc = QuantumCircuit(3)
        cb0, cb1, cb2 = qc.add_clbit(), qc.add_clbit(), qc.add_clbit()
        clbits_before = len(qc.clbits)
        qc.measure([0, 1, 2], [cb0, cb1, cb2])
        assert len(qc.clbits) == clbits_before

    def test_measure_single_appends_one_instruction(self):
        """Measuring a single qubit appends exactly one instruction."""

        qc = QuantumCircuit(1)
        qc.measure(0)
        assert len(qc.data) == 1

    def test_measure_list_appends_one_instruction_per_qubit(self):
        """Measuring a list of n qubits appends exactly n instructions."""

        qc = QuantumCircuit(3)
        qc.measure([0, 1, 2])
        assert len(qc.data) == 3

    def test_measure_instruction_has_measure_op_name(self):
        """Each instruction appended by measure has the operation name 'measure'."""

        qc = QuantumCircuit(2)
        qc.measure([0, 1])
        assert all(instr.op.name == "measure" for instr in qc.data)

    def test_measure_auto_clbits_paired_with_correct_qubits(self):
        """Each auto-allocated Clbit is paired with the corresponding Qubit."""

        qc = QuantumCircuit(3)
        qc.measure([0, 1, 2])

        for i, instr in enumerate(qc.data):
            assert instr.qubits == [qc.qubits[i]]
            assert instr.clbits == [qc.clbits[i]]

    def test_measure_explicit_clbit_used_in_instruction(self):
        """The explicit Clbit provided to measure appears in the instruction."""

        qc = QuantumCircuit(2)
        cb = qc.add_clbit()
        qc.measure(0, cb)
        assert qc.data[-1].clbits == [cb]

    def test_measure_explicit_clbit_list_preserves_pairing(self):
        """Explicit Clbits are paired with qubits in the order given."""

        qc = QuantumCircuit(2)
        cb0, cb1 = qc.add_clbit(), qc.add_clbit()
        qc.measure([0, 1], [cb0, cb1])

        assert qc.data[0].clbits == [cb0]
        assert qc.data[1].clbits == [cb1]

    # ------------------------------------------------------------------ #
    # parity                                                             #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "get_clbits",
        [
            lambda qc: qc.clbits,
            lambda qc: list(qc.clbits),
            lambda qc: tuple(qc.clbits),
        ],
        ids=["qc.clbits", "list", "tuple"],
    )
    def test_parity_accepts_sequence_types(self, get_clbits):
        """parity() accepts any sequence of Clbits (list, tuple, qc.clbits)."""

        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])
        handle = qc.parity(get_clbits(qc))
        assert len(handle.clbits) == 2

    def test_parity_accepts_single_clbit(self):
        """A single Clbit (not wrapped in a list) is accepted and treated as a
        length-1 parity check."""

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        handle = qc.parity(qc.clbits[0])
        assert len(handle.clbits) == 1

    def test_parity_appends_one_instruction(self):
        """Each call to parity() appends exactly one instruction to the circuit."""

        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])
        n_before = len(qc.data)
        qc.parity(qc.clbits)
        assert len(qc.data) == n_before + 1

    def test_parity_instruction_op_name(self):
        """The appended instruction has op name 'parity'."""

        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])
        qc.parity(qc.clbits)
        assert qc.data[-1].op.name == "parity"

    def test_parity_handle_clbits_match_input(self):
        """The handle's clbits are exactly the Clbits passed to parity(), in order."""

        qc = QuantumCircuit(3, 3)
        qc.measure([0, 1, 2], [0, 1, 2])
        handle = qc.parity(qc.clbits)
        assert handle.clbits == list(qc.clbits)

    @pytest.mark.parametrize("expectation", [0, 1])
    def test_parity_expectation_stored(self, expectation):
        """The expectation kwarg is stored in the handle."""

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        handle = qc.parity(qc.clbits[0], expectation=expectation)
        assert handle.expectation == expectation

    @pytest.mark.parametrize("observable", [False, True])
    def test_parity_observable_flag_stored(self, observable):
        """The observable kwarg is stored in the handle."""

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        handle = qc.parity(qc.clbits[0], observable=observable)
        assert handle.observable == observable

    def test_parity_multiple_calls_give_distinct_handles(self):
        """Each call to parity() returns a distinct handle pointing to its own
        instruction."""

        qc = QuantumCircuit(3, 3)
        qc.measure([0, 1, 2], [0, 1, 2])
        h0 = qc.parity([qc.clbits[0], qc.clbits[1]])
        h1 = qc.parity([qc.clbits[1], qc.clbits[2]])
        assert h0.instruction is not h1.instruction

    def test_parity_stim_detector_index(self):
        """A non-observable parity call maps to a Stim DETECTOR with index 0."""

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        handle = qc.parity(qc.clbits, expectation=0)
        _, _, det_map = qc.to_stim(
            return_measurement_map=True,
            return_detector_map=True,
        )
        assert handle in det_map
        assert det_map[handle] == 0

    def test_parity_stim_multiple_detectors_get_sequential_indices(self):
        """Two parity detectors are assigned consecutive Stim detector indices."""

        qc = QuantumCircuit(3, 3)
        qc.measure([0, 1, 2], [0, 1, 2])
        h0 = qc.parity([qc.clbits[0], qc.clbits[1]])
        h1 = qc.parity([qc.clbits[1], qc.clbits[2]])
        _, _, det_map = qc.to_stim(
            return_measurement_map=True,
            return_detector_map=True,
        )
        assert det_map[h0] == 0
        assert det_map[h1] == 1

    def test_parity_stim_observable_index(self):
        """An observable=True parity call maps to a Stim OBSERVABLE_INCLUDE."""

        qc = QuantumCircuit(2, 2)
        qc.measure([0, 1], [0, 1])
        handle = qc.parity(qc.clbits, observable=True)
        _, _, det_map, obs_map = qc.to_stim(
            return_measurement_map=True,
            return_detector_map=True,
            return_observable_map=True,
        )
        assert handle not in det_map
        assert handle in obs_map
        assert obs_map[handle] == 0


class TestQuantumCircuitDunderMethods:
    """Tests for QuantumCircuit dunder methods."""

    # ------------------------------------------------------------------ #
    # __hash__                                                           #
    # ------------------------------------------------------------------ #

    def test_empty_circuit_is_hashable(self):
        """An empty circuit (no instructions) can be hashed without error."""
        qc = QuantumCircuit(3)
        assert isinstance(hash(qc), int)

    def test_hash_is_stable(self):
        """Calling hash() on the same object twice returns the same value."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        assert hash(qc) == hash(qc)

    def test_circuit_usable_as_dict_key(self):
        """A circuit can be stored and retrieved as a dictionary key."""
        qc = QuantumCircuit(2)
        qc.h(0)
        d = {qc: "value"}
        assert d[qc] == "value"

    def test_circuit_usable_in_set(self):
        """A circuit can be added to a set. The same object is not duplicated."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        s = {qc, qc}
        assert len(s) == 1

    @pytest.mark.parametrize(
        "build",
        [
            # Simple single-qubit circuit
            lambda: ((lambda qc: (qc.h(0), qc)[-1])(QuantumCircuit(1))),
            # Multi-qubit entangling circuit
            lambda: (
                (lambda qc: (qc.h(0), qc.cx(0, 1), qc.cx(1, 2), qc)[-1])(
                    QuantumCircuit(3)
                )
            ),
            # Parametrized rotation
            lambda: ((lambda qc: (qc.rx(np.pi / 4, 0), qc)[-1])(QuantumCircuit(2))),
        ],
    )
    def test_identical_circuits_have_equal_hash(self, build):
        """Two circuits built with the same operations always hash identically."""
        assert hash(build()) == hash(build())

    def test_same_structure_different_qubit_names_equal_hash(self):
        """Qubit names are irrelevant. Only positional structure affects the hash."""
        qc1 = QuantumCircuit()
        qb_a = qc1.add_qubit(Qubit("alice"))
        qb_b = qc1.add_qubit(Qubit("bob"))
        qc1.cx(qb_a, qb_b)

        qc2 = QuantumCircuit()
        qb_x = qc2.add_qubit(Qubit("x"))
        qb_y = qc2.add_qubit(Qubit("y"))
        qc2.cx(qb_x, qb_y)

        assert hash(qc1) == hash(qc2)

    def test_composite_gate_equivalent_definitions_equal_hash(self):
        """Two circuits using structurally equivalent composite gates hash the same."""
        sub1 = QuantumCircuit(2)
        sub1.h(0)
        sub1.cx(0, 1)

        sub2 = QuantumCircuit(2)
        sub2.h(0)
        sub2.cx(0, 1)

        qc1 = QuantumCircuit(2)
        qc1.append(sub1.to_gate(), [0, 1])

        qc2 = QuantumCircuit(2)
        qc2.append(sub2.to_gate(), [0, 1])

        assert hash(qc1) == hash(qc2)

    def test_different_qubit_count_different_hash(self):
        """Circuits with a different number of qubits hash differently."""
        qc_small = QuantumCircuit(2)
        qc_small.h(0)
        qc_large = QuantumCircuit(4)
        qc_large.h(0)
        assert hash(qc_small) != hash(qc_large)

    def test_different_instruction_order_different_hash(self):
        """Reordering instructions changes the hash."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.x(1)

        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.h(0)

        assert hash(qc1) != hash(qc2)

    def test_different_qubit_wiring_different_hash(self):
        """Applying the same gate to different qubit positions changes the hash."""
        qc1 = QuantumCircuit(3)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(3)
        qc2.cx(0, 2)

        assert hash(qc1) != hash(qc2)

    def test_different_gate_same_position_different_hash(self):
        """Replacing one gate with another at the same qubit changes the hash."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(1)
        qc2.x(0)

        assert hash(qc1) != hash(qc2)

    def test_different_parameter_value_different_hash(self):
        """Changing a rotation angle changes the hash."""
        qc1 = QuantumCircuit(1)
        qc1.rx(np.pi / 4, 0)

        qc2 = QuantumCircuit(1)
        qc2.rx(np.pi / 3, 0)

        assert hash(qc1) != hash(qc2)

    def test_same_parameters_swapped_positions_different_hash(self):
        """The same parameter values in reversed instruction order hash differently."""
        qc1 = QuantumCircuit(1)
        qc1.rz(np.pi / 2, 0)
        qc1.rz(np.pi / 4, 0)

        qc2 = QuantumCircuit(1)
        qc2.rz(np.pi / 4, 0)
        qc2.rz(np.pi / 2, 0)

        assert hash(qc1) != hash(qc2)

    def test_composite_gate_different_definitions_different_hash(self):
        """Two circuits using composite gates with different sub-circuits hash differently."""
        sub1 = QuantumCircuit(2)
        sub1.h(0)
        sub1.cx(0, 1)

        sub2 = QuantumCircuit(2)
        sub2.h(1)  # gate applied to qubit 1 instead of qubit 0
        sub2.cx(0, 1)

        qc1 = QuantumCircuit(2)
        qc1.append(sub1.to_gate(), [0, 1])

        qc2 = QuantumCircuit(2)
        qc2.append(sub2.to_gate(), [0, 1])

        assert hash(qc1) != hash(qc2)

    # TODO: Implement tests for __str__, __repr__, __eq__, etc.
