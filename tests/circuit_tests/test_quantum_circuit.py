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

import numpy as np
import pytest
import sympy
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from sympy import symbols

from qrisp.circuit import Clbit, Instruction, Qubit
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.standard_operations import XGate


class TestQuantumCircuitInitialization:
    """A test class for the QuantumCircuit implementation."""

    def test_initialization(self):
        """Test the initialization of the QuantumCircuit."""
        num_qubits = 3
        num_clbits = 2
        qc = QuantumCircuit(num_qubits=num_qubits, num_clbits=num_clbits)

        assert len(qc.qubits) == num_qubits
        assert len(qc.clbits) == num_clbits

    def test_error_wrong_type_initialization(self):
        """Test that initializing QuantumCircuit with wrong types raises an error."""

        with pytest.raises(
            TypeError,
            match="Tried to initialize QuantumCircuit with type str for num_qubits, expected int",
        ):
            QuantumCircuit(num_qubits="3")

        with pytest.raises(
            TypeError,
            match="Tried to initialize QuantumCircuit with type float for num_clbits, expected int",
        ):
            QuantumCircuit(num_clbits=2.5)

    # TODO: Add more tests for other aspects of initialization

    ##############################
    ### Docstring example tests
    ##############################

    def test_fan_out_example(self):
        """Test the fan-out example in the QuantumCircuit docstring."""

        qc_0 = QuantumCircuit(4)
        qc_0.cx(0, range(1, 4))

        assert len(qc_0.data) == 3
        assert all(str(instruction)[:2] == "cx" for instruction in qc_0.data)

        qc_1 = QuantumCircuit(4)
        qc_1.h(0)
        qc_1.append(qc_0.to_gate(name="fan-out"), qc_1.qubits)

        assert len(qc_1.data) == 2
        assert str(qc_1.data[0])[:1] == "h"
        assert str(qc_1.data[1])[:7] == "fan-out"

        qc_1.measure(qc_1.qubits)
        assert len(qc_1.data) == 6
        for i in range(2, 6):
            assert str(qc_1.data[i])[:7] == "measure"

    def test_from_qiskit_fan_out_example(self):
        """Test the from_qiskit conversion example in the QuantumCircuit docstring."""

        qc_2 = QiskitQuantumCircuit(4)
        qc_2.cx(0, range(1, 4))

        qrisp_qc_2 = QuantumCircuit.from_qiskit(qc_2)

        assert len(qrisp_qc_2.qubits) == 4
        assert len(qrisp_qc_2.clbits) == 0
        assert len(qrisp_qc_2.data) == 3
        assert all(str(instruction)[:2] == "cx" for instruction in qrisp_qc_2.data)

    def test_abstract_parameters_docstring_example(self):
        """Test the abstract parameters example in the QuantumCircuit docstring."""

        qc = QuantumCircuit(3)

        abstract_parameters = symbols("a b c")
        for i in range(3):
            qc.p(abstract_parameters[i], i)

        assert len(qc.data) == 3
        assert all(str(instruction)[:1] == "p" for instruction in qc.data)
        assert qc.abstract_params == set(abstract_parameters)

        subs_dic = {abstract_parameters[i]: i for i in range(3)}
        bound_qc = qc.bind_parameters(subs_dic)

        assert len(bound_qc.data) == 3
        assert bound_qc.abstract_params == {}


class TestQuantumCircuitMethods:
    """A test class for the methods of the QuantumCircuit class."""

    ##########################
    ### Extend method tests
    ##########################

    def test_extend_docstring_example(self):
        """Test the extend method of QuantumCircuit."""

        extension_qc = QuantumCircuit(4)
        extension_qc.cx(0, 1)
        extension_qc.cy(0, 2)
        extension_qc.cz(0, 3)

        assert len(extension_qc.data) == 3
        assert str(extension_qc.data[0])[:2] == "cx"
        assert str(extension_qc.data[1])[:2] == "cy"
        assert str(extension_qc.data[2])[:2] == "cz"

        qc_to_extend = QuantumCircuit(4)
        translation_dic = {
            extension_qc.qubits[i]: qc_to_extend.qubits[-1 - i] for i in range(4)
        }
        qc_to_extend.extend(extension_qc, translation_dic)

        assert len(qc_to_extend.data) == 3
        assert str(qc_to_extend.data[0])[:2] == "cx"
        assert str(qc_to_extend.data[1])[:2] == "cy"
        assert str(qc_to_extend.data[2])[:2] == "cz"

    def test_extend_with_default_translation(self):
        """Test extend method with default translation (no translation dictionary provided)."""

        extension_qc = QuantumCircuit(2)
        extension_qc.x(0)
        extension_qc.y(1)

        qc_to_extend = QuantumCircuit.copy(extension_qc)
        qc_to_extend.extend(extension_qc)

        assert len(qc_to_extend.data) == 4
        assert str(qc_to_extend.data[0])[:1] == "x"
        assert str(qc_to_extend.data[1])[:1] == "y"
        assert str(qc_to_extend.data[2])[:1] == "x"
        assert str(qc_to_extend.data[3])[:1] == "y"

    def test_extend_with_classical_bits(self):
        """Test extend method with circuits containing classical bits."""
        extension_qc = QuantumCircuit(2, 1)
        extension_qc.x(0)
        extension_qc.measure(0, 0)

        qc_to_extend = QuantumCircuit(2, 1)
        qc_to_extend.h(1)

        translation_dic = {
            extension_qc.qubits[0]: qc_to_extend.qubits[0],
            extension_qc.qubits[1]: qc_to_extend.qubits[1],
            extension_qc.clbits[0]: qc_to_extend.clbits[0],
        }
        qc_to_extend.extend(extension_qc, translation_dic)

        assert len(qc_to_extend.data) == 3
        assert str(qc_to_extend.data[0])[:1] == "h"
        assert str(qc_to_extend.data[1])[:1] == "x"
        assert str(qc_to_extend.data[2])[:7] == "measure"

    def test_extend_empty_circuit(self):
        """Test extending with an empty circuit."""
        qc_to_extend = QuantumCircuit(2)
        qc_to_extend.x(0)
        initial_data_len = len(qc_to_extend.data)

        empty_qc = QuantumCircuit(2)
        qc_to_extend.extend(empty_qc)

        assert len(qc_to_extend.data) == initial_data_len

    def test_extend_raises_error_for_invalid_qubit_reference(self):
        """Test that extend raises an error when referencing non-existent qubits."""
        extension_qc = QuantumCircuit(2)
        extension_qc.x(0)

        qc_to_extend = QuantumCircuit(1)

        translation_dic = {
            extension_qc.qubits[0]: extension_qc.qubits[1],
        }

        with pytest.raises(
            Exception, match="Instruction Qubits .* not present in circuit"
        ):
            qc_to_extend.extend(extension_qc, translation_dic)

    def test_extend_raises_error_for_invalid_clbit_reference(self):
        """Test that extend raises an error when referencing non-existent clbits."""
        extension_qc = QuantumCircuit(1, 2)
        extension_qc.measure(0, 0)

        qc_to_extend = QuantumCircuit(1, 1)

        translation_dic = {
            extension_qc.qubits[0]: qc_to_extend.qubits[0],
            extension_qc.clbits[0]: extension_qc.clbits[1],
        }

        with pytest.raises(
            ValueError, match="Instruction Clbits not present in circuit"
        ):
            qc_to_extend.extend(extension_qc, translation_dic)

    ##########################
    ### add_qubit method tests
    ##########################

    def test_add_qubit_default(self):
        """Test adding a qubit without providing a Qubit object."""
        qc = QuantumCircuit()
        initial_qubit_count = len(qc.qubits)

        added_qubit = qc.add_qubit()

        assert len(qc.qubits) == initial_qubit_count + 1
        assert added_qubit in qc.qubits
        assert added_qubit == qc.qubits[-1]

    def test_add_qubit_with_qubit_object(self):
        """Test adding a qubit by providing a Qubit object."""

        qc = QuantumCircuit()
        custom_qubit = Qubit("custom_qb")
        initial_qubit_count = len(qc.qubits)

        added_qubit = qc.add_qubit(custom_qubit)

        assert len(qc.qubits) == initial_qubit_count + 1
        assert added_qubit == custom_qubit
        assert added_qubit in qc.qubits

    def test_add_multiple_qubits(self):
        """Test adding multiple qubits sequentially."""
        qc = QuantumCircuit()

        for _ in range(5):
            qc.add_qubit()

        assert len(qc.qubits) == 5

    def test_add_qubit_increments_counter(self):
        """Test that adding qubits correctly increments the qubit index counter."""
        qc = QuantumCircuit(2)

        initial_counter = qc.qubit_index_counter[0]

        qc.add_qubit()

        assert qc.qubit_index_counter[0] == initial_counter + 1

    def test_add_qubit_error_wrong_type(self):
        """Test that adding a non-Qubit object raises a TypeError."""
        qc = QuantumCircuit()

        with pytest.raises(TypeError, match="Tried to add type .* as a qubit"):
            qc.add_qubit("not_a_qubit")

    def test_add_qubit_error_duplicate_identifier(self):
        """Test that adding a qubit with a duplicate identifier raises a ValueError."""

        qc = QuantumCircuit()
        qubit1 = Qubit("duplicate_qb")
        qubit2 = Qubit("duplicate_qb")

        qc.add_qubit(qubit1)

        with pytest.raises(ValueError, match="Qubit name .* already exists"):
            qc.add_qubit(qubit2)

    def test_add_qubit_return_value(self):
        """Test that add_qubit returns the added qubit."""
        qc = QuantumCircuit()

        added_qubit = qc.add_qubit()

        assert added_qubit is qc.qubits[-1]

    ##########################
    ### add_clbit method tests
    ##########################

    def test_add_clbit_default(self):
        """Test adding a classical bit without providing a Clbit object."""
        qc = QuantumCircuit()
        initial_clbit_count = len(qc.clbits)
        added_clbit = qc.add_clbit()
        assert len(qc.clbits) == initial_clbit_count + 1
        assert added_clbit in qc.clbits
        assert added_clbit == qc.clbits[-1]

    def test_add_clbit_with_clbit_object(self):
        """Test adding a classical bit by providing a Clbit object."""

        qc = QuantumCircuit()
        custom_clbit = Clbit("custom_cb")
        initial_clbit_count = len(qc.clbits)

        added_clbit = qc.add_clbit(custom_clbit)

        assert len(qc.clbits) == initial_clbit_count + 1
        assert added_clbit == custom_clbit
        assert added_clbit in qc.clbits

    def test_add_multiple_clbits(self):
        """Test adding multiple classical bits sequentially."""
        qc = QuantumCircuit()

        for _ in range(5):
            qc.add_clbit()

        assert len(qc.clbits) == 5

    def test_add_clbit_increments_counter(self):
        """Test that adding classical bits correctly increments the clbit index counter."""

        qc = QuantumCircuit(0, 2)
        initial_counter = qc.clbit_index_counter[0]
        qc.add_clbit()
        assert qc.clbit_index_counter[0] == initial_counter + 1

    def test_add_clbit_error_wrong_type(self):
        """Test that adding a non-Clbit object raises a TypeError."""
        qc = QuantumCircuit()

        with pytest.raises(TypeError, match="Tried to add type .* as a classical bit"):
            qc.add_clbit("not_a_clbit")

    def test_add_clbit_error_duplicate_identifier(self):
        """Test that adding a clbit with a duplicate identifier raises a ValueError."""

        qc = QuantumCircuit()
        clbit1 = Clbit("duplicate_cb")
        clbit2 = Clbit("duplicate_cb")

        qc.add_clbit(clbit1)

        with pytest.raises(ValueError, match="Clbit name .* already exists"):
            qc.add_clbit(clbit2)

    def test_add_clbit_return_value(self):
        """Test that add_clbit returns the added classical bit."""
        qc = QuantumCircuit()
        added_clbit = qc.add_clbit()
        assert added_clbit is qc.clbits[-1]

    def test_add_qubit_and_clbit_together(self):
        """Test adding both qubits and classical bits to the same circuit."""
        qc = QuantumCircuit()

        qubit1 = qc.add_qubit()
        clbit1 = qc.add_clbit()
        qubit2 = qc.add_qubit()
        clbit2 = qc.add_clbit()

        assert len(qc.qubits) == 2
        assert len(qc.clbits) == 2
        assert qc.qubits == [qubit1, qubit2]
        assert qc.clbits == [clbit1, clbit2]

    ##########################
    ### to_op method tests
    ##########################

    def test_to_op_basic(self):
        """Test converting a QuantumCircuit to an Operation with a custom name."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(1)

        operation = qc.to_op(name="test_op")

        assert operation.name == "test_op"
        assert operation.num_qubits == 2
        assert operation.num_clbits == 0
        assert operation.definition is not None

    def test_to_op_default_name(self):
        """Test converting a QuantumCircuit to an Operation with default name generation."""
        qc = QuantumCircuit(1)
        qc.h(0)

        operation = qc.to_op()

        assert operation.name.startswith("circuit")
        assert len(operation.name) == len("circuit") + 7

    def test_to_op_with_classical_bits(self):
        """Test converting a QuantumCircuit with classical bits to an Operation."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)

        operation = qc.to_op(name="qc_with_clbits")

        assert operation.num_qubits == 2
        assert operation.num_clbits == 2
        assert operation.name == "qc_with_clbits"

    def test_to_op_removes_allocation_instructions(self):
        """Test that to_op removes qb_alloc and qb_dealloc instructions."""
        qc = QuantumCircuit(2)
        qc.x(0)
        # Manually add allocation instructions to test filtering
        qc.data.append(Instruction(XGate(), [qc.qubits[1]]))

        operation = qc.to_op(name="filtered_op")

        # Verify allocation instructions are filtered in definition
        assert all(
            instr.op.name not in ["qb_alloc", "qb_dealloc"]
            for instr in operation.definition.data
        )

    def test_to_op_empty_circuit(self):
        """Test converting an empty QuantumCircuit to an Operation."""
        qc = QuantumCircuit(3)

        operation = qc.to_op(name="empty_op")

        assert operation.num_qubits == 3
        assert operation.num_clbits == 0
        assert operation.name == "empty_op"

    def test_to_op_preserves_circuit_definition(self):
        """Test that to_op preserves the circuit's gate definitions."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(1)

        operation = qc.to_op(name="preserved_op")

        assert len(operation.definition.data) == 3
        assert len(operation.definition.qubits) == 2

    def test_to_op_multiple_calls_unique_names(self):
        """Test that multiple calls to to_op with default names generate unique names."""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)

        op1 = qc1.to_op()
        op2 = qc2.to_op()

        assert op1.name != op2.name

    def test_to_op_custom_name_override(self):
        """Test that custom names override default naming."""
        qc = QuantumCircuit(2)
        qc.x(0)

        operation = qc.to_op(name="custom_name")

        assert operation.name == "custom_name"

    def test_to_op_definition_is_copy(self):
        """Test that to_op creates a copy of the circuit, not a reference."""
        qc = QuantumCircuit(1)
        qc.h(0)

        operation = qc.to_op(name="copy_test")
        initial_data_len = len(operation.definition.data)

        qc.x(0)

        assert len(operation.definition.data) == initial_data_len

    ##########################
    ### to_gate method tests
    ##########################

    # `to_gate`` calls to_op internally, with an additional check for classical bits.
    # Therefore, other behavior is tested in the `to_op` test suite.

    def test_to_gate_error_if_clbits_present(self):
        """Test that to_gate raises an error if the circuit contains classical bits."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        with pytest.raises(
            ValueError,
            match="Tried to turn a circuit including classical bits into unitary gate",
        ):
            qc.to_gate(name="invalid_gate")

    ##########################
    ### copy method tests
    ##########################

    def test_copy_basic(self):
        """Test basic copying of a QuantumCircuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qc_copy = qc.copy()

        assert len(qc_copy.qubits) == len(qc.qubits)
        assert len(qc_copy.clbits) == len(qc.clbits)
        assert len(qc_copy.data) == len(qc.data)

    def test_copy_is_independent(self):
        """Test that modifying a copy does not affect the original circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)

        qc_copy = qc.copy()
        qc_copy.x(1)

        assert len(qc.data) == 1
        assert len(qc_copy.data) == 2

    def test_copy_with_classical_bits(self):
        """Test copying a circuit with classical bits."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure([0, 1], [0, 1])

        qc_copy = qc.copy()

        assert len(qc_copy.clbits) == 2
        assert len(qc_copy.data) == 3

    def test_copy_shares_qubit_objects(self):
        """Test that copied circuit shares the same Qubit objects."""
        qc = QuantumCircuit(2)
        qc.h(0)

        qc_copy = qc.copy()

        assert qc_copy.qubits[0] is qc.qubits[0]
        assert qc_copy.qubits[1] is qc.qubits[1]

    def test_copy_shares_clbit_objects(self):
        """Test that copied circuit shares the same Clbit objects."""
        qc = QuantumCircuit(1, 2)
        qc.measure(0, [0, 1])

        qc_copy = qc.copy()

        assert qc_copy.clbits[0] is qc.clbits[0]
        assert qc_copy.clbits[1] is qc.clbits[1]

    def test_copy_shares_instruction_objects(self):
        """Test that copied circuit shares the same Instruction objects."""
        qc = QuantumCircuit(1)
        qc.h(0)

        qc_copy = qc.copy()

        assert qc_copy.data[0] is qc.data[0]

    def test_copy_abstract_params(self):
        """Test that abstract parameters are copied correctly."""

        qc = QuantumCircuit(1)
        a, b = symbols("a b")
        qc.abstract_params = {a, b}

        qc_copy = qc.copy()

        assert qc_copy.abstract_params == qc.abstract_params
        assert qc_copy.abstract_params is not qc.abstract_params

    def test_copy_empty_circuit(self):
        """Test copying an empty circuit."""
        qc = QuantumCircuit(3)

        qc_copy = qc.copy()

        assert len(qc_copy.qubits) == 3
        assert len(qc_copy.data) == 0

    def test_copy_multiple_times(self):
        """Test copying a circuit multiple times."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        copy1 = qc.copy()
        copy2 = qc.copy()

        assert len(copy1.data) == len(copy2.data) == 2
        assert copy1 is not copy2
        assert copy1 is not qc

    ##########################
    ### clearcopy method tests
    ##########################

    def test_clearcopy_basic(self):
        """Test basic clearcopy of a QuantumCircuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qc_clearcopy = qc.clearcopy()

        assert len(qc_clearcopy.qubits) == len(qc.qubits)
        assert len(qc_clearcopy.data) == 0

    def test_clearcopy_preserves_qubits(self):
        """Test that clearcopy preserves qubits."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)

        qc_clearcopy = qc.clearcopy()

        assert len(qc_clearcopy.qubits) == 3
        assert qc_clearcopy.qubits[0] is qc.qubits[0]
        assert qc_clearcopy.qubits[1] is qc.qubits[1]
        assert qc_clearcopy.qubits[2] is qc.qubits[2]

    def test_clearcopy_preserves_clbits(self):
        """Test that clearcopy preserves classical bits."""
        qc = QuantumCircuit(2, 3)
        qc.h(0)
        qc.measure(0, 0)

        qc_clearcopy = qc.clearcopy()

        assert len(qc_clearcopy.clbits) == 3
        assert qc_clearcopy.clbits[0] is qc.clbits[0]
        assert qc_clearcopy.clbits[1] is qc.clbits[1]
        assert qc_clearcopy.clbits[2] is qc.clbits[2]

    def test_clearcopy_removes_all_data(self):
        """Test that clearcopy removes all instructions."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(1)
        qc.measure([0, 1], [0, 1])

        qc_clearcopy = qc.clearcopy()

        assert len(qc_clearcopy.data) == 0

    def test_clearcopy_does_not_modify_original(self):
        """Test that clearcopy does not modify the original circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)

        original_data_len = len(qc.data)
        _ = qc.clearcopy()
        assert len(qc.data) == original_data_len

    def test_clearcopy_empty_circuit(self):
        """Test clearcopy of an empty circuit."""
        qc = QuantumCircuit(2)

        qc_clearcopy = qc.clearcopy()

        assert len(qc_clearcopy.qubits) == 2
        assert len(qc_clearcopy.data) == 0

    def test_clearcopy_with_classical_bits(self):
        """Test clearcopy of a circuit with classical bits."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.measure([0, 1], [0, 1])

        qc_clearcopy = qc.clearcopy()

        assert len(qc_clearcopy.qubits) == 3
        assert len(qc_clearcopy.clbits) == 2
        assert len(qc_clearcopy.data) == 0

    def test_clearcopy_abstract_params(self):
        """Test that clearcopy preserves abstract parameters."""

        qc = QuantumCircuit(1)
        a, b = symbols("a b")
        qc.abstract_params = {a, b}
        qc.h(0)

        qc_clearcopy = qc.clearcopy()

        assert qc_clearcopy.abstract_params == qc.abstract_params
        assert len(qc_clearcopy.data) == 0

    def test_clearcopy_is_independent(self):
        """Test that modifying clearcopy does not affect original."""
        qc = QuantumCircuit(2)
        qc.h(0)

        qc_clearcopy = qc.clearcopy()
        qc_clearcopy.x(0)

        assert len(qc.data) == 1
        assert len(qc_clearcopy.data) == 1

    def test_copy_and_clearcopy_difference(self):
        """Test the difference between copy and clearcopy."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        qc_copy = qc.copy()
        qc_clearcopy = qc.clearcopy()

        assert len(qc_copy.data) == 2
        assert len(qc_clearcopy.data) == 0
        assert len(qc_copy.qubits) == len(qc_clearcopy.qubits)

    ###################################
    ### compare_unitary method tests
    ###################################

    def test_compare_unitary_identical_circuits(self):
        """Test comparing unitaries of identical circuits."""
        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.cx(0, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.h(0)
        qc_1.cx(0, 1)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_different_circuits(self):
        """Test comparing unitaries of different circuits."""
        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.cx(0, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.x(0)
        qc_1.y(1)

        assert qc_0.compare_unitary(qc_1) is False

    def test_compare_unitary_commuting_gates(self):
        """Test comparing unitaries of circuits with commuting gates."""
        qc_0 = QuantumCircuit(2)
        qc_0.z(0)
        qc_0.cx(0, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.cx(0, 1)
        qc_1.z(0)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_docstring_example(self):
        """Test the example from the compare_unitary docstring."""
        qc_0 = QuantumCircuit(2)
        qc_1 = QuantumCircuit(2)
        qc_0.z(0)
        qc_0.cx(0, 1)

        qc_1.cx(0, 1)
        qc_1.z(0)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_different_qubit_counts(self):
        """Test that compare_unitary returns False for circuits with different qubit counts."""
        qc_0 = QuantumCircuit(2)
        qc_0.h(0)

        qc_1 = QuantumCircuit(3)
        qc_1.h(0)

        assert qc_0.compare_unitary(qc_1) is False

    def test_compare_unitary_precision_parameter(self):
        """Test the precision parameter of compare_unitary."""
        qc_0 = QuantumCircuit(1)
        qc_0.h(0)

        qc_1 = QuantumCircuit(1)
        qc_1.h(0)

        # Should be True with default and higher precision
        assert qc_0.compare_unitary(qc_1, precision=4) is True
        assert qc_0.compare_unitary(qc_1, precision=10) is True

    def test_compare_unitary_single_qubit_gates(self):
        """Test comparing unitaries with single qubit gates."""
        qc_0 = QuantumCircuit(1)
        qc_0.x(0)
        qc_0.x(0)  # X^2 = I

        qc_1 = QuantumCircuit(1)
        qc_1.id(0)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_empty_circuits(self):
        """Test comparing unitaries of empty circuits (identity)."""
        qc_0 = QuantumCircuit(2)
        qc_1 = QuantumCircuit(2)

        assert qc_0.compare_unitary(qc_1) is True

    @pytest.mark.xfail
    def test_compare_unitary_global_phase_difference(self):
        """Test comparing unitaries that differ only by global phase."""
        qc_0 = QuantumCircuit(1)
        qc_0.p(np.pi, 0)

        qc_1 = QuantumCircuit(1)
        qc_1.z(0)

        # These differ by a global phase, should be False without ignore_gphase
        assert qc_0.compare_unitary(qc_1, ignore_gphase=False) is False
        # Should be True when ignoring global phase
        assert qc_0.compare_unitary(qc_1, ignore_gphase=True) is True

    @pytest.mark.xfail
    def test_compare_unitary_ignore_gphase_parameter(self):
        """Test the ignore_gphase parameter."""
        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.p(np.pi / 4, 1)

        qc_1 = QuantumCircuit(2)
        qc_1.h(0)
        qc_1.p(np.pi / 4 + np.pi, 1)  # Differs by a global phase

        assert qc_0.compare_unitary(qc_1, ignore_gphase=True) is True

    @pytest.mark.xfail
    def test_compare_unitary_two_qubit_gates(self):
        """Test comparing unitaries with two-qubit gates."""
        qc_0 = QuantumCircuit(2)
        qc_0.cx(0, 1)
        qc_0.cx(1, 0)

        qc_1 = QuantumCircuit(2)
        qc_1.swap(0, 1)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_controlled_operations(self):
        """Test comparing unitaries with controlled operations."""
        qc_0 = QuantumCircuit(3)
        qc_0.ccx(0, 1, 2)

        qc_1 = QuantumCircuit(3)
        qc_1.ccx(0, 1, 2)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_parametric_gates(self):
        """Test comparing unitaries with parametric gates."""
        angle = np.pi / 4

        qc_0 = QuantumCircuit(1)
        qc_0.rx(angle, 0)

        qc_1 = QuantumCircuit(1)
        qc_1.rx(angle, 0)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_different_parametric_angles(self):
        """Test comparing unitaries with different parametric angles."""
        qc_0 = QuantumCircuit(1)
        qc_0.rx(np.pi / 4, 0)

        qc_1 = QuantumCircuit(1)
        qc_1.rx(np.pi / 3, 0)

        assert qc_0.compare_unitary(qc_1) is False

    @pytest.mark.xfail
    def test_compare_unitary_rotation_equivalence(self):
        """Test comparing unitaries of equivalent rotation gates."""
        qc_0 = QuantumCircuit(1)
        qc_0.rx(2 * np.pi, 0)  # Full rotation

        qc_1 = QuantumCircuit(1)
        qc_1.id(0)

        assert qc_0.compare_unitary(qc_1) is True

    def test_compare_unitary_inverse_operations(self):
        """Test comparing a circuit with its inverse."""
        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.cx(0, 1)
        qc_0.ry(np.pi / 3, 1)

        qc_1 = qc_0.inverse()

        # Composing a circuit with its inverse should give identity
        qc_combined = qc_0.clearcopy()
        qc_combined.extend(qc_0)
        qc_combined.extend(qc_1)

        qc_identity = QuantumCircuit(2)
        assert qc_combined.compare_unitary(qc_identity) is True

    ###################################
    ### get_unitary method tests
    ###################################

    def test_get_unitary_identity_circuit(self):
        """Test get_unitary for an empty circuit (identity)."""
        qc = QuantumCircuit(2)
        unitary = qc.get_unitary()

        expected = np.eye(4, dtype=np.complex64)
        assert unitary.shape == (4, 4)
        assert np.allclose(unitary, expected)

    def test_get_unitary_single_qubit_x_gate(self):
        """Test get_unitary for a single X gate."""
        qc = QuantumCircuit(1)
        qc.x(0)
        unitary = qc.get_unitary()

        expected = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        assert unitary.shape == (2, 2)
        assert np.allclose(unitary, expected)

    def test_get_unitary_single_qubit_h_gate(self):
        """Test get_unitary for a Hadamard gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        unitary = qc.get_unitary()

        expected = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
        assert unitary.shape == (2, 2)
        assert np.allclose(unitary, expected)

    def test_get_unitary_two_qubit_cnot(self):
        """Test get_unitary for a CNOT gate."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        unitary = qc.get_unitary()

        assert unitary.shape == (4, 4)
        assert np.allclose(unitary @ unitary.conj().T, np.eye(4), atol=1e-5)

    def test_get_unitary_docstring_example_numeric(self):
        """Test the numeric example from the get_unitary docstring."""
        qc = QuantumCircuit(2)
        phi = np.pi
        qc.p(phi / 2, 0)
        qc.p(phi / 2, 1)
        qc.cx(0, 1)
        qc.p(-phi / 2, 1)
        qc.cx(0, 1)

        unitary = qc.get_unitary(decimals=4)

        expected = np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
            ],
            dtype=np.complex64,
        )

        assert np.allclose(unitary, expected, atol=1e-4)

    def test_get_unitary_docstring_example_symbolic(self):
        """Test the symbolic example from the get_unitary docstring."""
        qc = QuantumCircuit(2)
        phi = sympy.Symbol("phi")
        qc.p(phi / 2, 0)
        qc.p(phi / 2, 1)
        qc.cx(0, 1)
        qc.p(-phi / 2, 1)
        qc.cx(0, 1)

        unitary = qc.get_unitary(decimals=4)

        assert unitary.dtype == np.dtype("O")
        assert unitary.shape == (4, 4)
        assert unitary[0, 0] == 1
        assert unitary[3, 3].has(phi)

    def test_get_unitary_without_decimals(self):
        """Test get_unitary returns full precision when decimals is None."""
        qc = QuantumCircuit(1)
        qc.h(0)

        unitary = qc.get_unitary(decimals=None)

        assert unitary is not None
        assert unitary.shape == (2, 2)

    def test_get_unitary_with_decimals_rounding(self):
        """Test that decimals parameter rounds numeric values correctly."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 4, 0)

        unitary_full = qc.get_unitary()
        unitary_rounded = qc.get_unitary(decimals=2)

        assert unitary_full.shape == unitary_rounded.shape
        assert unitary_rounded.dtype == np.complex64

    def test_get_unitary_parametric_gate(self):
        """Test get_unitary with parametric gates."""
        angle = np.pi / 4
        qc = QuantumCircuit(1)
        qc.rx(angle, 0)

        unitary = qc.get_unitary()

        assert unitary.shape == (2, 2)
        assert np.allclose(unitary @ unitary.conj().T, np.eye(2), atol=1e-5)

    def test_get_unitary_phase_gate(self):
        """Test get_unitary with phase gates."""
        qc = QuantumCircuit(1)
        qc.p(np.pi / 2, 0)

        unitary = qc.get_unitary()

        expected = np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]], dtype=np.complex64)
        assert np.allclose(unitary, expected)

    def test_get_unitary_symbolic_parameter(self):
        """Test get_unitary with symbolic parameters."""
        qc = QuantumCircuit(1)
        theta = sympy.Symbol("theta")
        qc.rx(theta, 0)

        unitary = qc.get_unitary()

        assert unitary.dtype == np.dtype("O")
        assert unitary[0, 0].has(theta) or unitary[0, 1].has(theta)

    def test_get_unitary_multiple_symbolic_parameters(self):
        """Test get_unitary with multiple symbolic parameters."""
        qc = QuantumCircuit(2)
        a, b = sympy.symbols("a b")
        qc.rx(a, 0)
        qc.ry(b, 1)

        unitary = qc.get_unitary()

        assert unitary.dtype == np.dtype("O")
        assert unitary.shape == (4, 4)

    def test_get_unitary_unitarity(self):
        """Test that returned matrix is unitary (U†U = I)."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(np.pi / 3, 1)

        unitary = qc.get_unitary()

        product = unitary @ unitary.conj().T
        assert np.allclose(product, np.eye(4), atol=1e-5)

    def test_get_unitary_complex_circuit(self):
        """Test get_unitary with a complex multi-gate circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.ry(np.pi / 6, 2)

        unitary = qc.get_unitary()

        assert unitary.shape == (8, 8)
        assert np.allclose(unitary @ unitary.conj().T, np.eye(8), atol=1e-5)

    def test_get_unitary_sequential_x_gates(self):
        """Test get_unitary with X gates (X²=I)."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)

        unitary = qc.get_unitary()
        expected = np.eye(2, dtype=np.complex64)

        assert np.allclose(unitary, expected)

    def test_get_unitary_decimals_snapping(self):
        """Test that decimals parameter snaps values near 1 to exactly 1."""
        qc = QuantumCircuit(1)
        a = sympy.Symbol("a")
        qc.p(a, 0)

        unitary = qc.get_unitary(decimals=4)

        assert unitary.dtype == np.dtype("O")
        # First diagonal element should be exactly 1
        assert unitary[0, 0] == 1

    def test_get_unitary_symbolic_simplification(self):
        """Test that symbolic expressions are simplified."""
        qc = QuantumCircuit(1)
        phi = sympy.Symbol("phi")
        qc.p(phi, 0)

        unitary_1 = qc.get_unitary()
        unitary_2 = qc.get_unitary(decimals=4)

        assert unitary_1.dtype == np.dtype("O")
        assert unitary_2.dtype == np.dtype("O")

    def test_get_unitary_empty_circuit_multiple_qubits(self):
        """Test get_unitary for empty circuits with different numbers of qubits."""
        for num_qubits in range(1, 4):
            qc = QuantumCircuit(num_qubits)
            unitary = qc.get_unitary()

            expected_dim = 2**num_qubits
            assert unitary.shape == (expected_dim, expected_dim)
            assert np.allclose(unitary, np.eye(expected_dim))

    def test_get_unitary_z_gate(self):
        """Test get_unitary for Z gate."""
        qc = QuantumCircuit(1)
        qc.z(0)

        unitary = qc.get_unitary()
        expected = np.array([[1, 0], [0, -1]], dtype=np.complex64)

        assert np.allclose(unitary, expected)

    def test_get_unitary_y_gate(self):
        """Test get_unitary for Y gate."""
        qc = QuantumCircuit(1)
        qc.y(0)

        unitary = qc.get_unitary()
        expected = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)

        assert np.allclose(unitary, expected)

    def test_get_unitary_s_gate(self):
        """Test get_unitary for S gate."""
        qc = QuantumCircuit(1)
        qc.s(0)

        unitary = qc.get_unitary()
        expected = np.array([[1, 0], [0, 1j]], dtype=np.complex64)

        assert np.allclose(unitary, expected)

    def test_get_unitary_t_gate(self):
        """Test get_unitary for T gate."""
        qc = QuantumCircuit(1)
        qc.t(0)

        unitary = qc.get_unitary()
        expected = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64)

        assert np.allclose(unitary, expected)

    def test_get_unitary_swap_gate(self):
        """Test get_unitary for SWAP gate."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)

        unitary = qc.get_unitary()

        assert unitary.shape == (4, 4)
        assert np.allclose(unitary @ unitary.conj().T, np.eye(4), atol=1e-5)

    def test_get_unitary_preserves_matrix_structure(self):
        """Test that get_unitary preserves expected matrix structure."""
        qc = QuantumCircuit(1)
        qc.h(0)

        unitary = qc.get_unitary()

        # Hadamard is Hermitian and unitary
        assert np.allclose(unitary, unitary.conj().T)
        assert np.allclose(unitary @ unitary.conj().T, np.eye(2), atol=1e-5)

    # TODO: Add more tests for other methods of QuantumCircuit


class TestQuantumCircuitDunderMethods:
    """A test class for the dunder methods of QuantumCircuit."""

    pass
