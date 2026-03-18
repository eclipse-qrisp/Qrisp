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

import pytest

from qrisp.circuit import Clbit, Qubit
from qrisp.circuit.quantum_circuit import QuantumCircuit


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

    # TODO: Add more tests for other examples in the docstring


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

    # TODO: Add more tests for other methods of QuantumCircuit
