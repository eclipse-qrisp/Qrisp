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

    # TODO: Add more tests for other methods of QuantumCircuit
