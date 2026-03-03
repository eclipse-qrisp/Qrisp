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

from qrisp.circuit.quantum_circuit import QuantumCircuit
import pytest


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
