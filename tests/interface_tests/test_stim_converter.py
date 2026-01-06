"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from qrisp import QuantumCircuit

def test_basic_gates():
    """Test conversion of basic single-qubit gates."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(0)
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    # Check the circuit has the expected operations
    assert "H 0" in str(stim_circuit)
    assert "X 1" in str(stim_circuit)
    assert "Y 2" in str(stim_circuit)
    assert "Z 0" in str(stim_circuit)


def test_two_qubit_gates():
    """Test conversion of two-qubit gates."""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cz(1, 2)
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    assert "CX 0 1" in str(stim_circuit)
    assert "CZ 1 2" in str(stim_circuit)


def test_bell_state():
    """Test creating a Bell state."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1])
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    # Sample from the circuit
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # For a Bell state, we expect equal 00 and 11 outcomes (correlated)
    assert samples.shape == (100, 2)
    # Check that measurements are correlated
    assert (samples[:, 0] == samples[:, 1]).all()


def test_ghz_state():
    """Test creating a GHZ state."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2])
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    # Sample from the circuit
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # For a GHZ state, all measurements should be correlated (all 000 or all 111)
    assert samples.shape == (100, 3)
    assert (samples[:, 0] == samples[:, 1]).all()
    assert (samples[:, 1] == samples[:, 2]).all()


def test_s_and_sdg_gates():
    """Test S and S† gates."""
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.s_dg(1)
    qc.h(0)
    qc.h(1)
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    assert "S 0" in str(stim_circuit)
    assert "S_DAG 1" in str(stim_circuit)


def test_sx_and_sxdg_gates():
    """Test √X and √X† gates."""
    qc = QuantumCircuit(2)
    qc.sx(0)
    qc.sx_dg(1)
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    assert "SQRT_X 0" in str(stim_circuit)
    assert "SQRT_X_DAG 1" in str(stim_circuit)


def test_transpilation():
    """Test that transpilation works correctly."""
    from qrisp import QuantumVariable, h
    
    # Create a circuit with composite gates
    qv = QuantumVariable(2)
    h(qv)  # Hadamard on all qubits
    
    qc = qv.qs.compile()
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    # Verify the circuit was transpiled and contains Hadamard gates
    assert "H" in str(stim_circuit)


def test_classical_bit_mapping():
    """Test that classical bit indices are properly mapped."""
    # Create circuit with measurements into specific classical bits
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    
    # Measure into classical bits in non-sequential order
    # qubit 0 -> clbit 2
    # qubit 1 -> clbit 0  
    # qubit 2 -> clbit 1
    qc.measure(qc.qubits[0], qc.clbits[2])
    qc.measure(qc.qubits[1], qc.clbits[0])
    qc.measure(qc.qubits[2], qc.clbits[1])
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    # Verify the mapping contains all classical bits
    assert len(clbit_map) == 3
    assert qc.clbits[0] in clbit_map
    assert qc.clbits[1] in clbit_map
    assert qc.clbits[2] in clbit_map
    
    # Sample from the circuit
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(10)
    
    # Use the mapping to reorder samples to match Qrisp's classical bit order
    sorted_clbits = sorted(clbit_map.keys(), key=lambda cb: qc.clbits.index(cb))
    reordered_samples = samples[:, [clbit_map[cb] for cb in sorted_clbits]]
    
    # Verify: qubit 1 had X gate, so clbit 0 should always be True
    assert reordered_samples[:, 0].all(), "Clbit 0 should always be True (qubit 1 with X)"
    # Verify: qubit 2 had Y gate, so clbit 1 should always be True  
    assert reordered_samples[:, 1].all(), "Clbit 1 should always be True (qubit 2 with Y)"


def test_mid_circuit_measurement():
    """Test that mid-circuit measurements work correctly."""
    # Create circuit with mid-circuit measurement
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(qc.qubits[0], qc.clbits[0])  # Mid-circuit measurement
    qc.cx(0, 1)  # Use the measured qubit
    qc.measure(qc.qubits[1], qc.clbits[1])  # Final measurement
    
    stim_circuit, clbit_map = qc.to_stim(True)
    
    # Verify measurements are in the correct positions
    circuit_str = str(stim_circuit)
    lines = circuit_str.strip().split('\n')
    
    # Check that measurements appear in the circuit
    assert 'M' in circuit_str, "Should have measurements"
    # Verify we have the expected number of operations
    assert len(lines) == 4  # H, M, CX, M


def test_error_on_non_clifford():
    """Test that non-Clifford gates raise an error."""
    qc = QuantumCircuit(1)
    qc.t(0)  # T gate is not Clifford
    
    try:
        stim_circuit, clbit_map = qc.to_stim(True)
        assert False, "Expected NotImplementedError for T gate"
    except NotImplementedError:
        pass  # Expected behavior


def test_stim_errors():
    """Test conversion of Stim noise channels via StimError."""
    from qrisp.misc.stim_tools import StimError
    
    # Test 1-qubit errors
    qc = QuantumCircuit(1)
    
    # 1 parameter errors
    qc.append(StimError("DEPOLARIZE1", 0.1), [qc.qubits[0]])
    qc.append(StimError("X_ERROR", 0.2), [qc.qubits[0]])
    qc.append(StimError("Y_ERROR", 0.3), [qc.qubits[0]])
    qc.append(StimError("Z_ERROR", 0.4), [qc.qubits[0]])
    
    # 3 parameters (PAULI_CHANNEL_1)
    qc.append(StimError("PAULI_CHANNEL_1", 0.01, 0.02, 0.03), [qc.qubits[0]])
    
    # HERALDED_ERASE (1 parameter)
    qc.append(StimError("HERALDED_ERASE", 0.05), [qc.qubits[0]])

    # HERALDED_PAULI_CHANNEL_1 (4 parameters)
    qc.append(StimError("HERALDED_PAULI_CHANNEL_1", 0.01, 0.02, 0.03, 0.04), [qc.qubits[0]])

    stim_circuit = qc.to_stim()
    stim_str = str(stim_circuit)
    
    assert "DEPOLARIZE1(0.1) 0" in stim_str
    assert "X_ERROR(0.2) 0" in stim_str
    assert "Y_ERROR(0.3) 0" in stim_str
    assert "Z_ERROR(0.4) 0" in stim_str
    assert "PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0" in stim_str
    assert "HERALDED_ERASE(0.05) 0" in stim_str
    assert "HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0" in stim_str
    
    # Test 2-qubit errors
    qc = QuantumCircuit(2)
    
    # 1 parameter
    qc.append(StimError("DEPOLARIZE2", 0.05), qc.qubits)
    
    # 15 parameters for PAULI_CHANNEL_2
    # Verify that we can pass a large number of parameters
    params = [0.001 * i for i in range(1, 16)]
    qc.append(StimError("PAULI_CHANNEL_2", *params), qc.qubits)
    
    stim_circuit = qc.to_stim()
    stim_str = str(stim_circuit)
    
    assert "DEPOLARIZE2(0.05) 0 1" in stim_str
    
    # Check PAULI_CHANNEL_2 output format
    assert "PAULI_CHANNEL_2" in stim_str
    # Check a few parameters are present in the string
    assert "0.001" in stim_str
    assert "0.015" in stim_str

    # Test CORRELATED_ERROR (E)
    qc = QuantumCircuit(3)
    # E(0.1) X0 Y1 (Z2 skipped or implicit I) - User notation E_XYI
    qc.append(StimError("E", 0.1, pauli_string="XYI"), qc.qubits)
    
    # ELSE_CORRELATED_ERROR(0.2) Z0 Z1 Z2
    qc.append(StimError("ELSE_CORRELATED_ERROR", 0.2, pauli_string="ZZZ"), qc.qubits)

    stim_circuit = qc.to_stim()
    stim_str = str(stim_circuit)

    # Note: stim string representation formats targets like X0, Y1
    # Check for presence of E operations
    # Depending on stim version, E might be aliased to CORRELATED_ERROR in output or vice versa
    assert "E(0.1)" in stim_str or "CORRELATED_ERROR(0.1)" in stim_str
    assert "ELSE_CORRELATED_ERROR(0.2)" in stim_str
    
    # Check targets (format might comprise X0, Y1 etc)
    # The exact string representation depends on qubit indices
    # We applied to 0, 1, 2
    # So X0 Y1
    assert "X0" in stim_str
    assert "Y1" in stim_str
    # Z2 for ELSE_CORRELATED_ERROR
    assert "Z2" in stim_str
