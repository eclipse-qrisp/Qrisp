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
    """Test conversion of Stim noise channels via StimNoiseGate."""
    from qrisp.misc.stim_tools import StimNoiseGate
    
    # Test 1-qubit errors
    qc = QuantumCircuit(1)
    
    # 1 parameter errors
    qc.append(StimNoiseGate("DEPOLARIZE1", 0.1), [qc.qubits[0]])
    qc.append(StimNoiseGate("X_ERROR", 0.2), [qc.qubits[0]])
    qc.append(StimNoiseGate("Y_ERROR", 0.3), [qc.qubits[0]])
    qc.append(StimNoiseGate("Z_ERROR", 0.4), [qc.qubits[0]])
    
    # 3 parameters (PAULI_CHANNEL_1)
    qc.append(StimNoiseGate("PAULI_CHANNEL_1", 0.01, 0.02, 0.03), [qc.qubits[0]])
    
    # HERALDED_ERASE (1 parameter)
    qc.append(StimNoiseGate("HERALDED_ERASE", 0.05), [qc.qubits[0]])

    # HERALDED_PAULI_CHANNEL_1 (4 parameters)
    qc.append(StimNoiseGate("HERALDED_PAULI_CHANNEL_1", 0.01, 0.02, 0.03, 0.04), [qc.qubits[0]])

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
    qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), qc.qubits)
    
    # 15 parameters for PAULI_CHANNEL_2
    # Verify that we can pass a large number of parameters
    params = [0.001 * i for i in range(1, 16)]
    qc.append(StimNoiseGate("PAULI_CHANNEL_2", *params), qc.qubits)
    
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
    qc.append(StimNoiseGate("E", 0.1, pauli_string="XYI"), qc.qubits)
    
    # ELSE_CORRELATED_ERROR(0.2) Z0 Z1 Z2
    qc.append(StimNoiseGate("ELSE_CORRELATED_ERROR", 0.2, pauli_string="ZZZ"), qc.qubits)

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


def test_detector_operation_conversion():
    """
    Test manual creation of ParityOperation and conversion to Stim using qc.to_stim().
    
    With the new parity design, ParityOperation takes only n input clbits (no output clbit).
    The parity operation is tracked in the circuit and converted directly to Stim detectors.
    """
    from qrisp.jasp.primitives.parity_primitive import ParityOperation
    
    qc = QuantumCircuit(2, 2) # 2 qubits, 2 clbits (for measurements only)
    
    # Manual circuit construction simulating what JASP would do
    # measure qubit 0 into clbit 0
    qc.measure(0, 0)
    # measure qubit 1 into clbit 1
    qc.measure(1, 1)
    
    # ParityOperation now takes only input clbits (no result clbit)
    # expectation=0 means we expect even parity
    parity_op = ParityOperation(2, expectation=0)
    qc.append(parity_op, clbits=[qc.clbits[0], qc.clbits[1]])
    
    stim_circuit, meas_map, det_map = qc.to_stim(return_measurement_map=True, return_detector_map=True)
    
    # Check instructions
    lines = str(stim_circuit).splitlines()
    assert "M 0 1" in lines or ("M 0" in lines and "M 1" in lines)

    # We expect DETECTOR rec[-2] rec[-1] or rec[-1] rec[-2]
    det_lines = [l for l in lines if "DETECTOR" in l]
    assert len(det_lines) == 1
    assert "rec[-1]" in det_lines[0]
    assert "rec[-2]" in det_lines[0] # JASP Parity checks all inputs
    
    # Check maps
    assert len(meas_map) == 2
    assert len(det_map) == 1


def test_detector_permutation():
    import stim
    import numpy as np
    from qrisp.misc.stim_tools.detector_permutation import permute_detectors

    # Test 1: Validation
    print("Test 1: Validation...", end="")
    c = stim.Circuit("M 0\nDETECTOR(0) rec[-1]")
    try:
        permute_detectors(c, (1, 0)) # Too many indices
        raise Exception("Failed to catch length mismatch")
    except ValueError:
        pass
        
    try:
        permute_detectors(c, (0, 0)) 
        raise Exception("Failed to catch invalid permutation (duplicates)")
    except ValueError:
        pass
    print("Passed.")

    # Test 2: Sampler Consistency
    print("Test 2: Sampler Consistency...", end="")
    
    # Create a circuit where:
    # Error on q0 -> Triggers D0
    # Error on q1 -> Triggers D1
    # Error on q2 -> Triggers D2
    c = stim.Circuit("""
        X_ERROR(1) 0 1 2
        M 0 1 2
        DETECTOR(0) rec[-3]
        DETECTOR(1) rec[-2]
        DETECTOR(2) rec[-1]
    """)
    
    perm = (1, 2, 0)
    c_perm = permute_detectors(c, perm)
    
    sampler_orig = c.compile_detector_sampler()
    sampler_perm = c_perm.compile_detector_sampler()
    
    # Generate many samples to ensure we see errors
    shots = 1000
    samples_orig = sampler_orig.sample(shots, bit_packed=False)
    samples_perm = sampler_perm.sample(shots, bit_packed=False)
    
    c_det = stim.Circuit("""
        X_ERROR(1) 0 2
        M 0 1 2
        DETECTOR(0) rec[-3]
        DETECTOR(1) rec[-2]
        DETECTOR(2) rec[-1]
    """)
    # D0 checks M0 (X 0 -> 1) -> D0 triggers
    # D1 checks M1 (Identity -> 0) -> D1 off
    # D2 checks M2 (X 2 -> 1) -> D2 triggers
    # Result: [1, 0, 1]
    
    c_det_perm = permute_detectors(c_det, perm)
    # Permutation (1, 2, 0)
    # New D0 = Old D1 = 0
    # New D1 = Old D2 = 1
    # New D2 = Old D0 = 1
    # Expected: [0, 1, 1]
    
    s_orig = c_det.compile_detector_sampler().sample(1)[0]
    s_perm = c_det_perm.compile_detector_sampler().sample(1)[0]
    
    assert np.array_equal(s_orig, [True, False, True]), f"Setup failed: {s_orig}"
    
    reconstructed_perm = []
    for i in range(len(perm)):
        reconstructed_perm.append(s_orig[perm[i]])
    reconstructed_perm = np.array(reconstructed_perm)
    
    if not np.array_equal(s_perm, reconstructed_perm):
        raise Exception(f"Deterministic sampler mismatch.\nOrig: {s_orig}\nPerm: {s_perm}\nExpP: {reconstructed_perm}")
    print("Passed.")

    # Test 3: Coordinate Shift and Semantic Preservation
    print("Test 3: Coordinates...", end="")
    c_coords = stim.Circuit("""
        SHIFT_COORDS(10)
        M 0
        DETECTOR(1) rec[-1]
        SHIFT_COORDS(20)
        M 1
        DETECTOR(2) rec[-1]
    """)
    # D0: Coords(1)+Shift(10) = 11. Targets M0.
    # D1: Coords(2)+Shift(10+20) = 32. Targets M1.
    
    # Permute (1, 0)
    # New D0 = Old D1 (Abs 32, Targets M1)
    # New D1 = Old D0 (Abs 11, Targets M0)
    
    c_coords_perm = permute_detectors(c_coords, (1, 0))
    
    dets = [inst for inst in c_coords_perm if inst.name == "DETECTOR"]
    d0_args = dets[0].gate_args_copy()
    d1_args = dets[1].gate_args_copy()
    
    if list(d0_args) != [2.0]:
         raise Exception(f"D0 Coordinate mismatch. Expected [2.0], got {d0_args}")
    if list(d1_args) != [-19.0]:
         raise Exception(f"D1 Coordinate mismatch. Expected [-19.0], got {d1_args}")
         
    print("Passed.")

    # Test 4: Repeat Safety
    print("Test 4: Repeat Safety...", end="")
    c_repeat = stim.Circuit("""
        REPEAT 10 {
            M 0
            DETECTOR(0) rec[-1]
        }
    """)
    try:
        permute_detectors(c_repeat, list(range(10))) 
        raise Exception("Failed to catch REPEAT block")
    except ValueError as e:
        assert "REPEAT" in str(e)
    print("Passed.")

    # Test 5: Observables Interaction
    print("Test 5: Observables Interaction...", end="")
    # Check that Observables are preserved and don't interfere
    c_obs = stim.Circuit("""
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        DETECTOR(0) rec[-1]
        M 1
        OBSERVABLE_INCLUDE(1) rec[-1]
        DETECTOR(1) rec[-1]
    """)
    # Original: 
    # D0 checks M0. D1 checks M1.
    # L0 checks M0. L1 checks M1.
    
    # Observables should stay in 'other_instructions' and not be moved.
    
    c_obs_perm = permute_detectors(c_obs, (1, 0))
    
    # We expect:
    # M 0
    # Obs(0) rec[-1]
    # M 1
    # Obs(1) rec[-1]
    # DETECTOR(1) ... (was D1, now D0) -> targets M1 (rec[-1] relative to end)
    # DETECTOR(0) ... (was D0, now D1) -> targets M0 (rec[-2] relative to end)
    
    str_circ = str(c_obs_perm)
    print("Circuit:\n", str_circ)
    
    # Verify DETECTORS moved to end
    assert str_circ.strip().endswith("DETECTOR(0) rec[-2]"), f"Circuit did not end with expected detector string. output: {str_circ.strip()[-20:]}"
    
    # Verify OBSERVABLES stayed put (interleaved)
    lines = str_circ.splitlines()
    m0_idx = -1
    obs0_idx = -1
    m1_idx = -1
    obs1_idx = -1
    
    for i, line in enumerate(lines):
        if "M 0" in line: m0_idx = i
        if "OBSERVABLE_INCLUDE(0)" in line: obs0_idx = i
        if "M 1" in line: m1_idx = i
        if "OBSERVABLE_INCLUDE(1)" in line: obs1_idx = i
        
    assert m0_idx < obs0_idx < m1_idx < obs1_idx, f"Observables were moved or reordered improperly. Indices: M0={m0_idx}, Obs0={obs0_idx}, M1={m1_idx}, Obs1={obs1_idx}"
    
    print("Passed.")

    print("All tests passed.")


def test_observable_map():
    from qrisp import QuantumCircuit
    from qrisp.jasp.primitives.parity_primitive import ParityOperation

    qc = QuantumCircuit(2, 2)  # 2 qubits, 2 clbits (measurements only)
    qc.h(0)
    qc.cx(0, 1)
    
    # Explicitly measure into the clbits
    qc.measure(qc.qubits[0], qc.clbits[0])
    qc.measure(qc.qubits[1], qc.clbits[1])
    
    # Define parity/observable operation - now takes only input clbits
    parity_op = ParityOperation(2, observable=True)
    qc.append(parity_op, [], [qc.clbits[0], qc.clbits[1]])
    
    res = qc.to_stim(return_observable_map=True)
    stim_circuit = res[0]
    observable_map = res[1]
    
    # Check that the map contains the parity handle
    assert len(observable_map) == 1
    
    # Ensure detector/observable logic is correct
    res_det = qc.to_stim(return_detector_map=True, return_observable_map=True)
    det_map = res_det[1]
    obs_map = res_det[2]
    
    # Detector map should be empty (observable=True means observable, not detector)
    assert len(det_map) == 0
    assert len(obs_map) == 1


def test_qc_parity_method_basic():
    """Test the QuantumCircuit.parity method with basic usage."""
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import ParityHandle
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    # Test basic parity method call
    handle = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0)
    
    # Verify it returns a ParityHandle
    assert isinstance(handle, ParityHandle)
    
    # Verify ParityHandle properties
    assert handle.clbits == [qc.clbits[0], qc.clbits[1]]
    assert handle.expectation == 0
    assert handle.observable == False


def test_qc_parity_method_single_clbit():
    """Test the QuantumCircuit.parity method with a single clbit."""
    from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import ParityHandle
    
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    
    # Test with single clbit (not in a list)
    handle = qc.parity(qc.clbits[0], expectation=1)
    
    assert isinstance(handle, ParityHandle)
    assert handle.clbits == [qc.clbits[0]]
    assert handle.expectation == 1


def test_qc_parity_method_detector_map():
    """Test that parity method handles integrate correctly with to_stim detector_map."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    
    # Create parity detectors
    handle1 = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0)
    handle2 = qc.parity([qc.clbits[1], qc.clbits[2]], expectation=0)
    
    # Convert to Stim and get detector map
    stim_circuit, meas_map, det_map = qc.to_stim(
        return_measurement_map=True,
        return_detector_map=True
    )
    
    # Verify both handles are in the detector map
    assert handle1 in det_map
    assert handle2 in det_map
    
    # Verify detector indices are assigned
    assert det_map[handle1] == 0
    assert det_map[handle2] == 1
    
    # Verify DETECTOR instructions in Stim circuit
    stim_str = str(stim_circuit)
    assert stim_str.count("DETECTOR") == 2


def test_qc_parity_method_observable_map():
    """Test that parity method with observable=True integrates with observable_map."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    # Create observable parity
    handle = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0, observable=True)
    
    # Verify observable property
    assert handle.observable == True
    
    # Convert to Stim and get observable map
    stim_circuit, meas_map, det_map, obs_map = qc.to_stim(
        return_measurement_map=True,
        return_detector_map=True,
        return_observable_map=True
    )
    
    # Verify handle is in observable map, not detector map
    assert handle not in det_map
    assert handle in obs_map
    assert obs_map[handle] == 0
    
    # Verify OBSERVABLE_INCLUDE instruction in Stim circuit
    stim_str = str(stim_circuit)
    assert "OBSERVABLE_INCLUDE" in stim_str
    assert "DETECTOR" not in stim_str or stim_str.count("DETECTOR") == 0


def test_qc_parity_method_multiple_detectors():
    """Test multiple parity operations and their handles."""
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.measure(range(4), range(4))
    
    # Create multiple parity checks
    handles = []
    handles.append(qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0))
    handles.append(qc.parity([qc.clbits[0], qc.clbits[2]], expectation=0))
    handles.append(qc.parity([qc.clbits[0], qc.clbits[3]], expectation=0))
    
    # Convert and verify
    stim_circuit, meas_map, det_map = qc.to_stim(
        return_measurement_map=True,
        return_detector_map=True
    )
    
    # All handles should be in map with unique indices
    assert len(det_map) == 3
    for i, handle in enumerate(handles):
        assert handle in det_map
        assert det_map[handle] == i


def test_qc_parity_method_mixed_detectors_observables():
    """Test mixing detector and observable parity operations."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure(range(3), range(3))
    
    # Create mixed parity checks
    det_handle = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0, observable=False)
    obs_handle = qc.parity([qc.clbits[0], qc.clbits[2]], expectation=1, observable=True)
    
    # Convert
    stim_circuit, meas_map, det_map, obs_map = qc.to_stim(
        return_measurement_map=True,
        return_detector_map=True,
        return_observable_map=True
    )
    
    # Verify separation
    assert det_handle in det_map
    assert det_handle not in obs_map
    assert obs_handle in obs_map
    assert obs_handle not in det_map
    
    # Verify properties
    assert det_handle.observable == False
    assert obs_handle.observable == True
    assert obs_handle.expectation == 1


def test_qc_parity_method_handle_equality():
    """Test that ParityHandle equality works correctly (content-based)."""
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    
    # Create two parity operations with same parameters
    handle1 = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0)
    handle2 = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=0)
    
    # ParityHandle uses content-based equality (same clbits, expectation, observable)
    # This is by design to work across transpile calls
    assert handle1 == handle2
    
    # Handle with different expectation should not be equal
    handle3 = qc.parity([qc.clbits[0], qc.clbits[1]], expectation=1)
    assert handle1 != handle3
    
    # Both handles with same parameters map to the same detector
    # (only the last one will appear in the map due to equality)
    stim_circuit, meas_map, det_map = qc.to_stim(
        return_measurement_map=True,
        return_detector_map=True
    )
    
    # Since handle1 == handle2, they should map to the same detector
    assert handle1 in det_map
    assert handle2 in det_map
    assert det_map[handle1] == det_map[handle2]
    
    # handle3 has different expectation, so different detector
    assert handle3 in det_map
    assert det_map[handle3] != det_map[handle1]

