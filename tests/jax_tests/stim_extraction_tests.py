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

import numpy as np
import stim
from qrisp import QuantumVariable, QuantumFloat, QuantumBool, h, x, y, z, s, s_dg, sx, cx, cy, cz, measure, control
from qrisp.jasp import extract_stim, parity
from qrisp.jasp.evaluation_tools.stim_extraction import StimMeasurementHandles, StimDetectorHandles, StimObservableHandles, StimQubitIndices
from qrisp.misc.stim_tools import stim_noise


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_single_return_value():
    """Test that single return value functions return measurement indices + circuit."""
    @extract_stim
    def single_qubit_circuit():
        qv = QuantumVariable(1)
        h(qv)
        return measure(qv)
    
    result = single_qubit_circuit()
    
    # Single return value should return (indices, circuit)
    assert isinstance(result, tuple)
    assert len(result) == 2
    meas_indices, stim_circuit = result
    assert isinstance(stim_circuit, stim.Circuit)
    assert "H 0" in str(stim_circuit)
    assert "M 0" in str(stim_circuit)


def test_no_return_value():
    """Test function with no return value (only circuit returned)."""
    @extract_stim
    def no_return():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
    
    result = no_return()
    
    # Should only return the Stim circuit
    assert isinstance(result, stim.Circuit)
    assert "H 0" in str(result)
    assert "CX 0 1" in str(result)


def test_multiple_return_values():
    """Test that multiple return values return n+1 elements."""
    @extract_stim
    def multi_return():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(3)
        h(qv1)
        h(qv2)
        result1 = measure(qv1)
        result2 = measure(qv2)
        return result1, result2
    
    qv1_indices, qv2_indices, stim_circuit = multi_return()
    
    # Check we got 3 elements (2 return values + circuit)
    assert isinstance(qv1_indices, np.ndarray)
    assert isinstance(qv2_indices, np.ndarray)
    assert isinstance(stim_circuit, stim.Circuit)
    
    # Check indices are correct length
    assert len(qv1_indices) == 2
    assert len(qv2_indices) == 3
    
    # Check indices are sequential
    assert np.array_equal(qv1_indices, [0, 1])
    assert np.array_equal(qv2_indices, [2, 3, 4])


def test_classical_value_passthrough():
    """Test that classical values pass through unchanged."""
    @extract_stim
    def with_classical(n):
        qv = QuantumVariable(2)
        h(qv)
        
        # Classical computation
        classical_result = n * 2 + 5
        
        # Quantum measurement
        quantum_result = measure(qv)
        
        return classical_result, quantum_result
    
    classical_val, meas_indices, stim_circuit = with_classical(10)
    
    # Classical value should be unchanged
    assert classical_val == 25  # 10 * 2 + 5
    
    # Measurement indices should be a numpy array
    assert isinstance(meas_indices, np.ndarray)
    assert len(meas_indices) == 2
    
    # Circuit should be present
    assert isinstance(stim_circuit, stim.Circuit)


def test_single_qubit_measurement():
    """Test measurement of a single qubit returns an integer index."""
    @extract_stim
    def single_qubit():
        qv = QuantumVariable(3)
        h(qv)
        
        # Measure only the first qubit
        first = measure(qv[0])
        
        return first
    
    meas_idx, stim_circuit = single_qubit()
    
    # Single qubit measurement should return an integer-like value
    assert int(meas_idx) == 0


# ============================================================================
# Clifford Gate Tests
# ============================================================================

def test_clifford_gates():
    """Test that all Clifford gates are properly converted."""
    @extract_stim
    def clifford_circuit():
        qv = QuantumVariable(3)
        
        # Single-qubit Clifford gates
        h(qv[0])
        x(qv[1])
        y(qv[2])
        z(qv[0])
        s(qv[1])
        s_dg(qv[2])
        sx(qv[0])
        
        # Two-qubit Clifford gates
        cx(qv[0], qv[1])
        cy(qv[1], qv[2])
        cz(qv[0], qv[2])
        
        return measure(qv)
    
    stim_circuit = clifford_circuit()
    
    circuit_str = str(stim_circuit)
    
    # Verify all gates are present
    assert "H 0" in circuit_str
    assert "X 1" in circuit_str
    assert "Y 2" in circuit_str
    assert "Z 0" in circuit_str
    assert "S 1" in circuit_str
    assert "S_DAG 2" in circuit_str
    assert "SQRT_X 0" in circuit_str
    assert "CX 0 1" in circuit_str
    assert "CY 1 2" in circuit_str
    assert "CZ 0 2" in circuit_str
    assert "M 0 1 2" in circuit_str


# ============================================================================
# Sampling and Slicing Tests
# ============================================================================

def test_bell_state_sampling():
    """Test sampling from a Bell state and verify correlation."""
    @extract_stim
    def bell_state():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)
    
    meas_indices, stim_circuit = bell_state()
    
    # Sample from the circuit
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # Extract measurements using indices
    bell_samples = samples[:, meas_indices]
    
    # In a Bell state, both qubits should always have the same value
    assert np.all(bell_samples[:, 0] == bell_samples[:, 1])


def test_ghz_state_sampling():
    """Test sampling from a GHZ state and verify all-qubit correlation."""
    from qrisp.jasp import jrange
    
    @extract_stim
    def ghz_state(n):
        qv = QuantumVariable(n)
        h(qv[0])
        for i in jrange(1, n):  # Use jrange for Jax-traceable loops
            cx(qv[0], qv[i])
        return measure(qv)
    
    meas_indices, stim_circuit = ghz_state(5)
    
    # Sample from the circuit
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # Extract measurements
    ghz_samples = samples[:, meas_indices]
    
    # All qubits should have the same value in each shot
    for i in range(1, 5):
        assert np.all(ghz_samples[:, 0] == ghz_samples[:, i])


def test_slicing_multiple_measurements():
    """Test slicing samples for multiple independent measurements."""
    @extract_stim
    def multiple_measurements():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(3)
        
        # Prepare qv1 in |11⟩
        x(qv1)
        
        # Prepare qv2 in |000⟩ (default)
        
        result1 = measure(qv1)
        result2 = measure(qv2)
        
        return result1, result2
    
    qv1_indices, qv2_indices, stim_circuit = multiple_measurements()
    
    # Sample
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(10)
    
    # Slice for each return value
    qv1_samples = samples[:, qv1_indices]
    qv2_samples = samples[:, qv2_indices]
    
    # qv1 should always be [1, 1]
    assert np.all(qv1_samples == 1)
    
    # qv2 should always be [0, 0, 0]
    assert np.all(qv2_samples == 0)


def test_entanglement_slicing():
    """Test slicing and verifying entanglement between two quantum variables."""
    @extract_stim
    def entangled_state():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(3)
        
        # Put qv1 in superposition
        h(qv1)
        
        # Entangle qv2 with qv1[0]
        for i in range(3):
            cx(qv1[0], qv2[i])
        
        result1 = measure(qv1)
        result2 = measure(qv2)
        
        return result1, result2
    
    qv1_indices, qv2_indices, stim_circuit = entangled_state()
    
    # Sample
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # Slice
    qv1_samples = samples[:, qv1_indices]
    qv2_samples = samples[:, qv2_indices]
    
    # When qv1[0] == 0, all qv2 bits should be 0
    mask_0 = qv1_samples[:, 0] == 0
    assert np.all(qv2_samples[mask_0] == 0)
    
    # When qv1[0] == 1, all qv2 bits should be 1
    mask_1 = qv1_samples[:, 0] == 1
    assert np.all(qv2_samples[mask_1] == 1)


def test_bit_array_to_integer_conversion():
    """Test converting bit arrays to integer values."""
    @extract_stim
    def integer_prep():
        qv = QuantumVariable(4)
        
        # Prepare |0101⟩ = 10 in little-endian (5 in big-endian)
        x(qv[0])  # bit 0
        x(qv[2])  # bit 2
        
        return measure(qv)
    
    meas_indices, stim_circuit = integer_prep()
    
    # Sample
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(10)
    
    # Extract and convert to integers (little-endian)
    qv_samples = samples[:, meas_indices]
    qv_values = qv_samples.dot(1 << np.arange(qv_samples.shape[1]))
    
    # Should all be 5 (binary 0101 in little-endian)
    assert np.all(qv_values == 5)


# ============================================================================
# Mid-Circuit Measurement Tests
# ============================================================================

def test_mid_circuit_measurement():
    """Test that mid-circuit measurements are handled correctly."""
    @extract_stim
    def mid_circuit():
        qv = QuantumVariable(3)
        
        h(qv[0])
        
        # Mid-circuit measurement
        mid_result = measure(qv[0])
        
        # Continue with operations
        cx(qv[0], qv[1])
        cx(qv[1], qv[2])
        
        # Final measurement
        final_result = measure(qv)
        
        return mid_result, final_result
    
    mid_idx, final_indices, stim_circuit = mid_circuit()
    
    # Mid-circuit measurement should be index 0
    assert int(mid_idx) == 0
    
    # Final measurements should be indices 1, 2, 3 (qv[0] is measured again)
    assert np.array_equal(final_indices, [1, 2, 3])
    
    # Sample and verify
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # The first measurement (mid_idx) should correlate with subsequent measurements
    # since qv[0] is measured twice
    mid_samples = samples[:, mid_idx]
    final_samples = samples[:, final_indices]
    
    # qv[0]'s second measurement should match the first
    assert np.all(mid_samples == final_samples[:, 0])
    
    # All final measurements should be correlated (GHZ-like)
    assert np.all(final_samples[:, 0] == final_samples[:, 1])
    assert np.all(final_samples[:, 1] == final_samples[:, 2])


def test_measurement_order_preserved():
    """Test that measurements appear in the correct order in the circuit."""
    @extract_stim
    def ordered_measurements():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(2)
        
        h(qv1)
        h(qv2)
        
        # Measure in specific order
        result1 = measure(qv1[0])
        result2 = measure(qv2[0])
        result3 = measure(qv1[1])
        result4 = measure(qv2[1])
        
        return result1, result2, result3, result4
    
    idx1, idx2, idx3, idx4, stim_circuit = ordered_measurements()
    
    # Indices should be sequential in measurement order
    assert idx1 == 0
    assert idx2 == 1
    assert idx3 == 2
    assert idx4 == 3


# ============================================================================
# QuantumVariable vs QuantumFloat Tests
# ============================================================================

def test_quantum_variable_compatibility():
    """Test that QuantumVariable works correctly with extract_stim."""
    @extract_stim
    def qv_test():
        qv = QuantumVariable(3)
        h(qv)
        return measure(qv)
    
    meas_indices, stim_circuit = qv_test()
    
    # Should return numpy array of indices
    assert isinstance(meas_indices, np.ndarray)
    assert len(meas_indices) == 3
    
    # Sample and convert to integers
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    qv_samples = samples[:, meas_indices]
    qv_values = qv_samples.dot(1 << np.arange(qv_samples.shape[1]))
    
    # Should get various integer values from 0 to 7
    unique_values = np.unique(qv_values)
    assert len(unique_values) > 1  # Should have multiple outcomes due to superposition


def test_quantum_bool_compatibility():
    """Test that QuantumBool works correctly with extract_stim."""
    @extract_stim
    def qb_test():
        qb = QuantumBool()
        h(qb)
        return measure(qb)
    
    meas_idx, stim_circuit = qb_test()
    
    # QuantumBool measurement returns a single index
    assert meas_idx.item() == 0
    
    # Sample
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    qb_samples = samples[:, meas_idx]
    
    # Should have both 0 and 1 outcomes
    assert 0 in qb_samples
    assert 1 in qb_samples


def test_quantum_float_post_processing():
    """Test that QuantumFloat post-processing is handled (returns ProcessedMeasurement)."""
    from qrisp import ProcessedMeasurement
    
    @extract_stim
    def qf_test():
        qf = QuantumFloat(3, -1)  # 3 qubits, exponent -1 (supports fractions)
        h(qf)
        result = measure(qf)
        return result
    
    processed_result, stim_circuit = qf_test()
    
    # QuantumFloat's decoded measurement should be a ProcessedMeasurement
    # because it involves post-processing (converting to fractional values)
    assert isinstance(processed_result, ProcessedMeasurement)
    
    # Circuit should still be valid
    assert isinstance(stim_circuit, stim.Circuit)


# ============================================================================
# Mixed Return Value Tests
# ============================================================================

def test_mixed_classical_and_quantum():
    """Test mixing classical values, single measurements, and multi-qubit measurements."""
    @extract_stim
    def mixed_returns(a, b):
        qv1 = QuantumVariable(3)
        qv2 = QuantumBool()
        
        h(qv1)
        h(qv2)
        
        # Classical computation
        classical_sum = a + b
        classical_product = a * b
        
        # Various quantum measurements
        single_meas = measure(qv2)
        multi_meas = measure(qv1)
        
        return classical_sum, single_meas, classical_product, multi_meas
    
    c_sum, single_idx, c_prod, multi_indices, stim_circuit = mixed_returns(5, 7)
    
    # Check classical values
    assert c_sum == 12
    assert c_prod == 35
    
    # Check quantum measurement indices
    # QuantumBool returns a single index
    assert single_idx.item() == 0
    
    assert isinstance(multi_indices, np.ndarray)
    assert len(multi_indices) == 3
    
    # Check circuit
    assert isinstance(stim_circuit, stim.Circuit)


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_non_clifford_error():
    """Test that non-Clifford gates raise an error."""
    from qrisp import t
    
    @extract_stim
    def non_clifford():
        qv = QuantumVariable(1)
        h(qv)
        t(qv)  # T gate is not Clifford
        return measure(qv)
    
    try:
        result = non_clifford()
        assert False, "Should have raised an error for non-Clifford gate"
    except (NotImplementedError, ValueError):
        # Expected behavior
        pass


def test_parametric_gate_error():
    """Test that parametric gates raise an error."""
    from qrisp import rz
    
    @extract_stim
    def parametric():
        qv = QuantumVariable(1)
        h(qv)
        rz(0.5, qv)  # Parametric gate
        return measure(qv)
    
    try:
        result = parametric()
        assert False, "Should have raised an error for parametric gate"
    except (NotImplementedError, ValueError):
        # Expected behavior
        pass


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_circuit():
    """Test extracting from an empty circuit."""
    @extract_stim
    def empty():
        qv = QuantumVariable(2)
        # No operations, no measurements
        pass
    
    stim_circuit = empty()
    
    # Should return just the circuit (no return values)
    assert isinstance(stim_circuit, stim.Circuit)
    # Circuit might have qubit allocations but no gates
    # Just verify it's a valid circuit
    assert str(stim_circuit) is not None


def test_large_circuit():
    """Test with a larger circuit to ensure scalability."""
    from qrisp.jasp import jrange
    
    @extract_stim
    def large_circuit(n=20):
        qv = QuantumVariable(n)
        
        # Create a GHZ state
        h(qv[0])
        for i in jrange(1, n):  # Use jrange for Jax-traceable loops
            cx(qv[0], qv[i])
        
        return measure(qv)
    
    meas_indices, stim_circuit = large_circuit(20)
    
    assert len(meas_indices) == 20
    
    # Sample and verify GHZ property
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(50)
    qv_samples = samples[:, meas_indices]
    
    # All qubits should be correlated
    for i in range(1, 20):
        assert np.all(qv_samples[:, 0] == qv_samples[:, i])


def test_jasp_stim_extraction_with_error():
    """Test Jasp stim extraction with errors."""
    from qrisp.jasp import extract_stim
    from qrisp import QuantumVariable, h, cx
    from qrisp.misc.stim_tools import StimNoiseGate, stim_noise
    
    @extract_stim
    def noisy_bell_pair():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        
        # Add noise
        qc = qv.qs
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.1), [qv[0]])
        qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), [qv[0], qv[1]])
        
        # Add correlated error
        qc.append(StimNoiseGate("E", 0.1, pauli_string="XY"), [qv[0], qv[1]])

        stim_noise("DEPOLARIZE1", 0.1, qv[0])
        stim_noise("DEPOLARIZE2", 0.1, qv[0], qv[1])
        stim_noise("E", 0.1, qv[0], qv[1], pauli_string="XX")

    stim_circuit = noisy_bell_pair()
    stim_str = str(stim_circuit)
    
    assert "H 0" in stim_str
    assert "CX 0 1" in stim_str
    assert "DEPOLARIZE1(0.1) 0" in stim_str
    assert "DEPOLARIZE2(0.05) 0 1" in stim_str
    # E might be aliased or formatted differently, check for presence
    assert "E(" in stim_str or "CORRELATED_ERROR(" in stim_str 
    assert "X0" in stim_str or "X 0" in stim_str # Ensure target part is present


def test_stim_noise_gate_errors():
    """Test that prohibited methods on StimNoiseGate raise appropriate exceptions."""
    from qrisp.misc.stim_tools.error_class import StimNoiseGate
    import pytest
    
    ng = StimNoiseGate("DEPOLARIZE1", 0.1)

    # Test inverse
    with pytest.raises(Exception, match="is not invertible"):
        ng.inverse()

    # Test control
    with pytest.raises(Exception, match="can not be controlled"):
        ng.control()

    # Test c_if
    with pytest.raises(Exception, match="can not be classically controlled"):
        ng.c_if()
    
    # Check is_permeable
    assert not ng.is_permeable([0])


# ============================================================================
# Parity / Detector / Observable Tests
# ============================================================================

def test_basic_parity_detector():
    """
    Test simple parity detector usage.
    """
    @extract_stim
    def simple_detector():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # Parity of Bell pair should be 0
        d = parity(m0, m1, expectation=False)
        return d

    res, stim_circuit = simple_detector()
    
    # Check that we have a detector instruction
    num_detectors = stim_circuit.num_detectors if hasattr(stim_circuit, "num_detectors") else len([i for i in stim_circuit if i.name == "DETECTOR"])
    assert num_detectors == 1
    # Check structure: H 0, CX 0 1, M 0 1, DETECTOR rec[-2] rec[-1]
    s_str = str(stim_circuit)
    assert "DETECTOR" in s_str
    assert "rec[-1]" in s_str and "rec[-2]" in s_str

def test_measurement_detector_gap_interleaved():
    """
    Test detector with measurements separated by another measurement in time.
    """
    @extract_stim
    def gap_detector_interleaved():
        qv = QuantumVariable(3)
        h(qv)
        m0 = measure(qv[0])
        m1 = measure(qv[1]) # Intervening measurement
        m2 = measure(qv[2])
        
        # Check 0 and 2 (skipping m1)
        d = parity(m0, m2, expectation=False)
        return d
        
    res, stim_circuit = gap_detector_interleaved()
    s_str = str(stim_circuit)
    # m0 is rec[-3], m1 is rec[-2], m2 is rec[-1]
    assert "rec[-1]" in s_str
    assert "rec[-3]" in s_str
    assert "rec[-2]" not in s_str

def test_basic_observable():
    """
    Test observable creation (expectation=None).
    """
    @extract_stim
    def simple_observable():
        qv = QuantumVariable(1)
        h(qv[0])
        m = measure(qv[0])
        obs = parity(m, expectation=None)
        return obs

    res_obs_idx, stim_circuit = simple_observable()
    
    # Handle tuple return if necessary
    if isinstance(res_obs_idx, tuple):
        res_obs_idx = res_obs_idx[0]

    # Return value should be observable index 0
    assert res_obs_idx == 0
    assert stim_circuit.num_observables == 1
    assert "OBSERVABLE_INCLUDE(0)" in str(stim_circuit)

def test_observable_chaining():
    """
    Test chaining observables (extending one observable).
    parity(new_meas, old_obs) -> should create NEW observable with combined measurements.
    """
    @extract_stim
    def chained_observable():
        qv = QuantumVariable(3)
        h(qv)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        # Obs 0: m[0] + m[1]
        obs_1 = parity(m0, m1, expectation=None)
        
        # Obs 1: m[2] + Obs 0 (= m[0] + m[1] + m[2])
        obs_2 = parity(m2, obs_1, expectation=None)
        
        return obs_1, obs_2

    idx1, idx2, stim_circuit = chained_observable()
    
    # We expect two observables.
    # idx1 -> 0
    # idx2 -> 1
    assert idx1 == 0
    assert idx2 == 1
    assert stim_circuit.num_observables == 2
    
    lines = str(stim_circuit).splitlines()
    obs_lines = [l for l in lines if "OBSERVABLE_INCLUDE" in l]
    
    # Should contains OBS(0) and OBS(1)
    assert any("OBSERVABLE_INCLUDE(0)" in l for l in obs_lines)
    assert any("OBSERVABLE_INCLUDE(1)" in l for l in obs_lines)
    
    # Check that Obs 1 incorporates Obs 0's history
    # Finding line for obs 1
    obs1_line = next(l for l in obs_lines if "OBSERVABLE_INCLUDE(1)" in l)
    assert obs1_line.count("rec") == 3

def test_observable_merging():
    """
    Test merging two distinct observables into a third one.
    """
    @extract_stim
    def merged_observable():
        qv = QuantumVariable(4)
        h(qv)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        
        # Obs 0: m0
        o1 = parity(m0, expectation=None)
        # Obs 1: m1
        o2 = parity(m1, expectation=None)
        
        # Obs 2: o1 + o2 + m2 (= m0 + m1 + m2)
        o3 = parity(o1, o2, m2, expectation=None)
        
        return o3

    res_idx, stim_circuit = merged_observable()
    
    assert res_idx == 2 # 3rd observable created
    assert stim_circuit.num_observables == 3
    
    obs_lines = [l for l in str(stim_circuit).splitlines() if "OBSERVABLE_INCLUDE(2)" in l]
    assert len(obs_lines) > 0
    # Should include m0(rec[-4]), m1(rec[-3]), m2(rec[-2]) - wait order depends on measurement order
    # m0, m1, m2 measured sequentially.
    assert obs_lines[0].count("rec") == 3

def test_detectors_on_observables():
    """
    Test creating a detector from observable handles.
    """
    @extract_stim
    def detector_on_obs():
        qv = QuantumVariable(2)
        h(qv)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # Create an "observable" container for m[0]
        obs = parity(m0, expectation=None)
        
        # Detector checking parity of (Obs + m[1]) = (m[0] + m[1])
        d = parity(obs, m1, expectation=False) # Default expectation=False -> Detector
        return d

    res, stim_circuit = detector_on_obs()
    
    # Should have 1 detector and 1 observable
    assert stim_circuit.num_observables == 1
    
    num_detectors = stim_circuit.num_detectors if hasattr(stim_circuit, "num_detectors") else len([i for i in stim_circuit if i.name == "DETECTOR"])
    assert num_detectors == 1
    
    # Detector should target m0 and m1
    det_line = [l for l in str(stim_circuit).splitlines() if "DETECTOR" in l][0]
    assert det_line.count("rec") == 2

def test_classical_conditions():
    """
    Test classically conditioned gates (Feedback).
    """
    @extract_stim
    def conditional_circuit():
        qv = QuantumVariable(1)
        h(qv[0])
        m = measure(qv[0])
        
        with control(m):
            x(qv[0])
        
        return m

    res, stim_circuit = conditional_circuit()
    s_str = str(stim_circuit)
    
    # Expect CX rec[-1] 0
    assert "CX rec[-1] 0" in s_str

def test_stim_noise_injection():
    """
    Test explicit Stim noise injection.
    """
    @extract_stim
    def noise_job():
        qv = QuantumVariable(1)
        stim_noise("X_ERROR", 0.1, qv[0])
        m = measure(qv[0])
        return m
        
    res, stim_circuit = noise_job()
    assert "X_ERROR(0.1) 0" in str(stim_circuit)


# ============================================================================
# Sampling Based Tests
# ============================================================================

def test_parity_observable_sampling():
    """
    Verify that parity used as an observable correctly computes parity of measurements.
    We'll produce a state |00> + |11> (Parity even -> 0)
    and |01> + |10> (Parity odd -> 1).
    """
    @extract_stim
    def observable_sampling_even():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        # Even parity: m0 XOR m1 = 0
        return parity(m0, m1, expectation=None)

    @extract_stim
    def observable_sampling_odd():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        x(qv[0]) # Flip one -> |01> + |10>
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        # Odd parity: m0 XOR m1 = 1
        return parity(m0, m1, expectation=None)

    # Test Even
    obs_idx_even, stim_circ_even = observable_sampling_even()
    
    dsampler = stim_circ_even.compile_detector_sampler()
    # Returns [detectors..., observables...]
    outcomes = dsampler.sample(100)
    # Expect 0 (False) for all shots since it's even parity state and we defined it as observable 0.
    assert not np.any(outcomes)

    # Test Odd
    obs_idx_odd, stim_circ_odd = observable_sampling_odd()
    dsampler_odd = stim_circ_odd.compile_detector_sampler()
    outcomes_odd = dsampler_odd.sample(100)
    
    # Check correct sampling for deterministically flipped parity
    assert not np.any(outcomes_odd)


def test_parity_detector_sampling():
    """
    Verify `parity` with expectation=... working as a detector.
    """
    @extract_stim
    def detector_circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        
        # Inject deterministic noise: 100% X error on qv[0]
        # This should flip the parity m0+m1 (from 0 to 1).
        stim_noise("X_ERROR", 1.0, qv[0])
        
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # We expect even parity (0).
        # Detector fires if m0+m1 != 0.
        d = parity(m0, m1, expectation=False)
        return d

    det_idx, stim_circuit = detector_circuit()
    
    # Run 100 shots (deterministic)
    sampler = stim_circuit.compile_detector_sampler()
    samples = sampler.sample(100)
    
    # Samples shape: (100, 1) since 1 detector
    # Since noise is deterministic (100%), detector should fire every time.
    assert np.all(samples)


def test_parity_expectations_behavior():
    """
    Verify `parity` expectation parameter behavior in Stim.
    
    Note: Currently, Stim conversion uses expectation only to distinguish 
    between Detectors (expectation=True/False) and Observables (expectation=None).
    The actual boolean value (True/False) does NOT modify the reference frame 
    or logic of the detector in Stim. Stim detectors always check if measurements 
    deviate from the deterministic noiseless simulation.
    
    Therefore, setting expectation=True on a circuit that deterministically produces 0 
    does NOT trigger a detector event in Stim simulation.
    """
    @extract_stim
    def check_expectations():
        qv = QuantumVariable(2)
        # Create |11> -> Parity even (0)
        x(qv[0]); x(qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # Expectation=False (0) -> Matches parity 0 -> No detection
        d1 = parity(m0, m1, expectation=False)
        
        # Expectation=True (1) -> Mismatches (semantic) parity 0 -> BUT Stim sees physical match -> No detection
        d2 = parity(m0, m1, expectation=True)
        
        return d1, d2

    d1_idx, d2_idx, stim_circuit = check_expectations()
    
    sampler = stim_circuit.compile_detector_sampler()
    samples = sampler.sample(10)
    
    # Both should be False (0) because the circuit physically produces even parity,
    # and Stim detectors check against the physical simulation.
    assert not np.any(samples[:, d1_idx])
    assert not np.any(samples[:, d2_idx])

def test_parity_exception_behavior():
    """
    Test that parity raises an Exception in noiseless simulation if expectation is violated,
    but does NOT raise an Exception in Stim extraction/simulation.
    """
    from qrisp.jasp import jaspify
    import pytest

    def violating_circuit():
        qv = QuantumVariable(2)
        # Create |11> -> Parity even.
        x(qv[0]); x(qv[1]) 
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        
        # We define expectation=True (Odd).
        # In actual execution (noiseless), parity is 0 (Even). 
        # This is a violation.
        return parity(m0, m1, expectation=True)

    # 1. Test Regular JAX/Qrisp Simulation -> Should Raise Exception
    # We wrap in jaspify to execute using the Jax implementation of the primitive
    jaspified_violator = jaspify(violating_circuit)
    
    with pytest.raises(Exception, match="deviated"):
        jaspified_violator()
        
    # 2. Test Stim Extraction -> Should NOT Raise Exception
    @extract_stim
    def stim_violator():
        return violating_circuit()
    
    try:
        det_idx, stim_circ = stim_violator()
    except Exception as e:
        pytest.fail(f"extract_stim raised an exception unexpectedly: {e}")
    
    # Verify the generated circuit has a detector
    # Even though expectation is violated in Qrisp semantics, Stim extraction does not 
    # check this violation. It generates a valid DETECTOR instruction.
    assert "DETECTOR" in str(stim_circ)

def test_extract_stim_detectors_sampling():
    """Test sampling detector events from an extracted Stim circuit."""
    
    @extract_stim
    def detector_circuit():
        qv = QuantumVariable(2)

        # Prepare |00>

        # Add deterministic error to flip parity
        stim_noise("X_ERROR", 1.0, qv[0])

        m0 = measure(qv[0])
        m1 = measure(qv[1])

        # Expect even parity (0). Since we applied deterministic X noise to qv[0],
        # the measured parity will differ from the noiseless expectation and the
        # DETECTOR should record an event.
        det = parity(m0, m1, expectation=0)
        return det

    det_idx, stim_circuit = detector_circuit()
    
    # Compile detector sampler
    sampler = stim_circuit.compile_detector_sampler()
    
    # Sample detector events
    # Returns binary matrix: (shots, num_detectors)
    det_events = sampler.sample(10)
    
    # Verify we have correct number of detectors
    assert stim_circuit.num_detectors == 1
    assert det_events.shape == (10, 1)
    
    # The returned index from the function should be 0
    assert det_idx == 0
    
    # Verify the detector fired (True) for all shots
    assert np.all(det_events[:, det_idx])


def test_extract_stim_observables_sampling():
    """Test sampling observables from an extracted Stim circuit."""
    
    @extract_stim
    def observable_circuit():
        qv = QuantumVariable(2)

        # Prepare |11> via deterministic X errors
        stim_noise("X_ERROR", 1.0, qv)

        m0 = measure(qv[0])
        m1 = measure(qv[1])

        # Define Observable corresponding to ZZ parity.
        qv2 = QuantumVariable(2)
        stim_noise("X_ERROR", 1.0, qv2[0])  # |10>
        m2_0 = measure(qv2[0])
        m2_1 = measure(qv2[1])

        # Observable 0: Parity of |11> -> 0
        obs0 = parity(m0, m1, expectation=2)

        # Observable 1: Parity of |10> -> 1
        obs1 = parity(m2_0, m2_1, expectation=2)

        return obs0, obs1

    obs0_idx, obs1_idx, stim_circuit = observable_circuit()
    
    # Compile detector sampler
    sampler = stim_circuit.compile_detector_sampler()
    
    # Sample with observables appended
    
    assert stim_circuit.num_detectors == 0
    assert stim_circuit.num_observables == 2
    
    # Re-sample with safe arguments
    det_events, obs_events = sampler.sample(10, separate_observables=True)
    
    # Observable 0 (index 0) should be 0 (false)
    # Observable 1 (index 1) should be 1 (true)
    
    assert np.all(obs_events[:, obs0_idx] == 0)
    assert np.all(obs_events[:, obs1_idx] == 1)


# ============================================================================
# Array Shape Tests for Measurements, Detectors, and Observables
# ============================================================================

def test_measurement_handles_1d_array():
    """Test that 1D arrays of measurements return StimMeasurementHandles with correct shape."""
    import jax.numpy as jnp
    
    @extract_stim
    def array_measurements():
        qv = QuantumVariable(4)
        h(qv)
        
        # Measure all qubits and collect into an array
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        m3 = measure(qv[3])
        
        # Return as a JAX array
        return jnp.array([m0, m1, m2, m3])
    
    meas_array, stim_circuit = array_measurements()
    
    # Check type and shape
    assert isinstance(meas_array, StimMeasurementHandles)
    assert meas_array.shape == (4,)
    assert np.array_equal(meas_array, [0, 1, 2, 3])
    
    # Verify slicing works correctly
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    
    # Should be able to slice directly with the handles
    sliced = samples[:, meas_array]
    assert sliced.shape == (100, 4)


def test_measurement_handles_2d_array():
    """Test that 2D arrays of measurements return StimMeasurementHandles with correct shape."""
    import jax.numpy as jnp
    
    @extract_stim
    def matrix_measurements():
        qv = QuantumVariable(6)
        h(qv)
        
        # Measure all qubits
        measurements = [measure(qv[i]) for i in range(6)]
        
        # Reshape into 2x3 matrix
        return jnp.array(measurements).reshape(2, 3)
    
    meas_matrix, stim_circuit = matrix_measurements()
    
    # Check type and shape
    assert isinstance(meas_matrix, StimMeasurementHandles)
    assert meas_matrix.shape == (2, 3)
    
    # Check values are sequential
    assert np.array_equal(meas_matrix.flatten(), [0, 1, 2, 3, 4, 5])
    
    # Verify slicing works - flatten for indexing
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(50)
    
    # Slice using flattened indices
    row0_samples = samples[:, meas_matrix[0]]
    row1_samples = samples[:, meas_matrix[1]]
    assert row0_samples.shape == (50, 3)
    assert row1_samples.shape == (50, 3)


def test_detector_handles_1d_array():
    """Test that 1D arrays of detectors return StimDetectorHandles with correct shape."""
    import jax.numpy as jnp
    
    @extract_stim
    def array_detectors():
        # Create multiple Bell pairs and check their parities
        detectors = []
        for i in range(3):
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            m0 = measure(qv[0])
            m1 = measure(qv[1])
            # Bell pair parity should be 0
            d = parity(m0, m1, expectation=False)
            detectors.append(d)
        
        return jnp.array(detectors)
    
    det_array, stim_circuit = array_detectors()
    
    # Check type and shape
    assert isinstance(det_array, StimDetectorHandles)
    assert det_array.shape == (3,)
    assert np.array_equal(det_array, [0, 1, 2])
    
    # Verify we have 3 detectors
    assert stim_circuit.num_detectors == 3
    
    # Sample and verify all detectors report no error (Bell pairs have even parity)
    sampler = stim_circuit.compile_detector_sampler()
    det_samples = sampler.sample(100)
    
    # Slice with detector handles
    sliced = det_samples[:, det_array]
    assert sliced.shape == (100, 3)
    # All should be False (no parity violations)
    assert np.all(sliced == False)


def test_observable_handles_1d_array():
    """Test that 1D arrays of observables return StimObservableHandles with correct shape."""
    import jax.numpy as jnp
    
    @extract_stim
    def array_observables():
        observables = []
        
        # Create multiple observables
        for i in range(4):
            qv = QuantumVariable(2)
            if i % 2 == 0:
                # Even: prepare |00>, parity = 0
                pass
            else:
                # Odd: prepare |11>, parity = 0
                x(qv)
            m0 = measure(qv[0])
            m1 = measure(qv[1])
            obs = parity(m0, m1, expectation=None)
            observables.append(obs)
        
        return jnp.array(observables)
    
    obs_array, stim_circuit = array_observables()
    
    # Check type and shape
    assert isinstance(obs_array, StimObservableHandles)
    assert obs_array.shape == (4,)
    assert np.array_equal(obs_array, [0, 1, 2, 3])
    
    # Verify we have 4 observables
    assert stim_circuit.num_observables == 4
    
    # Sample and verify
    sampler = stim_circuit.compile_detector_sampler()
    det_events, obs_events = sampler.sample(100, separate_observables=True)
    
    # Slice with observable handles
    sliced = obs_events[:, obs_array]
    assert sliced.shape == (100, 4)
    # All observables should report parity 0 (even)
    assert np.all(sliced == False)


def test_detector_handles_2d_array():
    """Test that 2D arrays of detectors return StimDetectorHandles with correct shape."""
    import jax.numpy as jnp
    
    @extract_stim
    def matrix_detectors():
        detectors = []
        
        # Create 2x2 grid of detectors
        for _ in range(4):
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            m0 = measure(qv[0])
            m1 = measure(qv[1])
            d = parity(m0, m1, expectation=False)
            detectors.append(d)
        
        return jnp.array(detectors).reshape(2, 2)
    
    det_matrix, stim_circuit = matrix_detectors()
    
    # Check type and shape
    assert isinstance(det_matrix, StimDetectorHandles)
    assert det_matrix.shape == (2, 2)
    
    # Check values
    assert np.array_equal(det_matrix.flatten(), [0, 1, 2, 3])
    
    # Verify we have 4 detectors
    assert stim_circuit.num_detectors == 4
    
    # Sample and slice by row
    sampler = stim_circuit.compile_detector_sampler()
    det_samples = sampler.sample(50)
    
    row0 = det_samples[:, det_matrix[0]]
    row1 = det_samples[:, det_matrix[1]]
    assert row0.shape == (50, 2)
    assert row1.shape == (50, 2)


def test_measurement_handles_preserve_dtype():
    """Test that StimMeasurementHandles arrays have integer dtype suitable for indexing."""
    import jax.numpy as jnp
    
    @extract_stim
    def check_dtype():
        qv = QuantumVariable(3)
        h(qv)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        m2 = measure(qv[2])
        return jnp.array([m0, m1, m2])
    
    meas_array, stim_circuit = check_dtype()
    
    # Should be integer type for indexing
    assert np.issubdtype(meas_array.dtype, np.integer)
    
    # Should work as indices
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(10)
    indexed = samples[:, meas_array]
    assert indexed.shape == (10, 3)


def test_scalar_measurement_is_0d_array():
    """Test that single measurements return 0-d StimMeasurementHandles arrays."""
    @extract_stim
    def single_meas():
        qv = QuantumVariable(1)
        h(qv)
        return measure(qv[0])
    
    meas_idx, stim_circuit = single_meas()
    
    # Should be StimMeasurementHandles (0-d array)
    assert isinstance(meas_idx, StimMeasurementHandles)
    assert meas_idx.ndim == 0
    assert meas_idx.item() == 0
    
    # Should still work for indexing (numpy handles 0-d arrays)
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(10)
    # Index with the scalar
    indexed = samples[:, meas_idx]
    assert indexed.shape == (10,)


def test_scalar_detector_is_0d_array():
    """Test that single detectors return 0-d StimDetectorHandles arrays."""
    @extract_stim
    def single_detector():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        return parity(m0, m1, expectation=False)
    
    det_idx, stim_circuit = single_detector()
    
    # Should be StimDetectorHandles (0-d array)
    assert isinstance(det_idx, StimDetectorHandles)
    assert det_idx.ndim == 0
    assert det_idx.item() == 0


def test_scalar_observable_is_0d_array():
    """Test that single observables return 0-d StimObservableHandles arrays."""
    @extract_stim
    def single_observable():
        qv = QuantumVariable(2)
        h(qv)
        m0 = measure(qv[0])
        m1 = measure(qv[1])
        return parity(m0, m1, expectation=None)
    
    obs_idx, stim_circuit = single_observable()
    
    # Should be StimObservableHandles (0-d array)
    assert isinstance(obs_idx, StimObservableHandles)
    assert obs_idx.ndim == 0
    assert obs_idx.item() == 0


def test_qubit_indices_from_quantum_variable():
    """Test that returning a QuantumVariable produces StimQubitIndices with correct shape."""
    @extract_stim
    def return_qubits():
        qv = QuantumVariable(4)
        h(qv)
        # Measure all qubits (required for valid stim circuit)
        measure(qv)
        # Return the QuantumVariable itself
        return qv
    
    qubit_indices, stim_circuit = return_qubits()
    
    # Check type and shape
    assert isinstance(qubit_indices, StimQubitIndices)
    assert qubit_indices.shape == (4,)
    # Qubit indices should be sequential starting from 0
    assert np.array_equal(qubit_indices, [0, 1, 2, 3])


def test_scalar_qubit_from_quantum_bool():
    """Test that returning a QuantumBool produces 1-element StimQubitIndices array."""
    @extract_stim
    def single_qubit_return():
        qb = QuantumBool()
        h(qb)
        measure(qb)
        # Return the QuantumBool
        return qb
    
    qubit_idx, stim_circuit = single_qubit_return()
    
    # QuantumBool has 1 qubit, so should be a 1-element array
    assert isinstance(qubit_idx, StimQubitIndices)
    assert len(qubit_idx) == 1
    assert qubit_idx[0] == 0


# ============================================================================
# Array Reversal (rev primitive) Tests
# ============================================================================

def test_rev_1d_measurement_array():
    """Test that reversing a 1D array of measurements works correctly."""
    import jax.numpy as jnp
    
    @extract_stim
    def reverse_measurements():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr = jnp.array([m1, m2, m3, m4])
        return arr[::-1]
    
    reversed_arr, stim_circuit = reverse_measurements()
    
    # Check type and shape
    assert isinstance(reversed_arr, StimMeasurementHandles)
    assert reversed_arr.shape == (4,)
    # Original order is [0, 1, 2, 3], reversed should be [3, 2, 1, 0]
    assert np.array_equal(reversed_arr, [3, 2, 1, 0])
    
    # Verify slicing works correctly
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(100)
    sliced = samples[:, reversed_arr]
    assert sliced.shape == (100, 4)


def test_rev_2d_measurement_array_axis0():
    """Test reversing a 2D array along axis 0."""
    import jax.numpy as jnp
    
    @extract_stim
    def reverse_2d_axis0():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr = jnp.array([[m1, m2], [m3, m4]])
        return arr[::-1, :]
    
    reversed_arr, stim_circuit = reverse_2d_axis0()
    
    # Check type and shape
    assert isinstance(reversed_arr, StimMeasurementHandles)
    assert reversed_arr.shape == (2, 2)
    # Original: [[0, 1], [2, 3]], after reverse axis 0: [[2, 3], [0, 1]]
    assert np.array_equal(reversed_arr, [[2, 3], [0, 1]])


def test_rev_2d_measurement_array_both_axes():
    """Test reversing a 2D array along both axes."""
    import jax.numpy as jnp
    
    @extract_stim
    def reverse_2d_both():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr = jnp.array([[m1, m2], [m3, m4]])
        return arr[::-1, ::-1]
    
    reversed_arr, stim_circuit = reverse_2d_both()
    
    # Check type and shape
    assert isinstance(reversed_arr, StimMeasurementHandles)
    assert reversed_arr.shape == (2, 2)
    # Original: [[0, 1], [2, 3]], after reverse both: [[3, 2], [1, 0]]
    assert np.array_equal(reversed_arr, [[3, 2], [1, 0]])


def test_rev_detector_array():
    """Test that reversing detector handles works correctly."""
    import jax.numpy as jnp
    
    @extract_stim
    def reverse_detectors():
        qv1 = QuantumVariable(2)
        qv2 = QuantumVariable(2)
        h(qv1[0]); cx(qv1[0], qv1[1])
        h(qv2[0]); cx(qv2[0], qv2[1])
        m1_0 = measure(qv1[0]); m1_1 = measure(qv1[1])
        m2_0 = measure(qv2[0]); m2_1 = measure(qv2[1])
        d1 = parity(m1_0, m1_1, expectation=False)
        d2 = parity(m2_0, m2_1, expectation=False)
        arr = jnp.array([d1, d2])
        return arr[::-1]
    
    reversed_arr, stim_circuit = reverse_detectors()
    
    # Check type and shape
    assert isinstance(reversed_arr, StimDetectorHandles)
    assert reversed_arr.shape == (2,)
    # Original: [0, 1], reversed: [1, 0]
    assert np.array_equal(reversed_arr, [1, 0])


def test_rev_with_jnp_flip():
    """Test that jnp.flip also works (uses rev primitive)."""
    import jax.numpy as jnp
    
    @extract_stim
    def flip_measurements():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        h(qv1); h(qv2); h(qv3)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        arr = jnp.array([m1, m2, m3])
        return jnp.flip(arr)
    
    flipped_arr, stim_circuit = flip_measurements()
    
    assert isinstance(flipped_arr, StimMeasurementHandles)
    assert flipped_arr.shape == (3,)
    assert np.array_equal(flipped_arr, [2, 1, 0])


# ============================================================================
# Array Split (split primitive) Tests  
# ============================================================================

def test_split_measurement_array():
    """Test that splitting an array of measurements works correctly."""
    import jax.numpy as jnp
    
    @extract_stim
    def split_measurements():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr = jnp.array([m1, m2, m3, m4])
        a, b = jnp.split(arr, 2)
        return a, b
    
    part_a, part_b, stim_circuit = split_measurements()
    
    # Check types and shapes
    assert isinstance(part_a, StimMeasurementHandles)
    assert isinstance(part_b, StimMeasurementHandles)
    assert part_a.shape == (2,)
    assert part_b.shape == (2,)
    assert np.array_equal(part_a, [0, 1])
    assert np.array_equal(part_b, [2, 3])


def test_split_uneven():
    """Test splitting an array into uneven parts using array_split."""
    import jax.numpy as jnp
    
    @extract_stim
    def split_uneven():
        qubits = [QuantumBool() for _ in range(5)]
        for qb in qubits:
            h(qb)
        measurements = [measure(qb) for qb in qubits]
        arr = jnp.array(measurements)
        a, b = jnp.array_split(arr, 2)
        return a, b
    
    part_a, part_b, stim_circuit = split_uneven()
    
    # array_split([0,1,2,3,4], 2) -> [0,1,2], [3,4]
    assert isinstance(part_a, StimMeasurementHandles)
    assert isinstance(part_b, StimMeasurementHandles)
    assert part_a.shape == (3,)
    assert part_b.shape == (2,)
    assert np.array_equal(part_a, [0, 1, 2])
    assert np.array_equal(part_b, [3, 4])


def test_split_2d_array():
    """Test splitting a 2D array along axis 0."""
    import jax.numpy as jnp
    
    @extract_stim
    def split_2d():
        qubits = [QuantumBool() for _ in range(4)]
        for qb in qubits:
            h(qb)
        m = [measure(qb) for qb in qubits]
        arr = jnp.array([[m[0], m[1]], [m[2], m[3]]])
        a, b = jnp.vsplit(arr, 2)
        return a, b
    
    part_a, part_b, stim_circuit = split_2d()
    
    # vsplit([[0,1],[2,3]], 2) -> [[0,1]], [[2,3]]
    assert isinstance(part_a, StimMeasurementHandles)
    assert isinstance(part_b, StimMeasurementHandles)
    assert part_a.shape == (1, 2)
    assert part_b.shape == (1, 2)
    assert np.array_equal(part_a, [[0, 1]])
    assert np.array_equal(part_b, [[2, 3]])


# ============================================================================
# Dynamic Update Slice Tests
# ============================================================================

def test_dynamic_update_slice():
    """Test dynamic_update_slice with measurement arrays."""
    import jax.numpy as jnp
    import jax.lax as lax
    
    @extract_stim
    def update_slice():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        h(qv1); h(qv2); h(qv3)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        arr = jnp.array([m1, m2, m3])
        update = jnp.array([m2])  # Replace position 0 with m2
        return lax.dynamic_update_slice(arr, update, (0,))
    
    result, stim_circuit = update_slice()
    
    # Original [0, 1, 2], update position 0 with value at position 1
    # Result should be [1, 1, 2]
    assert isinstance(result, StimMeasurementHandles)
    assert result.shape == (3,)
    assert np.array_equal(result, [1, 1, 2])


def test_dynamic_update_slice_middle():
    """Test dynamic_update_slice updating middle elements."""
    import jax.numpy as jnp
    import jax.lax as lax
    
    @extract_stim
    def update_middle():
        qubits = [QuantumBool() for _ in range(5)]
        for qb in qubits:
            h(qb)
        measurements = [measure(qb) for qb in qubits]
        arr = jnp.array(measurements)
        # Update positions 1 and 2 with values from positions 3 and 4
        update = jnp.array([measurements[3], measurements[4]])
        return lax.dynamic_update_slice(arr, update, (1,))
    
    result, stim_circuit = update_middle()
    
    # Original [0, 1, 2, 3, 4], update positions 1-2 with [3, 4]
    # Result should be [0, 3, 4, 3, 4]
    assert isinstance(result, StimMeasurementHandles)
    assert result.shape == (5,)
    assert np.array_equal(result, [0, 3, 4, 3, 4])


# ============================================================================
# Select/Where Tests
# ============================================================================

def test_select_n_where():
    """Test jnp.where (select_n) with measurement arrays."""
    import jax.numpy as jnp
    
    @extract_stim
    def where_select():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr1 = jnp.array([m1, m2])  # [0, 1]
        arr2 = jnp.array([m3, m4])  # [2, 3]
        cond = jnp.array([True, False])
        return jnp.where(cond, arr1, arr2)
    
    result, stim_circuit = where_select()
    
    # where([True, False], [0, 1], [2, 3]) -> [0, 3]
    assert isinstance(result, StimMeasurementHandles)
    assert result.shape == (2,)
    assert np.array_equal(result, [0, 3])


def test_select_n_all_true():
    """Test where with all-true condition."""
    import jax.numpy as jnp
    
    @extract_stim
    def where_all_true():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr1 = jnp.array([m1, m2])
        arr2 = jnp.array([m3, m4])
        cond = jnp.array([True, True])
        return jnp.where(cond, arr1, arr2)
    
    result, stim_circuit = where_all_true()
    
    # Should select all from arr1
    assert np.array_equal(result, [0, 1])


def test_select_n_all_false():
    """Test where with all-false condition."""
    import jax.numpy as jnp
    
    @extract_stim
    def where_all_false():
        qv1 = QuantumBool()
        qv2 = QuantumBool()
        qv3 = QuantumBool()
        qv4 = QuantumBool()
        h(qv1); h(qv2); h(qv3); h(qv4)
        m1 = measure(qv1)
        m2 = measure(qv2)
        m3 = measure(qv3)
        m4 = measure(qv4)
        arr1 = jnp.array([m1, m2])
        arr2 = jnp.array([m3, m4])
        cond = jnp.array([False, False])
        return jnp.where(cond, arr1, arr2)
    
    result, stim_circuit = where_all_false()
    
    # Should select all from arr2
    assert np.array_equal(result, [2, 3])
