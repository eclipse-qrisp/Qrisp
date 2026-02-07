"""
Tests for the find_detectors decorator.

Each test is a standalone ``test_*`` function at module scope so it can be
run individually with ``pytest tests/test_find_detectors.py::test_name``
or manually via ``python -c "from tests.test_find_detectors import test_name; test_name()"``.
"""
import pytest
import stim
from qrisp import QuantumArray, QuantumBool, QuantumFloat, cx, measure, reset, h
from qrisp.jasp.evaluation_tools.stim_extraction import extract_stim

from qrisp.misc.stim_tools.find_detectors import (
    find_detectors,
    _prepare_for_tqecd,
    _RESET_OPS,
    _MEASUREMENT_OPS,
    _TWO_QUBIT_OPS,
)


# ─────────────────────── helpers ───────────────────────────────────────

def _run_and_get_stim(main_fn):
    """Call an @extract_stim function and return the stim circuit."""
    result = main_fn()
    return result[-1]


def _sample_detectors(stim_circ, shots=2000):
    """Return detector sample array."""
    sampler = stim_circ.compile_detector_sampler()
    return sampler.sample(shots)


# ─────────────── single-round syndrome extraction ─────────────────────

def test_single_round_detectors_found():
    """Two parity checks should produce two detectors."""

    @find_detectors
    def syndrome(qa):
        reset(qa[3])
        reset(qa[4])
        cx(qa[0], qa[3])
        cx(qa[1], qa[3])
        cx(qa[1], qa[4])
        cx(qa[2], qa[4])
        return measure(qa[3]), measure(qa[4])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m3, m4 = syndrome(qa)
        return detectors, m3, m4

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 2


def test_single_round_noiseless_zero():
    """In a noiseless circuit all detector samples must be 0."""

    @find_detectors
    def syndrome(qa):
        reset(qa[3])
        reset(qa[4])
        cx(qa[0], qa[3])
        cx(qa[1], qa[3])
        cx(qa[1], qa[4])
        cx(qa[2], qa[4])
        return measure(qa[3]), measure(qa[4])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m3, m4 = syndrome(qa)
        return detectors, m3, m4

    stim_circ = _run_and_get_stim(main)
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0, "Noiseless detectors should all be 0"


# ──────────────── two-round repetition code ────────────────────────────

def _build_two_round_rep_code():
    @find_detectors
    def two_rounds(qa):
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        cx(qa[2], qa[3]); cx(qa[4], qa[3])
        m1_r1 = measure(qa[1]); m3_r1 = measure(qa[3])
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        cx(qa[2], qa[3]); cx(qa[4], qa[3])
        m1_r2 = measure(qa[1]); m3_r2 = measure(qa[3])
        return m1_r1, m3_r1, m1_r2, m3_r2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m1_r1, m3_r1, m1_r2, m3_r2 = two_rounds(qa)
        return detectors, m1_r1, m3_r1, m1_r2, m3_r2

    return main


def test_two_round_cross_round_detectors():
    """Two rounds should yield >= 4 detectors (2 per round + cross-round)."""
    stim_circ = _run_and_get_stim(_build_two_round_rep_code())
    assert stim_circ.num_detectors >= 4


def test_two_round_noiseless_zero():
    stim_circ = _run_and_get_stim(_build_two_round_rep_code())
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


# ───────────────── return_circuits debug mode ──────────────────────────

def test_return_circuits_no_crash():
    """return_circuits=True must not raise."""

    @find_detectors(return_circuits=True)
    def syndrome(qa):
        reset(qa[1])
        cx(qa[0], qa[1])
        cx(qa[2], qa[1])
        return measure(qa[1])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m1, raw, tqecd_in, annotated = syndrome(qa)
        return detectors, m1

    _run_and_get_stim(main)


def test_return_circuits_are_stim_instances():
    """Intermediate circuits should be stim.Circuit instances."""

    @find_detectors(return_circuits=True)
    def syndrome(qa):
        reset(qa[1])
        cx(qa[0], qa[1])
        cx(qa[2], qa[1])
        return measure(qa[1])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        result = syndrome(qa)
        detectors, m1 = result[0], result[1]
        raw_stim, tqecd_input, annotated = result[2], result[3], result[4]
        assert isinstance(raw_stim, stim.Circuit)
        assert isinstance(tqecd_input, stim.Circuit)
        assert isinstance(annotated, stim.Circuit)
        return detectors, m1

    _run_and_get_stim(main)


# ─────────────── _prepare_for_tqecd unit tests ────────────────────────

def test_tqecd_qubit_coords_present():
    """Output must have QUBIT_COORDS for every qubit."""
    circ = stim.Circuit("R 0 1 2\nCX 0 1\nM 0 1 2")
    out = _prepare_for_tqecd(circ)
    coords = [i for i in out.flattened() if i.name == "QUBIT_COORDS"]
    assert len(coords) == 3


def test_tqecd_implicit_resets_first_round():
    """Qubits not explicitly reset should get an implicit R."""
    circ = stim.Circuit("R 0\nCX 0 1\nM 0 1")
    out = _prepare_for_tqecd(circ)
    reset_targets = set()
    for inst in out.flattened():
        if inst.name in _RESET_OPS:
            for t in inst.targets_copy():
                reset_targets.add(t.value)
    assert 0 in reset_targets
    assert 1 in reset_targets


def test_tqecd_two_round_splitting():
    """Two measurement layers should still produce exactly 2 measurements."""
    circ = stim.Circuit("R 0 1\nCX 0 1\nM 1\nR 1\nCX 0 1\nM 1")
    out = _prepare_for_tqecd(circ)
    meas_count = sum(1 for i in out.flattened() if i.name in _MEASUREMENT_OPS)
    assert meas_count == 2


def test_tqecd_two_qubit_gate_moments():
    """Each two-qubit gate should be followed by a TICK."""
    circ = stim.Circuit("R 0 1 2 3\nCX 0 1 2 3\nM 0 1 2 3")
    out = _prepare_for_tqecd(circ)
    instructions = list(out.flattened())
    cx_indices = [i for i, inst in enumerate(instructions) if inst.name == "CX"]
    for idx in cx_indices:
        assert instructions[idx + 1].name == "TICK"


# ─────────────── edge cases ───────────────────────────────────────────

def test_no_reset_pipeline_survives():
    """CX + measure (no reset) — pipeline must not crash."""

    @find_detectors
    def no_meas(qa):
        cx(qa[0], qa[1])
        return measure(qa[0])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(2,))
        detectors, m = no_meas(qa)
        return detectors, m

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_measurements >= 1


def test_single_qubit_two_rounds():
    """One qubit, reset+measure twice -> detector."""

    @find_detectors
    def minimal(qa):
        reset(qa[0])
        m1 = measure(qa[0])
        reset(qa[0])
        m2 = measure(qa[0])
        return m1, m2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(1,))
        detectors, m1, m2 = minimal(qa)
        return detectors, m1, m2

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 1
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_type_error_non_quantum_array():
    """Passing a non-QuantumArray should raise TypeError."""

    @find_detectors
    def bad(x):
        return measure(x)

    with pytest.raises(TypeError, match="QuantumArray"):
        @extract_stim
        def main():
            bad(42)
        main()


# ─────────────── stress tests ─────────────────────────────────────────

def test_stress_single_return():
    """Single return value (not a tuple) — normalize-to-tuple logic."""

    @find_detectors
    def single_ret(qa):
        reset(qa[0])
        return measure(qa[0])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(1,))
        detectors, m = single_ret(qa)
        return detectors, m

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_measurements >= 1


def test_stress_multiple_quantum_arrays():
    """Multiple QuantumArray args receive disjoint qubit slices."""

    @find_detectors
    def multi_array(data, ancilla):
        reset(ancilla[0])
        cx(data[0], ancilla[0])
        cx(data[1], ancilla[0])
        return measure(ancilla[0])

    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(2,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(1,))
        detectors, m = multi_array(data, ancilla)
        return detectors, m

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_measurements >= 1


def test_stress_three_rounds_detectors():
    """Three rounds with 1 ancilla -> at least 3 detectors."""

    @find_detectors
    def three_rounds(qa):
        results = []
        for _ in range(3):
            reset(qa[1])
            cx(qa[0], qa[1])
            cx(qa[2], qa[1])
            results.append(measure(qa[1]))
        return tuple(results)

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m1, m2, m3 = three_rounds(qa)
        return detectors, m1, m2, m3

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 3


def test_stress_three_rounds_noiseless():
    @find_detectors
    def three_rounds(qa):
        results = []
        for _ in range(3):
            reset(qa[1])
            cx(qa[0], qa[1])
            cx(qa[2], qa[1])
            results.append(measure(qa[1]))
        return tuple(results)

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m1, m2, m3 = three_rounds(qa)
        return detectors, m1, m2, m3

    stim_circ = _run_and_get_stim(main)
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_stress_partial_return():
    """Return subset of measurements — unreferenced detectors filtered."""

    @find_detectors
    def partial(qa):
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        cx(qa[2], qa[3]); cx(qa[4], qa[3])
        m1 = measure(qa[1])
        m3 = measure(qa[3])
        return m1  # discard m3

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m1 = partial(qa)
        return detectors, m1

    stim_circ = _run_and_get_stim(main)
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_stress_reversed_return_order():
    """Return measurements in opposite order to execution."""

    @find_detectors
    def reversed_ret(qa):
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        cx(qa[2], qa[3]); cx(qa[4], qa[3])
        m1 = measure(qa[1])
        m3 = measure(qa[3])
        return m3, m1  # reversed

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m3, m1 = reversed_ret(qa)
        return detectors, m3, m1

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 1
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def _build_d5_rep_code():
    @find_detectors
    def rep_code_d5(qa):
        ancillas = [1, 3, 5, 7]
        results = []
        for _ in range(2):
            for a in ancillas:
                reset(qa[a])
            for a in ancillas:
                cx(qa[a - 1], qa[a])
                cx(qa[a + 1], qa[a])
            for a in ancillas:
                results.append(measure(qa[a]))
        return tuple(results)

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(9,))
        result = rep_code_d5(qa)
        detectors = result[0]
        measurements = result[1:]
        return (detectors,) + measurements

    return main


def test_stress_d5_rep_code_detectors():
    """Distance-5 rep code, 2 rounds, 9 qubits — 8 detectors, 8 meas."""
    stim_circ = _run_and_get_stim(_build_d5_rep_code())
    assert stim_circ.num_detectors >= 8
    assert stim_circ.num_measurements == 8


def test_stress_d5_rep_code_noiseless():
    stim_circ = _run_and_get_stim(_build_d5_rep_code())
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_stress_no_reset_between_rounds():
    """No explicit reset between rounds — should not crash."""

    @find_detectors
    def no_reset_round2(qa):
        reset(qa[1])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        m1 = measure(qa[1])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        m2 = measure(qa[1])
        return m1, m2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m1, m2 = no_reset_round2(qa)
        return detectors, m1, m2

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_measurements >= 2


def test_stress_reset_unmeasured_qubit():
    """Reset a qubit that is never measured."""

    @find_detectors
    def reset_extra(qa):
        reset(qa[0])
        reset(qa[1])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        return measure(qa[1])

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m = reset_extra(qa)
        return detectors, m

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_measurements >= 1


def test_stress_back_to_back_measurements():
    """Back-to-back measurements on different qubits."""

    @find_detectors
    def back_to_back(qa):
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[3])
        m1 = measure(qa[1]); m3 = measure(qa[3])
        reset(qa[1]); reset(qa[3])
        cx(qa[0], qa[1]); cx(qa[2], qa[3])
        m1b = measure(qa[1]); m3b = measure(qa[3])
        return m1, m3, m1b, m3b

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(4,))
        detectors, m1, m3, m1b, m3b = back_to_back(qa)
        return detectors, m1, m3, m1b, m3b

    stim_circ = _run_and_get_stim(main)
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_stress_return_circuits_inspection():
    """return_circuits=True debug circuits are accessible and valid."""

    @find_detectors(return_circuits=True)
    def debug_fn(qa):
        reset(qa[1])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        m = measure(qa[1])
        reset(qa[1])
        cx(qa[0], qa[1]); cx(qa[2], qa[1])
        m2 = measure(qa[1])
        return m, m2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        result = debug_fn(qa)
        detectors, m, m2 = result[0], result[1], result[2]
        raw, tqecd_in, annotated = result[3], result[4], result[5]
        assert isinstance(raw, stim.Circuit)
        assert isinstance(tqecd_in, stim.Circuit)
        assert isinstance(annotated, stim.Circuit)
        assert "QUBIT_COORDS" in str(tqecd_in)
        assert "DETECTOR" in str(annotated)
        return detectors, m, m2

    _run_and_get_stim(main)


def test_stress_interleaved_reset_gate():
    """Interleaved reset/gate ordering within a round."""

    @find_detectors
    def interleaved(qa):
        reset(qa[1]); cx(qa[0], qa[1])
        reset(qa[3]); cx(qa[2], qa[3])
        m1 = measure(qa[1]); m3 = measure(qa[3])
        reset(qa[1]); cx(qa[0], qa[1])
        reset(qa[3]); cx(qa[2], qa[3])
        m1b = measure(qa[1]); m3b = measure(qa[3])
        return m1, m3, m1b, m3b

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(4,))
        detectors, m1, m3, m1b, m3b = interleaved(qa)
        return detectors, m1, m3, m1b, m3b

    stim_circ = _run_and_get_stim(main)
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_stress_double_measure_same_qubit():
    """Measure same qubit twice in one round, no reset between."""

    @find_detectors
    def double_meas(qa):
        reset(qa[0]); reset(qa[1])
        cx(qa[0], qa[1])
        m1 = measure(qa[1])
        m2 = measure(qa[1])
        return m1, m2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(2,))
        detectors, m1, m2 = double_meas(qa)
        return detectors, m1, m2

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_measurements >= 2


def test_stress_weight4_check():
    """Weight-4 parity check, 2 rounds."""

    @find_detectors
    def weight4(qa):
        reset(qa[4])
        cx(qa[0], qa[4]); cx(qa[1], qa[4])
        cx(qa[2], qa[4]); cx(qa[3], qa[4])
        m = measure(qa[4])
        reset(qa[4])
        cx(qa[0], qa[4]); cx(qa[1], qa[4])
        cx(qa[2], qa[4]); cx(qa[3], qa[4])
        m2 = measure(qa[4])
        return m, m2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        detectors, m, m2 = weight4(qa)
        return detectors, m, m2

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 2
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_stress_hadamard_cnot_x_basis():
    """Hadamard + CNOT pattern (X-basis stabilizer check)."""

    @find_detectors
    def x_check(qa):
        reset(qa[2]); h(qa[2])
        cx(qa[2], qa[0]); cx(qa[2], qa[1])
        h(qa[2]); m = measure(qa[2])
        reset(qa[2]); h(qa[2])
        cx(qa[2], qa[0]); cx(qa[2], qa[1])
        h(qa[2]); m2 = measure(qa[2])
        return m, m2

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m, m2 = x_check(qa)
        return detectors, m, m2

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 1
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


# ──────────────── noise handling tests ─────────────────────────────────

def test_noise_x_error_100_percent():
    """Test that expectations are computed correctly even with 100% X errors."""
    
    @find_detectors
    def syndrome_with_noise(qa):
        reset(qa[1])
        cx(qa[0], qa[1])
        cx(qa[2], qa[1])
        return measure(qa[1])
    
    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, m = syndrome_with_noise(qa)
        return detectors, m
    
    # Get the stim circuit
    stim_circ = _run_and_get_stim(main)
    
    # Manually add 100% X error on the ancilla (qubit 1) before measurement.
    # This flips the measurement outcome and should trigger the detector.
    noisy_circ = stim.Circuit()
    for instruction in stim_circ:
        # Insert X_ERROR on qubit 1 right before any measurement on it
        if instruction.name in ("M", "MR") and 1 in [t.value for t in instruction.targets_copy()]:
            noisy_circ.append("X_ERROR", [1], 1.0)
        noisy_circ.append(instruction)
    
    # The noiseless circuit should still have detectors that sample to 0
    # (the decorated function computed expectations from noiseless simulation)
    samples_noiseless = _sample_detectors(stim_circ)
    assert samples_noiseless.sum() == 0, "Noiseless circuit should have zero detector samples"
    
    # The noisy circuit should have non-zero detector samples
    samples_noisy = _sample_detectors(noisy_circ, shots=100)
    # With 100% error, we expect many non-zero detector samples
    assert samples_noisy.sum() > 0, "Noisy circuit should have non-zero detector samples"


# ──────────────── static parameter tests ───────────────────────────────

def test_static_parameter_integer():
    """Test that static integer parameters work correctly."""
    
    @find_detectors
    def syndrome_with_rounds(qa, num_rounds):
        results = []
        for _ in range(num_rounds):
            reset(qa[1])
            cx(qa[0], qa[1])
            cx(qa[2], qa[1])
            results.append(measure(qa[1]))
        return tuple(results)
    
    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        detectors, *measurements = syndrome_with_rounds(qa, 3)  # 3 rounds
        return (detectors,) + tuple(measurements)
    
    stim_circ = _run_and_get_stim(main)
    # Should find >= 3 detectors (one per round)
    assert stim_circ.num_detectors >= 3
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_static_parameter_boolean():
    """Test that static boolean parameters work correctly."""
    
    @find_detectors
    def syndrome_conditional(qa, include_second_check):
        reset(qa[1])
        cx(qa[0], qa[1])
        cx(qa[2], qa[1])
        m1 = measure(qa[1])
        
        if include_second_check:
            reset(qa[3])
            cx(qa[2], qa[3])
            cx(qa[4], qa[3])
            m2 = measure(qa[3])
            return m1, m2
        else:
            return m1,
    
    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(5,))
        result = syndrome_conditional(qa, True)  # Include second check
        detectors = result[0]
        measurements = result[1:]
        return (detectors,) + measurements
    
    stim_circ = _run_and_get_stim(main)
    # Should find >= 2 detectors (one per check)
    assert stim_circ.num_detectors >= 2
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_static_parameter_mixed():
    """Test mixing quantum arrays and multiple static parameters."""
    
    @find_detectors
    def syndrome_parameterized(data, ancilla, num_rounds, apply_hadamard):
        results = []
        for _ in range(num_rounds):
            reset(ancilla[0])
            if apply_hadamard:
                h(ancilla[0])
            cx(data[0], ancilla[0])
            cx(data[1], ancilla[0])
            if apply_hadamard:
                h(ancilla[0])
            results.append(measure(ancilla[0]))
        return tuple(results)
    
    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(2,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(1,))
        detectors, *measurements = syndrome_parameterized(data, ancilla, 2, False)
        return (detectors,) + tuple(measurements)
    
    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 2
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_static_parameter_custom_object():
    """Test that a non-flattenable custom object can be passed as a static parameter."""

    class SyndromeConfig:
        """Plain class with no JAX flattening — not a pytree, not a QuantumArray."""
        def __init__(self, num_rounds, pairs):
            self.num_rounds = num_rounds
            self.pairs = pairs  # list of (data_idx, ancilla_idx) tuples

    @find_detectors
    def syndrome_from_config(qa, cfg):
        results = []
        for _ in range(cfg.num_rounds):
            for data_idx, anc_idx in cfg.pairs:
                reset(qa[anc_idx])
                cx(qa[data_idx], qa[anc_idx])
            results.extend(measure(qa[anc_idx]) for _, anc_idx in cfg.pairs)
        return tuple(results)

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(4,))
        cfg = SyndromeConfig(num_rounds=2, pairs=[(0, 2), (1, 3)])
        detectors, *measurements = syndrome_from_config(qa, cfg)
        return (detectors,) + tuple(measurements)

    stim_circ = _run_and_get_stim(main)
    assert stim_circ.num_detectors >= 2
    samples = _sample_detectors(stim_circ)
    assert samples.sum() == 0


def test_traced_integer_raises_error():
    """Test that passing a traced (non-static) integer raises TypeError."""
    from qrisp import QuantumFloat

    @find_detectors
    def syndrome_with_param(qa, num_rounds):
        results = []
        for _ in range(num_rounds):
            reset(qa[1])
            cx(qa[0], qa[1])
            cx(qa[2], qa[1])
            results.append(measure(qa[1]))
        return tuple(results)

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(3,))
        qf = QuantumFloat(3)
        # measure returns a tracer — passing it as an integer arg should fail
        traced_val = measure(qf)
        detectors, *ms = syndrome_with_param(qa, traced_val)
        return (detectors,) + tuple(ms)

    with pytest.raises(TypeError, match="tracer was leaked"):
        main()