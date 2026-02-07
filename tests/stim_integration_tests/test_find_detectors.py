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
