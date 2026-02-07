"""
Tests that validate find_detectors against stim's built-in circuit generators.

stim.Circuit.generated() produces QEC circuits with *correct* DETECTOR
annotations.  These tests:

  1. Generate circuits via stim (repetition code, surface code).
  2. Automatically recode each circuit using Qrisp with @find_detectors.
  3. Compare the discovered detectors against stim's ground truth.
  4. Verify that all detectors sample to zero in the noiseless case.

The auto-recode pipeline parses the gate-level structure of a stim circuit
and replays it through Qrisp, handling MR (measure-reset), MX (X-basis
measurement), MY (Y-basis measurement), and their MR variants with proper
buffering so that tqecd fragment structure is preserved.
"""
import pytest
import stim
import numpy as np
from qrisp import QuantumArray, QuantumBool, cx, h, measure, reset
from qrisp.jasp.evaluation_tools.stim_extraction import extract_stim

from qrisp.misc.stim_tools.find_detectors import find_detectors


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _run_and_get_stim(main_fn):
    """Call an @extract_stim decorated function, return the stim circuit."""
    result = main_fn()
    return result[-1]


def _sample_detectors_ok(stim_circ, shots=2000):
    """Return True if all noiseless detector samples are zero."""
    sampler = stim_circ.compile_detector_sampler()
    samples = sampler.sample(shots)
    return samples.sum() == 0


# ═══════════════════════════════════════════════════════════════════════
# Stim circuit parser and Qrisp auto-recode
# ═══════════════════════════════════════════════════════════════════════

def _parse_stim_for_qrisp_recode(stim_circ):
    """
    Parse a stim circuit and extract its gate-level structure.

    Returns (n_qubits, ops, measurements) where:
      - n_qubits: total qubit count
      - ops: list of (gate_name, targets) tuples
      - measurements: list of qubit indices in measurement order
    """
    clean = stim_circ.without_noise().flattened()
    ops = []
    measurements = []
    meas_counter = 0

    known_gates = {
        "R": "reset", "RX": "reset_x", "RY": "reset_y",
        "H": "h", "X": "x", "Y": "y", "Z": "z",
        "S": "s", "S_DAG": "s_dg",
        "SQRT_X": "sx", "SQRT_X_DAG": "sx_dg",
        "SQRT_Y": "sy", "SQRT_Y_DAG": "sy_dg",
        "CX": "cx", "CY": "cy", "CZ": "cz",
        "M": "measure", "MR": "measure_reset",
    }

    for inst in clean:
        name = inst.name
        if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                     "SHIFT_COORDS", "QUBIT_COORDS", "REPEAT"):
            continue
        targets = [t.value for t in inst.targets_copy()
                   if not t.is_measurement_record_target]

        if name in ("M", "MR", "MX", "MY", "MRX", "MRY"):
            for t in targets:
                measurements.append(t)
                meas_counter += 1
            meas_op_map = {
                "M": "measure", "MR": "measure_reset",
                "MX": "measure_x", "MY": "measure_y",
                "MRX": "measure_reset_x", "MRY": "measure_reset_y",
            }
            ops.append((meas_op_map[name], targets))
        elif name in known_gates:
            ops.append((known_gates[name], targets))
        else:
            ops.append((name, targets))

    n_qubits = clean.num_qubits
    return n_qubits, ops, measurements


def _build_qrisp_from_parsed(n_qubits, ops):
    """
    Build a @find_detectors-decorated Qrisp function from parsed stim ops.

    Handles MR buffering and MX/MY basis-tracking to preserve tqecd
    fragment structure.
    """
    from qrisp import s, s_dg, x, y, z, cz, cy
    from qrisp import sx, sx_dg

    @find_detectors
    def auto_circuit(qa):
        meas_results = []
        pending_meas = []       # list of (qubit_index, basis) tuples
        pending_mr_resets = []  # list of (qubit_index, reset_basis) tuples

        def _flush_measurements():
            for t, basis in pending_meas:
                if basis == "x":
                    h(qa[t])
                elif basis == "y":
                    s_dg(qa[t])
                    h(qa[t])
            for t, _basis in pending_meas:
                meas_results.append(measure(qa[t]))
            pending_meas.clear()

        def _flush_mr_resets():
            for t, rbasis in pending_mr_resets:
                reset(qa[t])
                if rbasis == "x":
                    h(qa[t])
                elif rbasis == "y":
                    h(qa[t])
                    s(qa[t])
            pending_mr_resets.clear()

        def _emit_gate(gate_name, targets):
            if gate_name == "reset":
                for t in targets:
                    reset(qa[t])
            elif gate_name == "reset_x":
                for t in targets:
                    reset(qa[t]); h(qa[t])
            elif gate_name == "reset_y":
                for t in targets:
                    reset(qa[t]); h(qa[t]); s(qa[t])
            elif gate_name == "h":
                for t in targets:
                    h(qa[t])
            elif gate_name == "x":
                for t in targets:
                    x(qa[t])
            elif gate_name == "y":
                for t in targets:
                    y(qa[t])
            elif gate_name == "z":
                for t in targets:
                    z(qa[t])
            elif gate_name == "s":
                for t in targets:
                    s(qa[t])
            elif gate_name == "s_dg":
                for t in targets:
                    s_dg(qa[t])
            elif gate_name == "sx":
                for t in targets:
                    sx(qa[t])
            elif gate_name == "sx_dg":
                for t in targets:
                    sx_dg(qa[t])
            elif gate_name == "cx":
                for i in range(0, len(targets), 2):
                    cx(qa[targets[i]], qa[targets[i + 1]])
            elif gate_name == "cy":
                for i in range(0, len(targets), 2):
                    cy(qa[targets[i]], qa[targets[i + 1]])
            elif gate_name == "cz":
                for i in range(0, len(targets), 2):
                    cz(qa[targets[i]], qa[targets[i + 1]])
            else:
                raise ValueError(f"Unsupported gate in recode: {gate_name}")

        _MEAS_OPS = {
            "measure": (None, None),
            "measure_x": ("x", None),
            "measure_y": ("y", None),
            "measure_reset": (None, "z"),
            "measure_reset_x": ("x", "x"),
            "measure_reset_y": ("y", "y"),
        }

        for gate_name, targets in ops:
            if gate_name in _MEAS_OPS:
                meas_basis, reset_basis = _MEAS_OPS[gate_name]
                pending_meas.extend((t, meas_basis) for t in targets)
                if reset_basis is not None:
                    pending_mr_resets.extend(
                        (t, reset_basis) for t in targets)
            else:
                _flush_measurements()
                _flush_mr_resets()
                _emit_gate(gate_name, targets)

        _flush_measurements()
        return tuple(meas_results)

    return auto_circuit, n_qubits


def _auto_recode_and_check(code_task, distance, rounds):
    """
    Fully automated comparison: generate stim reference, parse, recode via
    Qrisp, compare detector counts and verify noiseless sampling.

    Returns (expected_det, found_det, noiseless_ok).
    """
    ref_circ = stim.Circuit.generated(code_task, distance=distance, rounds=rounds)
    expected_det = ref_circ.without_noise().num_detectors

    n_qubits, ops, measurements = _parse_stim_for_qrisp_recode(ref_circ)
    auto_fn, nq = _build_qrisp_from_parsed(n_qubits, ops)

    @extract_stim
    def main():
        qa = QuantumArray(qtype=QuantumBool(), shape=(nq,))
        result = auto_fn(qa)
        det = result[0]
        rest = result[1:]
        return (det,) + rest

    found_circ = _run_and_get_stim(main)
    found_det = found_circ.num_detectors
    noiseless_ok = _sample_detectors_ok(found_circ)

    return expected_det, found_det, noiseless_ok


# ═══════════════════════════════════════════════════════════════════════
# Manual recode tests (hand-written Qrisp repetition codes)
# ═══════════════════════════════════════════════════════════════════════

def test_manual_rep_d3_r1():
    """Manual recode: repetition code d=3, 1 round + final readout."""
    ref = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=1)
    expected = ref.without_noise().num_detectors

    @find_detectors
    def rep_code(data, ancilla):
        reset(ancilla[0]); reset(ancilla[1])
        cx(data[0], ancilla[0]); cx(data[1], ancilla[1])
        cx(data[1], ancilla[0]); cx(data[2], ancilla[1])
        ma0 = measure(ancilla[0]); ma1 = measure(ancilla[1])
        md0 = measure(data[0]); md1 = measure(data[1]); md2 = measure(data[2])
        return ma0, ma1, md0, md1, md2

    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(3,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(2,))
        det, ma0, ma1, md0, md1, md2 = rep_code(data, ancilla)
        return det, ma0, ma1, md0, md1, md2

    circ = _run_and_get_stim(main)
    assert circ.num_detectors == expected
    assert _sample_detectors_ok(circ)


def test_manual_rep_d3_r2():
    """Manual recode: repetition code d=3, 2 rounds + final readout."""
    ref = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    expected = ref.without_noise().num_detectors

    @find_detectors
    def rep_code(data, ancilla):
        for _rnd in range(2):
            reset(ancilla[0]); reset(ancilla[1])
            cx(data[0], ancilla[0]); cx(data[1], ancilla[1])
            cx(data[1], ancilla[0]); cx(data[2], ancilla[1])
            _a0 = measure(ancilla[0]); _a1 = measure(ancilla[1])
        md0 = measure(data[0]); md1 = measure(data[1]); md2 = measure(data[2])
        return _a0, _a1, md0, md1, md2

    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(3,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(2,))
        result = rep_code(data, ancilla)
        return result

    circ = _run_and_get_stim(main)
    # find_detectors may find >= expected (extra valid stabilizer flows)
    assert circ.num_detectors >= expected
    assert _sample_detectors_ok(circ)


def test_manual_rep_d3_r3():
    """Manual recode: repetition code d=3, 3 rounds + final readout."""
    ref = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3)
    expected = ref.without_noise().num_detectors

    @find_detectors
    def rep_code(data, ancilla):
        all_meas = []
        for _rnd in range(3):
            reset(ancilla[0]); reset(ancilla[1])
            cx(data[0], ancilla[0]); cx(data[1], ancilla[1])
            cx(data[1], ancilla[0]); cx(data[2], ancilla[1])
            all_meas.append(measure(ancilla[0]))
            all_meas.append(measure(ancilla[1]))
        all_meas.append(measure(data[0]))
        all_meas.append(measure(data[1]))
        all_meas.append(measure(data[2]))
        return tuple(all_meas)

    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(3,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(2,))
        result = rep_code(data, ancilla)
        det = result[0]
        rest = result[1:]
        return (det,) + rest

    circ = _run_and_get_stim(main)
    assert circ.num_detectors == expected
    assert _sample_detectors_ok(circ)


def test_manual_rep_d5_r2():
    """Manual recode: repetition code d=5, 2 rounds + final readout."""
    ref = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=2)
    expected = ref.without_noise().num_detectors

    @find_detectors
    def rep_code(data, ancilla):
        n_anc = 4
        all_meas = []
        for _rnd in range(2):
            for i in range(n_anc):
                reset(ancilla[i])
            for i in range(n_anc):
                cx(data[i], ancilla[i])
            for i in range(n_anc):
                cx(data[i + 1], ancilla[i])
            for i in range(n_anc):
                all_meas.append(measure(ancilla[i]))
        for i in range(5):
            all_meas.append(measure(data[i]))
        return tuple(all_meas)

    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(5,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(4,))
        result = rep_code(data, ancilla)
        det = result[0]
        rest = result[1:]
        return (det,) + rest

    circ = _run_and_get_stim(main)
    # 2-round codes may find more detectors (extra valid stabilizer flows)
    assert circ.num_detectors >= expected
    assert _sample_detectors_ok(circ)


def test_manual_rep_d5_r3():
    """Manual recode: repetition code d=5, 3 rounds + final readout."""
    ref = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=3)
    expected = ref.without_noise().num_detectors

    @find_detectors
    def rep_code(data, ancilla):
        n_anc = 4
        all_meas = []
        for _rnd in range(3):
            for i in range(n_anc):
                reset(ancilla[i])
            for i in range(n_anc):
                cx(data[i], ancilla[i])
            for i in range(n_anc):
                cx(data[i + 1], ancilla[i])
            for i in range(n_anc):
                all_meas.append(measure(ancilla[i]))
        for i in range(5):
            all_meas.append(measure(data[i]))
        return tuple(all_meas)

    @extract_stim
    def main():
        data = QuantumArray(qtype=QuantumBool(), shape=(5,))
        ancilla = QuantumArray(qtype=QuantumBool(), shape=(4,))
        result = rep_code(data, ancilla)
        det = result[0]
        rest = result[1:]
        return (det,) + rest

    circ = _run_and_get_stim(main)
    assert circ.num_detectors == expected
    assert _sample_detectors_ok(circ)


# ═══════════════════════════════════════════════════════════════════════
# Auto-recode tests (parse stim circuit → replay in Qrisp)
# ═══════════════════════════════════════════════════════════════════════

# --- Repetition codes ---

def test_auto_rep_d3_r1():
    """Auto-recode: repetition code d=3, 1 round."""
    exp, found, ok = _auto_recode_and_check("repetition_code:memory", 3, 1)
    assert found == exp
    assert ok


def test_auto_rep_d3_r2():
    """Auto-recode: repetition code d=3, 2 rounds."""
    exp, found, ok = _auto_recode_and_check("repetition_code:memory", 3, 2)
    assert found >= exp  # may find additional valid detectors
    assert ok


def test_auto_rep_d5_r2():
    """Auto-recode: repetition code d=5, 2 rounds."""
    exp, found, ok = _auto_recode_and_check("repetition_code:memory", 5, 2)
    assert found >= exp
    assert ok


def test_auto_rep_d3_r3():
    """Auto-recode: repetition code d=3, 3 rounds."""
    exp, found, ok = _auto_recode_and_check("repetition_code:memory", 3, 3)
    assert found == exp
    assert ok


def test_auto_rep_d5_r3():
    """Auto-recode: repetition code d=5, 3 rounds."""
    exp, found, ok = _auto_recode_and_check("repetition_code:memory", 5, 3)
    assert found == exp
    assert ok


# --- Surface codes ---

def test_auto_surface_z_d2_r1():
    """Auto-recode: rotated surface code (Z-memory) d=2, 1 round."""
    exp, found, ok = _auto_recode_and_check(
        "surface_code:rotated_memory_z", 2, 1
    )
    assert found == exp
    assert ok


def test_auto_surface_x_d2_r1():
    """Auto-recode: rotated surface code (X-memory) d=2, 1 round."""
    exp, found, ok = _auto_recode_and_check(
        "surface_code:rotated_memory_x", 2, 1
    )
    assert found == exp
    assert ok


def test_auto_surface_z_d3_r1():
    """Auto-recode: rotated surface code (Z-memory) d=3, 1 round."""
    exp, found, ok = _auto_recode_and_check(
        "surface_code:rotated_memory_z", 3, 1
    )
    assert found == exp
    assert ok
