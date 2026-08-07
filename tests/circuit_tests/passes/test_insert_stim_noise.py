"""********************************************************************************
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

from __future__ import annotations

from collections import defaultdict

import pytest

from qrisp import PassManager, layerize, promote_barriers
from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.pass_management.passes.insert_stim_noise import insert_stim_noise
from qrisp.circuit.pass_management.scheduling import asap_layers, is_error_channel, is_transparent
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.misc.stim_tools.error_class import StimNoiseGate

P1 = 0.001
P2 = 0.01
PX = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noise_count_per_qubit(noisy_qc: QuantumCircuit) -> dict[object, int]:
    """Number of noise instructions acting on each qubit of *noisy_qc*."""
    counts: dict[object, int] = defaultdict(int)
    for instr in noisy_qc.data:
        if isinstance(instr.op, StimNoiseGate):
            for q in instr.qubits:
                counts[q] += 1
    return counts


def _time_steps(qc: QuantumCircuit) -> set[int]:
    """The layers of *qc* that are physical time steps.

    A layer holding nothing but barriers or bookkeeping is a scheduling
    artefact; a layer holding an error channel is a time step, since an idle
    error is something happening.  This is the same predicate
    :func:`~qrisp.layerize` uses to decide where to put a ``TICK``.
    """
    return {
        layer
        for layer, instr in zip(asap_layers(qc), qc.data, strict=True)
        if not is_transparent(instr) or is_error_channel(instr)
    }


def _assert_model_verifies(orig_qc: QuantumCircuit, p1=P1, p2=P2, px=PX):
    """Assert the inserted model survives being re-layerized.

    This is the automated form of the visual check: insert noise, then schedule
    the result with the very scheduler the pass used, and require that

    * the noise did not change the number of time steps, and
    * every qubit carries exactly one error channel in every one of them.

    Both would break if the pass counted layers differently from
    :func:`~qrisp.layerize` — which is exactly what a private scheduler in the
    pass used to cause for circuits containing partial barriers.
    """
    noisy = insert_stim_noise(p1, p2, px)(orig_qc)

    steps_in = _time_steps(orig_qc)
    steps_out = _time_steps(noisy)
    assert len(steps_in) == len(steps_out), (
        f"noise insertion changed the number of time steps: {len(steps_in)} -> {len(steps_out)}"
    )

    per_slot: dict[tuple[object, int], int] = defaultdict(int)
    for layer, instr in zip(asap_layers(noisy), noisy.data, strict=True):
        if is_error_channel(instr):
            for q in instr.qubits:
                per_slot[(q, layer)] += 1

    for q in noisy.qubits:
        for layer in sorted(steps_out):
            count = per_slot[(q, layer)]
            assert count == 1, (
                f"qubit {noisy.qubits.index(q)} carries {count} channels in time step {layer}, expected 1"
            )

    return noisy


def _assert_one_noise_per_layer(orig_qc: QuantumCircuit, noisy_qc: QuantumCircuit, num_layers: int):
    """Assert every qubit received exactly one noise instruction per layer.

    *num_layers* is the expected number of physical time steps of *orig_qc*
    and is spelled out by each test, so this check does not depend on the
    pass' own scheduling.  Where the individual noise instructions end up is
    covered by the adjacency tests (channel directly after its gate,
    ``X_ERROR`` directly before its measurement).
    """
    counts = _noise_count_per_qubit(noisy_qc)
    for q in orig_qc.qubits:
        assert counts[q] == num_layers, f"{q}: {counts[q]} noise instructions, expected {num_layers}"


def _op_names(qc: QuantumCircuit) -> list[str]:
    """Names of all non-noise instructions, in circuit order."""
    return [i.op.name for i in qc.data if not isinstance(i.op, StimNoiseGate)]


# ---------------------------------------------------------------------------
# Factory behaviour
# ---------------------------------------------------------------------------


def test_factory_returns_circuit_pass():
    pass_obj = insert_stim_noise(P1, P2, PX)
    assert isinstance(pass_obj, CircuitPass)

    qc = QuantumCircuit(2)
    qc.h(0)
    result = pass_obj(qc)
    assert isinstance(result, QuantumCircuit)


def test_invalid_strengths_raise():
    with pytest.raises(ValueError):
        insert_stim_noise(-0.1, P2, PX)
    with pytest.raises(ValueError):
        insert_stim_noise(P1, 1.5, PX)
    with pytest.raises(ValueError):
        insert_stim_noise(P1, P2, 2.0)


# ---------------------------------------------------------------------------
# Noise insertion rules
# ---------------------------------------------------------------------------


def test_one_qubit_gate_gets_depolarize1():
    qc = QuantumCircuit(1)
    qc.h(0)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert noisy.data[0].op.name == "h"
    assert isinstance(noisy.data[1].op, StimNoiseGate)
    assert noisy.data[1].op.stim_name == "DEPOLARIZE1"
    assert list(noisy.data[1].op.params) == [P1]


def test_two_qubit_gate_gets_depolarize2():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert noisy.data[0].op.name == "cx"
    assert isinstance(noisy.data[1].op, StimNoiseGate)
    assert noisy.data[1].op.stim_name == "DEPOLARIZE2"
    assert list(noisy.data[1].op.params) == [P2]
    assert set(noisy.data[1].qubits) == set(qc.qubits)


def test_multi_qubit_gate_raises():
    from qrisp.circuit import Operation

    qc = QuantumCircuit(3)
    qc.h(0)
    # A gate acting on more than two qubits cannot be annotated by the
    # noise model and must raise.
    qc.append(Operation("ccx", num_qubits=3), qc.qubits)

    with pytest.raises(ValueError, match="acts on 3 qubits"):
        insert_stim_noise(P1, P2, PX)(qc)


def test_idle_qubits_get_depolarize1():
    # h on qubit 0 only; qubit 1 idles in the same layer
    qc = QuantumCircuit(2)
    qc.h(0)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    noise = [i for i in noisy.data if isinstance(i.op, StimNoiseGate)]
    # h(0) -> DEPOLARIZE1 on 0, idle -> DEPOLARIZE1 on 1
    assert len(noise) == 2
    assert all(i.op.stim_name == "DEPOLARIZE1" for i in noise)
    assert set(n.qubits[0] for n in noise) == set(qc.qubits)


def test_measurement_gets_x_error_before():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure(0)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    data = noisy.data
    for i, instr in enumerate(data):
        if instr.op.name == "measure":
            assert isinstance(data[i - 1].op, StimNoiseGate)
            assert data[i - 1].op.stim_name == "X_ERROR"
            assert list(data[i - 1].op.params) == [PX]


def test_reset_gets_x_error_after():
    qc = QuantumCircuit(1)
    qc.reset(0)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    data = noisy.data
    for i, instr in enumerate(data):
        if instr.op.name == "reset":
            assert isinstance(data[i + 1].op, StimNoiseGate)
            assert data[i + 1].op.stim_name == "X_ERROR"
            assert list(data[i + 1].op.params) == [PX]


def test_every_qubit_exactly_one_noise_per_layer():
    # h(0), h(2) in layer 0; cx(0,1), cx(2,3) in layer 1; measure(1), reset(3)
    # in layer 2.
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(2)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.measure(1)
    qc.reset(3)

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    _assert_one_noise_per_layer(qc, noisy, 3)


def test_gates_preserved():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    gate_names = [i.op.name for i in noisy.data if not isinstance(i.op, StimNoiseGate)]
    assert gate_names == ["h", "cx", "cx"]


def test_instruction_order_preserved():
    # The pass schedules into layers to decide *what* noise to insert, but it
    # must not reorder the circuit: the measurement order (and hence the Stim
    # measurement record) has to stay intact.
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.measure(0)
    qc.measure(1)
    qc.cx(1, 2)
    qc.measure(2)

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert _op_names(noisy) == _op_names(qc)

    _, orig_map = qc.to_stim(return_measurement_map=True)
    _, noisy_map = noisy.to_stim(return_measurement_map=True)
    assert noisy_map == orig_map


def test_input_not_mutated():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    before = list(qc.data)

    insert_stim_noise(P1, P2, PX)(qc)
    assert qc.data == before


# ---------------------------------------------------------------------------
# Instructions that carry no time step: no noise, no noise layer
# ---------------------------------------------------------------------------


def test_parity_is_no_time_step():
    # A parity annotation becomes a Stim DETECTOR. It is classical
    # post-processing and must not create a noise layer of its own.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure(0)
    qc.measure(1)
    qc.parity([qc.clbits[0], qc.clbits[1]])

    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert any(i.op.name == "parity" for i in noisy.data)
    # Two layers: {h(0), measure(1)} and {measure(0)}. The parity rides along
    # with the layer of the measurements it refers to.
    _assert_one_noise_per_layer(qc, noisy, 2)


def test_gphase_is_no_time_step():
    # gphase is a one-qubit Operation in Qrisp, but it only records a global
    # phase, so it neither receives noise nor creates a noise layer.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.gphase(0.5, 0)

    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert any(i.op.name == "gphase" for i in noisy.data)
    _assert_one_noise_per_layer(qc, noisy, 1)


def test_gphase_only_circuit_gets_no_noise():
    qc = QuantumCircuit(2)
    qc.gphase(0.5, 0)

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert not any(isinstance(i.op, StimNoiseGate) for i in noisy.data)


def test_qubit_bookkeeping_is_no_time_step():
    from qrisp.circuit import QubitAlloc, QubitDealloc

    qc = QuantumCircuit(2)
    qc.append(QubitAlloc(), [1])
    qc.h(0)
    qc.cx(0, 1)
    qc.append(QubitDealloc(), [1])

    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert _op_names(noisy) == ["qb_alloc", "h", "cx", "qb_dealloc"]
    # Two layers: h(0) and cx(0, 1). The alloc/dealloc markers add none.
    _assert_one_noise_per_layer(qc, noisy, 2)


# ---------------------------------------------------------------------------
# Barriers: synchronization points between layers, not noise sources
# ---------------------------------------------------------------------------


def _depolarize1_count(qc: QuantumCircuit) -> int:
    return sum(1 for i in qc.data if isinstance(i.op, StimNoiseGate) and i.op.stim_name == "DEPOLARIZE1")


def test_barrier_passed_through_no_noise():
    # A trailing barrier must be preserved but must not generate idle noise.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert any(i.op.name == "barrier" for i in noisy.data)
    # Only the h-layer produces noise: DEPOLARIZE1 after h(0) + idle qubit 1.
    assert _depolarize1_count(noisy) == 2
    # The barrier layer is not a time step, so there is exactly one layer.
    _assert_one_noise_per_layer(qc, noisy, 1)


def test_barrier_enforces_ordering():
    # Two independent gates on disjoint qubits separated by a barrier must be
    # scheduled into separate layers, so each qubit idles (and receives idle
    # noise) in the other's layer.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.h(1)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    _assert_one_noise_per_layer(qc, noisy, 2)


def test_barrier_between_gates_same_qubit():
    # A barrier between two gates on the same qubit must not add an extra
    # noise layer: each gate layer contributes exactly one DEPOLARIZE1.
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.barrier()
    qc.h(0)
    noisy = insert_stim_noise(P1, P2, PX)(qc)

    assert _depolarize1_count(noisy) == 2


def test_full_width_barrier_flushes_its_own_layer():
    # A barrier sits *in* the layer it ends, so the idle noise for that layer
    # has to be emitted before it -- otherwise a channel belonging to the time
    # step ends up on the far side of the TICK that closes it.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    names = [i.op.name for i in noisy.data]
    assert names.index("barrier") == len(names) - 1, f"noise emitted after the barrier: {names}"


def test_partial_barrier_does_not_synchronize_unnamed_qubits():
    # A partial barrier constrains only the qubits it names, so q1's
    # measurement stays in layer 0 and the circuit has three time steps, not
    # four.  Flushing q1 at the barrier would emit its idle noise ahead of its
    # own X_ERROR/measurement and invent an extra round of noise.
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.h(0)
    qc.h(0)
    qc.barrier([qc.qubits[0]])
    qc.measure(1, qc.clbits[0])

    noisy = _assert_model_verifies(qc)

    # q1 is measured in layer 0 and idles through layers 1 and 2: one X_ERROR
    # plus two DEPOLARIZE1, and the X_ERROR comes first.
    q1_noise = [i.op.stim_name for i in noisy.data if is_error_channel(i) and qc.qubits[1] in i.qubits]
    assert q1_noise == ["X_ERROR", "DEPOLARIZE1", "DEPOLARIZE1"]


def test_partial_barrier_still_fences_the_qubits_it_names():
    # The named qubit's outstanding noise must not cross its own fence.  A
    # partial barrier floats to the earliest layer its named qubits allow, so
    # this one lands in layer 0 -- the very layer q1 idles through -- and has to
    # take q1's idle channel with it rather than leave it for the h(1) after.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)
    qc.barrier([qc.qubits[1]])
    qc.h(1)

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    names = [i.op.name for i in noisy.data]
    barrier_at = names.index("barrier")
    q1_before = sum(1 for i in noisy.data[:barrier_at] if is_error_channel(i) and qc.qubits[1] in i.qubits)
    assert q1_before == 1, f"idle noise crossed the fence: {names}"

    # q0 is not named, so its layers are untouched by the fence and it keeps one
    # channel per gate.  q1 idles in layer 0 and gates in layer 1.
    counts = _noise_count_per_qubit(noisy)
    assert counts[qc.qubits[0]] == 2
    assert counts[qc.qubits[1]] == 2


def test_promote_barriers_restores_global_synchronization():
    # Widening the fences is the documented way to get the old, globally
    # synchronizing behaviour back -- at the price of a wider schedule.
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.h(0)
    qc.h(0)
    qc.barrier([qc.qubits[0]])
    qc.measure(1, qc.clbits[0])

    assert len(_time_steps(qc)) == 3
    promoted = promote_barriers(qc)
    assert len(_time_steps(promoted)) == 4

    noisy = _assert_model_verifies(promoted)
    # Now the measurement is pushed past the barrier, so q1 idles through all
    # three h-layers before being measured.
    q1_noise = [i.op.stim_name for i in noisy.data if is_error_channel(i) and qc.qubits[1] in i.qubits]
    assert q1_noise == ["DEPOLARIZE1", "DEPOLARIZE1", "DEPOLARIZE1", "X_ERROR"]


# ---------------------------------------------------------------------------
# The model survives being re-layerized -- the automated form of the visual
# check that the pipeline insert_stim_noise -> layerize is built for.
# ---------------------------------------------------------------------------


def test_model_verifies_independent_chains():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    _assert_model_verifies(qc)


def test_model_verifies_repetition_code_round():
    qc = QuantumCircuit(3)
    for _ in range(2):
        qc.cx(0, 1)
        qc.cx(2, 1)
        clbit = qc.add_clbit()
        qc.measure(1, clbit)
        qc.parity([clbit])
        qc.reset(1)
        qc.barrier(qc.qubits)

    _assert_model_verifies(qc)


def test_model_verifies_with_ancilla_only_barrier():
    # The realistic partial-barrier case: the syndrome ancilla is fenced per
    # round while the data qubits keep working.
    qc = QuantumCircuit(3)
    for _ in range(2):
        qc.cx(0, 1)
        qc.cx(2, 1)
        clbit = qc.add_clbit()
        qc.measure(1, clbit)
        qc.reset(1)
        qc.barrier([qc.qubits[1]])
        qc.h(0)
        qc.h(2)

    _assert_model_verifies(qc)


def test_layerize_renders_one_tick_per_time_step():
    # The end of the pipeline: one TICK per time step of the input.
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(2, 1)
    qc.measure(1)
    qc.reset(1)

    pm = PassManager()
    pm += insert_stim_noise(P1, P2, PX)
    pm += layerize(insert_barriers=True)

    assert pm.run(qc).to_stim().num_ticks == len(_time_steps(qc)) == 4


def test_passmanager_composition():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for q in qc.qubits:
        qc.measure(q)

    pm = PassManager()
    pm += insert_stim_noise(P1, P2, PX)
    pm += layerize()
    noisy = pm.run(qc.copy())

    assert any(isinstance(i.op, StimNoiseGate) for i in noisy.data)
    # no gate may sit between the X_ERROR and the measurement on the same qubit
    for m_idx, instr in enumerate(noisy.data):
        if instr.op.name != "measure":
            continue
        last_on_qubit = None
        for j in range(m_idx - 1, -1, -1):
            if instr.qubits[0] in noisy.data[j].qubits:
                last_on_qubit = noisy.data[j].op.name
                break
        assert last_on_qubit == "stim.X_ERROR"


# ---------------------------------------------------------------------------
# only_necessary: respect user-placed noise, do not duplicate it
# ---------------------------------------------------------------------------


def _noise_names(qc: QuantumCircuit) -> list[str]:
    return [i.op.stim_name for i in qc.data if isinstance(i.op, StimNoiseGate) and i.op.stim_name != "X_ERROR"]


def test_only_necessary_skips_after_two_qubit_gate():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), qc.qubits)  # user-placed noise

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    # Exactly one DEPOLARIZE2 (the user's) — no duplicate after the CX.
    assert _noise_names(noisy).count("DEPOLARIZE2") == 1
    # It must be the user's strength, and directly after the CX.
    assert noisy.data[1].op.stim_name == "DEPOLARIZE2"
    assert list(noisy.data[1].op.params) == [0.05]


def test_only_necessary_skips_after_one_qubit_gate():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.append(StimNoiseGate("DEPOLARIZE1", 0.05), [qc.qubits[0]])

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert _noise_names(noisy).count("DEPOLARIZE1") == 1
    assert list(noisy.data[1].op.params) == [0.05]


def test_only_necessary_skips_before_measurement():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.append(StimNoiseGate("X_ERROR", 0.05), [qc.qubits[0]])
    qc.measure(0)

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    x_errors = [i for i in noisy.data if isinstance(i.op, StimNoiseGate) and i.op.stim_name == "X_ERROR"]
    # Only the user's X_ERROR before the measurement.
    assert len(x_errors) == 1
    assert list(x_errors[0].op.params) == [0.05]
    # The X_ERROR annotates the measurement only. It must not also be counted
    # as the depolarizing channel of the preceding h, which still needs one.
    assert _noise_names(noisy).count("DEPOLARIZE1") == 1
    _assert_one_noise_per_layer(qc, noisy, 2)


def test_only_necessary_requires_matching_channel():
    # A DEPOLARIZE1 after a two-qubit gate is not the channel the noise model
    # would insert, so the DEPOLARIZE2 is inserted regardless.
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.append(StimNoiseGate("DEPOLARIZE1", 0.05), [qc.qubits[0]])

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert _noise_names(noisy).count("DEPOLARIZE2") == 1


def test_only_necessary_requires_matching_qubits():
    # The user's DEPOLARIZE2 covers only two of the three qubits of the
    # (decomposed) gate pair, so it does not match the expected channel.
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.append(StimNoiseGate("DEPOLARIZE1", P1), [qc.qubits[1]])

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert _noise_names(noisy).count("DEPOLARIZE2") == 1


def test_only_necessary_ignores_strength():
    # A matching channel of a different strength is still respected.
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.append(StimNoiseGate("DEPOLARIZE1", 0.5), [qc.qubits[0]])

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert _noise_names(noisy).count("DEPOLARIZE1") == 1
    assert list(noisy.data[1].op.params) == [0.5]


def test_only_necessary_skips_after_reset():
    qc = QuantumCircuit(1)
    qc.reset(0)
    qc.append(StimNoiseGate("X_ERROR", 0.05), [qc.qubits[0]])

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    x_errors = [i for i in noisy.data if isinstance(i.op, StimNoiseGate) and i.op.stim_name == "X_ERROR"]
    assert len(x_errors) == 1
    assert list(x_errors[0].op.params) == [0.05]


def test_only_necessary_false_adds_full_model():
    # With only_necessary=False the pass duplicates the user-placed noise.
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), qc.qubits)

    noisy = insert_stim_noise(P1, P2, PX, only_necessary=False)(qc)
    assert _noise_names(noisy).count("DEPOLARIZE2") == 2


def test_only_necessary_idle_noise_still_added():
    # A user-placed DEPOLARIZE2 after the CX suppresses the duplicate, but an
    # idle third qubit still receives its DEPOLARIZE1 for that layer.
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), qc.qubits[:2])

    noisy = insert_stim_noise(P1, P2, PX)(qc)
    assert _noise_names(noisy).count("DEPOLARIZE1") == 1

    # Exactly one noise per qubit in the single layer.
    _assert_one_noise_per_layer(qc, noisy, 1)


# ---------------------------------------------------------------------------
# Noise gates behave as identity -> the pass preserves unitarity / statistics
# ---------------------------------------------------------------------------


def test_compare_unitary():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pass_obj = insert_stim_noise(P1, P2, PX)
    assert pass_obj.compare_unitary(qc, precision=4)


def test_compare_measurement():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    for q in qc.qubits:
        qc.measure(q)
    pass_obj = insert_stim_noise(P1, P2, PX)
    assert pass_obj.compare_measurement(qc, precision=6)
