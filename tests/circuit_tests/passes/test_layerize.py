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

import pytest

from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management.passes import layerize


@pytest.fixture
def _stim():
    """Skip a test when the optional Stim dependency is unavailable."""
    return pytest.importorskip("stim")


def _gate_names(qc: QuantumCircuit) -> list[str]:
    """Return the ordered list of operation names in the circuit."""
    return [instr.op.name for instr in qc.data]


def _layer_positions(qc: QuantumCircuit) -> dict[str, int]:
    """Return a mapping from ``"op_name@qubit_id"`` → instruction index."""
    result = {}
    for i, instr in enumerate(qc.data):
        for q in instr.qubits:
            key = f"{instr.op.name}@{q.identifier}"
            result[key] = i
    return result


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestLayerizeBasic:
    """Smoke tests — basic invariants."""

    def test_returns_quantum_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = layerize()(qc)
        assert isinstance(result, QuantumCircuit)

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        result = layerize()(qc)
        assert len(result.data) == 0

    def test_instruction_count_preserved(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        result = layerize()(qc)
        assert len(result.data) == len(qc.data)

    def test_gate_multiset_preserved(self):
        """layerize must not add or remove gates, only reorder."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)
        qc.z(0)
        qc.measure(qc.qubits)

        before_names = sorted(instr.op.name for instr in qc.data)
        result = layerize()(qc)
        after_names = sorted(instr.op.name for instr in result.data)

        assert before_names == after_names


# ---------------------------------------------------------------------------
# Reordering behaviour
# ---------------------------------------------------------------------------


class TestLayerizeReordering:
    """Verify that disjoint gates are pulled into earlier layers."""

    def test_independent_single_qubit_gates_compacted(self):
        """h on q2 should move before cx(0,1) since it shares no qubits."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)  # independent
        qc.cx(1, 2)

        result = layerize()(qc)
        names = _gate_names(result)
        # h(2) must appear before cx(0,1)
        assert names.index("h") < names.index("cx"), f"h(2) not before first cx: {names}"

    def test_independent_cx_gates_compacted(self):
        """CX gates on disjoint qubit pairs should move to the same layer."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)  # h(2) and cx(2,3) share qubit 2 → must stay ordered

        result = layerize()(qc)
        names = _gate_names(result)
        # cx(0,1) is independent of h(2) and can move before it
        assert names == ["cx", "h", "cx"], f"Unexpected order: {names}"

    def test_relative_order_on_same_qubit_preserved(self):
        """Operations on the same qubit must stay in original order."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        qc.h(0)

        result = layerize()(qc)
        names = _gate_names(result)
        # On a single qubit nothing can move — must be identical
        assert names == ["h", "x", "z", "h"], f"Single-qubit order changed: {names}"


# ---------------------------------------------------------------------------
# Bookkeeping instructions (qb_alloc / qb_dealloc)
# ---------------------------------------------------------------------------


class TestLayerizeBookkeeping:
    """Verify that qb_alloc / qb_dealloc are excluded from scheduling."""

    def test_qb_alloc_does_not_delay_real_gates(self):
        """Alloc instructions must not push real gates to later layers."""
        from qrisp.circuit.standard_operations import QubitAlloc
        from qrisp.circuit import Qubit

        qc = QuantumCircuit(3)
        # Manually append qb_alloc instructions interleaved with gates
        q0, q1, q2 = qc.qubits

        qc.append(QubitAlloc(), [q0])
        qc.h(0)
        qc.append(QubitAlloc(), [q1])
        qc.cx(0, 1)
        qc.append(QubitAlloc(), [q2])
        qc.h(2)

        result = layerize()(qc)
        names = _gate_names(result)

        # All qb_alloc should be at the very beginning
        alloc_indices = [i for i, n in enumerate(names) if n == "qb_alloc"]
        assert alloc_indices == [0, 1, 2], f"Allocs not at front: {alloc_indices}"

        # Real gates h, cx, h — h(2) is independent of cx(0,1) and moves up
        real_names = [n for n in names if n != "qb_alloc"]
        assert real_names == ["h", "h", "cx"], f"Real gate order broken: {real_names}"


# ---------------------------------------------------------------------------
# Correctness: unitary and measurement statistics
# ---------------------------------------------------------------------------


class TestLayerizeCorrectness:
    """Verify layerize preserves circuit semantics."""

    def test_unitary_preserved_simple(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        assert layerize().compare_unitary(qc)

    def test_unitary_preserved_layered(self):
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)
        assert layerize().compare_unitary(qc)

    def test_measurement_statistics_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        qc.measure(qc.qubits)
        assert layerize().compare_measurement(qc)

    def test_unitary_preserved_with_swaps(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.h(2)
        assert layerize().compare_unitary(qc)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestLayerizeEdgeCases:
    """Corner cases and robustness."""

    def test_idempotent(self):
        """Applying twice should give the same result as once."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i + 1)

        once = layerize()(qc)
        twice = layerize()(once)
        assert _gate_names(once) == _gate_names(twice)

    def test_all_independent_gates_stay_at_front(self):
        """All single-qubit gates on distinct qubits should be at the front."""
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
        qc.cx(0, 1)

        result = layerize()(qc)
        names = _gate_names(result)
        # All four h gates should come before the cx
        last_h = max(i for i, n in enumerate(names) if n == "h")
        first_cx = names.index("cx")
        assert last_h < first_cx, f"h gates not all before cx: {names}"


# ---------------------------------------------------------------------------
# Barrier handling
# ---------------------------------------------------------------------------


class TestLayerizeBarriers:
    """Barriers constrain the qubits they name, and nothing else."""

    def test_partial_barrier_lets_unrelated_gates_pass(self):
        """A gate on a qubit the barrier does not name may move before it."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(1)

        result = layerize()(qc)
        names = _gate_names(result)
        assert names == ["h", "h", "barrier"], f"Unexpected order: {names}"

    def test_full_width_barrier_holds_everything_back(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier()
        qc.h(1)

        result = layerize()(qc)
        assert _gate_names(result) == ["h", "barrier", "h"]

    def test_gates_do_not_cross_a_barrier_on_their_own_qubit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.x(0)

        names = _gate_names(layerize()(qc))
        assert names.index("h") < names.index("barrier") < names.index("x")

    def test_barrier_count_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.barrier()
        qc.h(1)

        result = layerize()(qc)
        assert _gate_names(result).count("barrier") == 2


# ---------------------------------------------------------------------------
# Parallel rendering within a layer
# ---------------------------------------------------------------------------


class TestLayerizeParallelRendering:
    """A layer must be emitted so that Stim can draw it as parallel columns.

    Stim renders instructions strictly in order and never goes back to fill an
    earlier column, so ``H 0 | X_ERROR 0 | H 5 | X_ERROR 5`` is drawn four
    columns wide even though all four instructions share a layer.
    """

    def test_two_chains_with_error_channels(self, _stim):
        from qrisp.misc.stim_tools import StimNoiseGate

        qc = QuantumCircuit(6)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.h(5)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[5]])
        qc.cx(5, 4)
        qc.cx(4, 3)

        result = layerize()(qc)
        assert _gate_names(result) == [
            "h",
            "h",
            "stim.X_ERROR",
            "stim.X_ERROR",
            "cx",
            "cx",
            "cx",
            "cx",
        ]

        # Both H gates land in one Stim instruction, as do both error channels.
        text = str(result.to_stim())
        assert "H 0 5" in text
        assert "X_ERROR(0.5) 0 5" in text

    def test_layer_packs_into_the_minimum_number_of_columns(self, _stim):
        """A tick must need as many columns as a qubit is used, and no more."""
        from qrisp.misc.stim_tools import StimNoiseGate

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.h(2)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[2]])
        qc.cx(0, 1)
        qc.cx(2, 3)

        result = layerize(insert_barriers=True)(qc)
        gates, entanglers = _moments(result.to_stim())

        # First tick: every qubit is used twice (gate + channel), so Stim needs
        # two columns - one fused H and one fused X_ERROR, not four instructions.
        assert gates == [("H", (0, 2)), ("X_ERROR", (0, 2))]
        # Second tick: both CX gates are disjoint, so a single column suffices.
        assert entanglers == [("CX", (0, 1, 2, 3))]

    def test_ordering_within_a_layer_is_semantics_preserving(self, _stim):
        from qrisp.misc.stim_tools import StimNoiseGate

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(2, 3)

        # The channel must stay after the h it annotates and before the cx.
        names = _gate_names(layerize()(qc))
        assert names.index("h") < names.index("stim.X_ERROR") < names.index("cx")

    def test_measurement_statistics_preserved(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.z(0)
        qc.measure(qc.qubits)
        assert layerize().compare_measurement(qc)


# ---------------------------------------------------------------------------
# insert_barriers
# ---------------------------------------------------------------------------


def _moments(stim_circuit) -> list[list[tuple[str, tuple[int, ...]]]]:
    """Split a Stim circuit at its TICKs into ``(gate name, qubit targets)`` lists.

    Stim fuses consecutive identical instructions (``H 0`` then ``H 5`` becomes
    ``H 0 5``), so tick contents have to be inspected through the instruction
    API rather than by matching substrings of the circuit text.
    """
    moments: list[list[tuple[str, tuple[int, ...]]]] = [[]]
    for inst in stim_circuit.flattened():
        if inst.name == "TICK":
            moments.append([])
            continue
        moments[-1].append((inst.name, tuple(t.qubit_value for t in inst.targets_copy())))
    return [moment for moment in moments if moment]


def _num_layers(qc: QuantumCircuit) -> int:
    """Number of distinct layers holding at least one real instruction."""
    from qrisp.circuit.pass_management.scheduling import asap_layers, is_transparent

    return len({layer for layer, instr in zip(asap_layers(qc), qc.data, strict=True) if not is_transparent(instr)})


class TestLayerizeInsertBarriers:
    """Writing the schedule back into the circuit as a TICK stream."""

    def test_off_by_default(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert "barrier" not in _gate_names(layerize()(qc))

    def test_one_barrier_per_layer(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(2)
        qc.cx(0, 1)
        expected_layers = _num_layers(qc)

        result = layerize(insert_barriers=True)(qc)
        assert _gate_names(result) == ["h", "h", "barrier", "cx", "barrier"]
        assert _gate_names(result).count("barrier") == expected_layers == 2

    def test_one_tick_per_layer(self, _stim):
        """The canonical Stim shape: exactly one time-step boundary per layer."""
        qc = QuantumCircuit(3, 1)
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.measure(qc.qubits[1], qc.clbits[0])
        expected_layers = _num_layers(qc)

        stim_circuit = layerize(insert_barriers=True)(qc).to_stim()
        assert stim_circuit.num_ticks == expected_layers == 3

    def test_no_doubled_ticks(self, _stim):
        """A doubled TICK is an empty moment - a blank column in the diagram."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(1, 2)

        stim_circuit = layerize(insert_barriers=True)(qc).to_stim()
        assert str(stim_circuit).count("TICK\nTICK") == 0

    def test_existing_global_barrier_is_reused(self, _stim):
        """§2.10: a boundary the user already marked must not be marked twice."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(1, 2)
        expected_layers = _num_layers(qc)

        result = layerize(insert_barriers=True)(qc)
        assert _gate_names(result).count("barrier") == expected_layers == 3
        assert result.to_stim().num_ticks == 3

    def test_partial_barrier_does_not_count_as_a_boundary(self):
        """A local fence is not a time boundary, so the layer still gets marked."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.x(0)

        result = layerize(insert_barriers=True)(qc)
        names = _gate_names(result)
        # Two layers -> two full-width barriers, plus the user's partial one.
        assert names.count("barrier") == 3

    def test_idempotent(self, _stim):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(1, 2)

        once = layerize(insert_barriers=True)(qc)
        twice = layerize(insert_barriers=True)(once)
        assert _gate_names(once) == _gate_names(twice)
        assert once.to_stim().num_ticks == twice.to_stim().num_ticks

    def test_gate_and_its_error_channel_share_a_tick(self, _stim):
        """Error channels must be visualised in the same tick as their gate.

        This is what makes it possible to read off, per tick, that every gate
        and every qubit carries its corresponding error instruction.
        """
        from qrisp.misc.stim_tools import StimNoiseGate

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[2]])
        qc.cx(0, 1)
        qc.append(StimNoiseGate("DEPOLARIZE2", 0.01), [qc.qubits[0], qc.qubits[1]])

        result = layerize(insert_barriers=True)(qc)

        # No barrier may come between a gate and the channel annotating it, and
        # the idle channel on q2 stays in the layer it was written into.
        names = _gate_names(result)
        assert names == [
            "h",
            "stim.DEPOLARIZE1",
            "stim.DEPOLARIZE1",
            "barrier",
            "cx",
            "stim.DEPOLARIZE2",
            "barrier",
        ]

        first, second = _moments(result.to_stim())
        assert ("H", (0,)) in first
        assert any(name == "DEPOLARIZE1" and 0 in targets for name, targets in first)
        assert any(name == "DEPOLARIZE1" and 2 in targets for name, targets in first)
        assert ("CX", (0, 1)) in second
        assert any(name == "DEPOLARIZE2" and set(targets) == {0, 1} for name, targets in second)

    def test_error_channels_do_not_advance_the_clock(self, _stim):
        """A run of error channels must not each earn a tick of its own."""
        from qrisp.misc.stim_tools import StimNoiseGate

        qc = QuantumCircuit(1)
        qc.h(0)
        for _ in range(3):
            qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])

        result = layerize(insert_barriers=True)(qc)
        assert _gate_names(result).count("barrier") == 1
        assert result.to_stim().num_ticks == 1

    def test_bookkeeping_does_not_create_a_layer(self, _stim):
        """qb_alloc / qb_dealloc must not earn a TICK of their own."""
        from qrisp.circuit.standard_operations import QubitAlloc, QubitDealloc

        qc = QuantumCircuit(2)
        qc.append(QubitAlloc(), [qc.qubits[0]])
        qc.h(0)
        qc.append(QubitAlloc(), [qc.qubits[1]])
        qc.cx(0, 1)
        qc.append(QubitDealloc(), [qc.qubits[0]])

        result = layerize(insert_barriers=True)(qc)
        assert _gate_names(result).count("barrier") == _num_layers(qc) == 2
        assert result.to_stim().num_ticks == 2

    def test_empty_circuit_gets_no_barrier(self):
        result = layerize(insert_barriers=True)(QuantumCircuit(3))
        assert len(result.data) == 0

    def test_only_bookkeeping_gets_no_barrier(self):
        from qrisp.circuit.standard_operations import QubitAlloc

        qc = QuantumCircuit(2)
        qc.append(QubitAlloc(), [qc.qubits[0]])
        result = layerize(insert_barriers=True)(qc)
        assert _gate_names(result) == ["qb_alloc"]

    def test_measurement_statistics_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(1, 2)
        qc.measure(qc.qubits)
        assert layerize(insert_barriers=True).compare_measurement(qc)


# ---------------------------------------------------------------------------
# End-to-end: detector semantics must survive reordering
# ---------------------------------------------------------------------------


def _rep_code_d3(rounds: int) -> QuantumCircuit:
    """A d=3 repetition-code memory experiment with parity detectors."""
    # 3 data qubits (0, 2, 4) and 2 ancillas (1, 3); one clbit per measurement.
    qc = QuantumCircuit(5, 2 * rounds + 3)
    data = [qc.qubits[0], qc.qubits[2], qc.qubits[4]]
    ancilla = [qc.qubits[1], qc.qubits[3]]

    previous: list = []
    clbit = 0
    for _ in range(rounds):
        for anc in ancilla:
            qc.reset(anc)
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[1])
        qc.cx(data[1], ancilla[0])
        qc.cx(data[2], ancilla[1])

        current = []
        for anc in ancilla:
            qc.measure(anc, qc.clbits[clbit])
            current.append(qc.clbits[clbit])
            clbit += 1

        for i, cb in enumerate(current):
            # Compare against the same ancilla in the previous round.
            qc.parity([cb, previous[i]] if previous else [cb])
        previous = current

    # Final destructive readout of the data qubits, checked against the last
    # round of ancilla measurements.
    readout = []
    for qb in data:
        qc.measure(qb, qc.clbits[clbit])
        readout.append(qc.clbits[clbit])
        clbit += 1

    qc.parity([readout[0], readout[1], previous[0]])
    qc.parity([readout[1], readout[2], previous[1]])
    qc.parity([readout[0]], observable=True)
    return qc


class TestLayerizeDetectorSemantics:
    """Reordering must not disturb detectors, observables or the record order."""

    @staticmethod
    def _detection_events(qc: QuantumCircuit) -> int:
        sampler = qc.to_stim().compile_detector_sampler()
        return int(sampler.sample(shots=64).sum())

    def test_noiseless_rep_code_has_no_detection_events(self, _stim):
        qc = _rep_code_d3(rounds=2)
        assert self._detection_events(qc) == 0

    def test_layerize_preserves_detector_count(self, _stim):
        qc = _rep_code_d3(rounds=2)
        expected = qc.to_stim().num_detectors
        assert layerize()(qc).to_stim().num_detectors == expected
        assert layerize(insert_barriers=True)(qc).to_stim().num_detectors == expected

    def test_layerize_keeps_detectors_silent(self, _stim):
        """The load-bearing assertion: a noiseless code must stay noiseless."""
        qc = _rep_code_d3(rounds=2)
        assert self._detection_events(layerize()(qc)) == 0
        assert self._detection_events(layerize(insert_barriers=True)(qc)) == 0

    def test_insert_barriers_gives_one_tick_per_layer_on_a_real_code(self, _stim):
        qc = _rep_code_d3(rounds=2)
        result = layerize(insert_barriers=True)(qc)
        stim_circuit = result.to_stim()
        assert stim_circuit.num_ticks == _num_layers(qc)
        assert str(stim_circuit).count("TICK\nTICK") == 0
