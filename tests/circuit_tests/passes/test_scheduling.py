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
from qrisp.circuit.pass_management.scheduling import (
    asap_layers,
    asap_schedule,
    is_error_channel,
    is_transparent,
)
from qrisp.circuit.quantum_circuit import is_full_width_barrier
from qrisp.circuit.standard_operations import QubitAlloc, QubitDealloc

stim = pytest.importorskip("stim")

from qrisp.misc.stim_tools import StimNoiseGate  # noqa: E402  (needs the importorskip above)


# ---------------------------------------------------------------------------
# asap_layers
# ---------------------------------------------------------------------------


class TestAsapLayers:
    """The layer assignment itself."""

    def test_one_layer_per_instruction(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert len(asap_layers(qc)) == len(qc.data)

    def test_empty_circuit(self):
        assert asap_layers(QuantumCircuit(3)) == []

    def test_chain_is_serial(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        assert asap_layers(qc) == [0, 1, 2]

    def test_independent_chains_start_at_zero(self):
        """Layer indices are not monotone along qc.data - both chains start at 0."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)
        assert asap_layers(qc) == [0, 1, 0, 1]

    def test_clbits_count_as_resources(self):
        """A measurement and a later write to the same clbit cannot share a layer."""
        qc = QuantumCircuit(2, 1)
        qc.measure(qc.qubits[0], qc.clbits[0])
        qc.measure(qc.qubits[1], qc.clbits[0])
        assert asap_layers(qc) == [0, 1]


class TestAsapLayersTransparency:
    """Instructions that must not advance the layer clock."""

    def test_alloc_dealloc_are_transparent(self):
        qc = QuantumCircuit(2)
        qc.append(QubitAlloc(), [qc.qubits[0]])
        qc.h(0)
        qc.append(QubitAlloc(), [qc.qubits[1]])
        qc.cx(0, 1)
        qc.append(QubitDealloc(), [qc.qubits[0]])
        # The allocs ride along in the layer of the gate they precede and the
        # dealloc in the layer of the cx; none of them adds a layer of its own.
        assert asap_layers(qc) == [0, 0, 0, 1, 1]

    def test_gphase_is_transparent(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.gphase(0.5, 0)
        qc.h(0)
        assert asap_layers(qc) == [0, 0, 1]

    def test_parity_is_transparent(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(qc.qubits[0], qc.clbits[0])
        qc.parity([qc.clbits[0]])
        assert asap_layers(qc) == [0, 0]

    def test_noise_gate_rides_with_its_gate(self):
        """A StimNoiseGate must stay in the layer of the gate it annotates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.cx(0, 1)
        qc.append(StimNoiseGate("DEPOLARIZE2", 0.01), qc.qubits)
        assert asap_layers(qc) == [0, 0, 1, 1]

    def test_channels_on_disjoint_qubits_share_a_layer(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
        assert asap_layers(qc) == [0, 0, 0, 0]


class TestLayerBudget:
    """Per qubit per layer: one gate, one error channel, any other annotations."""

    def test_one_gate_one_error_and_any_annotations(self):
        from qrisp.circuit.standard_operations import QubitAlloc, QubitDealloc

        qc = QuantumCircuit(1)
        qc.append(QubitAlloc(), [qc.qubits[0]])
        qc.gphase(0.5, 0)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.append(QubitDealloc(), [qc.qubits[0]])
        # All of it fits in one layer: one gate, one error, three annotations.
        assert asap_layers(qc) == [0, 0, 0, 0, 0]

    def test_a_second_gate_needs_a_second_layer(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.x(0)
        assert asap_layers(qc) == [0, 1]

    def test_a_second_error_needs_a_second_layer(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.append(StimNoiseGate("Z_ERROR", 0.5), [qc.qubits[0]])
        assert asap_layers(qc) == [0, 0, 1]

    def test_stacked_errors_delay_the_following_gate(self):
        """Each error takes a time step, so the gate behind them moves back.

        Deliberate: five errors in a row are five errors, not one time step.  The
        gate shares the last of those layers, since it needs the gate slot rather
        than the error slot.
        """
        qc = QuantumCircuit(1)
        for _ in range(5):
            qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.x(0)
        assert asap_layers(qc) == [0, 1, 2, 3, 4, 4]

    def test_a_single_error_does_not_delay_the_following_gate(self):
        """The realistic case: idle noise beside a gate in the same time step."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[1]])
        qc.x(1)
        assert asap_layers(qc) == [0, 0, 0]


class TestAsapLayersOneErrorPerLayer:
    """A qubit carries at most one error channel per time step."""

    def test_two_idle_channels_occupy_two_layers(self):
        """A second error on a qubit is a second error, so it is a second layer."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[1]])
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[1]])
        assert asap_layers(qc) == [0, 0, 1]

    def test_gate_noise_and_readout_noise_split(self):
        """Gate noise takes the layer's slot, so readout noise moves on to the
        measurement's layer without needing to be recognised as readout noise.
        """
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.001), [qc.qubits[0]])
        qc.append(StimNoiseGate("X_ERROR", 0.005), [qc.qubits[0]])
        qc.measure(qc.qubits[0], qc.clbits[0])
        assert asap_layers(qc) == [0, 0, 1, 1]

    def test_channels_pile_up_across_layers(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        for _ in range(4):
            qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        assert asap_layers(qc) == [0, 0, 1, 2, 3]

    def test_a_two_qubit_channel_needs_both_slots_free(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
        qc.append(StimNoiseGate("DEPOLARIZE2", 0.01), qc.qubits)
        # q1's slot in layer 0 is taken, so the two-qubit channel moves on even
        # though q0's slot is still free.
        assert asap_layers(qc) == [0, 0, 1]

    def test_pushed_channel_does_not_overtake_the_next_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.append(StimNoiseGate("Z_ERROR", 0.5), [qc.qubits[0]])
        qc.x(0)

        names = [instr.op.name for instr in layerize()(qc).data]
        assert names == ["h", "stim.X_ERROR", "stim.Z_ERROR", "x"]


class TestAsapLayersErrorChannels:
    """An error channel belongs to the time step it was written into."""

    def test_idle_channel_stays_in_its_layer(self):
        """A channel on a qubit with no gate in this layer must not drift back."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
        # Plain transparency would put the q1 channel at -1, since q1 has no gate.
        assert asap_layers(qc) == [0, 0, 0]

    def test_idle_channel_in_a_later_layer(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.append(StimNoiseGate("DEPOLARIZE2", 0.01), qc.qubits)
        qc.x(0)
        qc.append(StimNoiseGate("X_ERROR", 0.01), [qc.qubits[0]])
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
        # h, h | cx, DEP2 | x, X_ERROR, and q1's idle channel joins layer 2.
        assert asap_layers(qc) == [0, 0, 1, 1, 2, 2, 2]

    def test_channel_never_overtakes_a_gate_on_its_own_qubit(self):
        """Data order is not layer-monotone, so a channel placed in the layer it
        was written into must not end up emitted after a gate it was written in
        front of.  It may *share* that gate's layer - the fence pulls the gate up
        rather than holding the channel back - as long as the order survives.
        """
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(2)
        qc.x(2)
        qc.z(2)  # q2 chain reaches layer 2 while q0 is still at layer 0
        qc.append(StimNoiseGate("X_ERROR", 0.01), [qc.qubits[0]])
        qc.x(0)

        # Both the channel and the gate it precedes act on q0; the circuit also
        # contains an x on q2, so match on the qubit rather than the name alone.
        q0 = qc.qubits[0]
        out = layerize()(qc).data
        channel = next(i for i, instr in enumerate(out) if "X_ERROR" in instr.op.name)
        gate = next(i for i, instr in enumerate(out) if instr.op.name == "x" and q0 in instr.qubits)
        assert channel < gate, f"channel overtook its own gate: {[i.op.name for i in out]}"

    def test_channel_cannot_be_pushed_across_a_barrier(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.barrier()
        qc.h(1)

        names = [instr.op.name for instr in layerize()(qc).data]
        assert names.index("stim.DEPOLARIZE1") < names.index("barrier"), names

    def test_bookkeeping_is_not_rationed(self):
        """Only error channels are one-per-layer; allocation markers are free."""
        qc = QuantumCircuit(2)
        qc.append(QubitAlloc(), [qc.qubits[0]])
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        # The alloc must not consume q0's error slot, or the gate noise would be
        # pushed out of the layer of the gate it annotates.
        assert asap_layers(qc) == [0, 0, 0]


class TestAsapLayersBarriers:
    """Barriers constrain exactly the qubits they name."""

    def test_partial_barrier_does_not_delay_other_qubits(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(0)
        qc.h(1)
        # h(1) stays in layer 0: the barrier never named its qubit.
        assert asap_layers(qc) == [0, 0, 1, 0]

    def test_full_width_barrier_synchronises_everything(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.h(1)
        # h(1) is pushed past the barrier because the barrier names q1 too.
        assert asap_layers(qc) == [0, 0, 1]

    def test_barrier_closes_the_layer_it_rides_in(self):
        """A barrier occupies no time step: it is reported in the layer it ends."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.barrier()
        qc.h(0)
        assert asap_layers(qc) == [0, 0, 1]

    def test_barrier_blocks_two_independent_chains_separately(self):
        """§2.8 of the barrier/TICK design note: a fence on chain A only."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier([qc.qubits[0], qc.qubits[1]])
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)
        qc.h(2)
        qc.cx(2, 3)
        # Chain B is untouched by the fence, so the circuit is 4 layers deep -
        # the physically correct answer.  A global fence would give 6.
        assert len(set(asap_layers(qc))) == 4

    def test_promoting_the_fence_costs_two_layers(self):
        """Same circuit as above, but with the barrier widened to full width."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(2, 3)
        qc.h(2)
        qc.cx(2, 3)
        assert len(set(asap_layers(qc))) == 6

    def test_repeated_barriers_do_not_pile_up(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.barrier()
        qc.barrier()
        qc.h(0)
        # The second barrier finds no new instructions to fence off.
        assert asap_layers(qc) == [0, 0, 0, 1]

    def test_noise_gate_cannot_drift_back_across_a_barrier(self):
        """A transparent instruction still respects a fence on its own qubit."""
        qc = QuantumCircuit(1)
        qc.barrier()
        qc.append(StimNoiseGate("X_ERROR", 0.01), [qc.qubits[0]])
        layers = asap_layers(qc)
        assert layers[1] >= layers[0]


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


class TestScheduleOrder:
    """The emission order groups a layer into parallel columns."""

    @staticmethod
    def _names(qc):
        schedule = asap_schedule(qc)
        return [qc.data[i].op.name for i in schedule.order]

    def test_order_is_a_permutation(self):
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        qc.measure(qc.qubits[1], qc.clbits[0])
        assert sorted(asap_schedule(qc).order) == list(range(len(qc.data)))

    def test_order_respects_shared_resources(self):
        """A topological order: nothing is emitted before something it follows."""
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.cx(0, 1)
        qc.h(2)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[2]])
        qc.barrier([qc.qubits[0]])
        qc.measure(qc.qubits[0], qc.clbits[0])
        qc.parity([qc.clbits[0]])

        position = {i: p for p, i in enumerate(asap_schedule(qc).order)}
        seen: dict[object, int] = {}
        for i, instr in enumerate(qc.data):
            for r in [*instr.qubits, *instr.clbits]:
                if r in seen:
                    assert position[seen[r]] < position[i], f"{i} overtook {seen[r]}"
                seen[r] = i

    def test_disjoint_gates_are_emitted_together(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        assert asap_schedule(qc).order == [0, 1, 2]

    def test_interleaved_chains_pack_in_parallel(self):
        """Both chains' gates come out together, then both their channels."""
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.cx(0, 1)
        qc.h(5)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[5]])
        qc.cx(5, 4)

        schedule = asap_schedule(qc)
        assert schedule.layers == [0, 0, 1, 0, 0, 1]
        assert schedule.order == [0, 3, 1, 4, 2, 5]

    def test_a_gate_and_its_channel_keep_their_order(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.x(0)
        assert self._names(qc) == ["h", "stim.X_ERROR", "x"]

    def test_parity_follows_its_measurement(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(qc.qubits[0], qc.clbits[0])
        qc.parity([qc.clbits[0]])
        assert asap_schedule(qc).order == [0, 1]


class TestIsTransparent:
    """is_transparent must agree with what asap_layers does."""

    def test_real_gate_is_not_transparent(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        assert not is_transparent(qc.data[0])

    def test_measure_is_not_transparent(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(qc.qubits[0], qc.clbits[0])
        assert not is_transparent(qc.data[0])

    @pytest.mark.parametrize("op", [QubitAlloc(), QubitDealloc()])
    def test_bookkeeping_is_transparent(self, op):
        qc = QuantumCircuit(1)
        qc.append(op, [qc.qubits[0]])
        assert is_transparent(qc.data[0])

    def test_barrier_is_transparent(self):
        qc = QuantumCircuit(1)
        qc.barrier()
        assert is_transparent(qc.data[0])

    def test_noise_gate_is_transparent(self):
        qc = QuantumCircuit(1)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        assert is_transparent(qc.data[0])


class TestIsFullWidthBarrier:
    """The predicate that decides whether a barrier is a global time boundary."""

    def test_bare_barrier_is_full_width(self):
        qc = QuantumCircuit(3)
        qc.barrier()
        assert is_full_width_barrier(qc.data[0], qc)

    def test_partial_barrier_is_not(self):
        qc = QuantumCircuit(3)
        qc.barrier([qc.qubits[0], qc.qubits[1]])
        assert not is_full_width_barrier(qc.data[0], qc)

    def test_explicit_all_qubits_is_full_width(self):
        qc = QuantumCircuit(3)
        qc.barrier(list(qc.qubits))
        assert is_full_width_barrier(qc.data[0], qc)

    def test_non_barrier_is_never_full_width(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        assert not is_full_width_barrier(qc.data[0], qc)

    def test_single_qubit_circuit(self):
        """On a one-qubit register even a one-qubit barrier is full width."""
        qc = QuantumCircuit(1)
        qc.barrier([qc.qubits[0]])
        assert is_full_width_barrier(qc.data[0], qc)


# ---------------------------------------------------------------------------
# Stim conversion: TICK only for full-width barriers
# ---------------------------------------------------------------------------


class TestBarrierToTick:
    """A barrier becomes a TICK exactly when it is a global time boundary."""

    def test_full_width_barrier_ticks(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier()
        qc.h(0)
        assert qc.to_stim().num_ticks == 1

    def test_partial_barrier_does_not_tick(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(0)
        assert qc.to_stim().num_ticks == 0

    def test_barrier_over_every_qubit_explicitly_ticks(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier(list(qc.qubits))
        qc.h(0)
        assert qc.to_stim().num_ticks == 1

    def test_mixed_barriers(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(0)
        qc.barrier()
        qc.h(0)
        assert qc.to_stim().num_ticks == 1

    def test_partial_barrier_still_fences_the_compiler(self):
        """Losing the TICK must not lose the reordering constraint."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.x(0)
        names = [instr.op.name for instr in layerize()(qc).data]
        assert names.index("h") < names.index("barrier") < names.index("x")
