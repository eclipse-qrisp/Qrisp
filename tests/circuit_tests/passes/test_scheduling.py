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
    intra_layer_substeps,
    is_full_width_barrier,
    is_transparent,
)
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
        # allocs ride at their qubit's current layer (-1 = untouched), the
        # dealloc rides in the cx layer, and neither adds a layer of its own.
        assert asap_layers(qc) == [-1, 0, -1, 1, 1]

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

    def test_several_noise_gates_share_a_layer(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.append(StimNoiseGate("X_ERROR", 0.01), [qc.qubits[0]])
        assert asap_layers(qc) == [0, 0, 0]


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
        """The upper clamp: data order is not layer-monotone, so pushing a
        channel to the nearest preceding time step must not move it past a
        later gate on its own qubit.
        """
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(2)
        qc.x(2)
        qc.z(2)  # q2 chain reaches layer 2 while q0 is still at layer 0
        qc.append(StimNoiseGate("X_ERROR", 0.01), [qc.qubits[0]])
        qc.x(0)

        layers = asap_layers(qc)
        channel_layer, gate_layer = layers[4], layers[5]
        assert channel_layer < gate_layer, f"channel overtook its own gate: {layers}"

    def test_channel_cannot_be_pushed_across_a_barrier(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
        qc.barrier()
        qc.h(1)

        layers = asap_layers(qc)
        assert layers[1] < layers[2], f"channel crossed the barrier: {layers}"

    def test_bookkeeping_is_not_pulled_forward(self):
        """Only error channels get the refinement - allocs still float to the front."""
        qc = QuantumCircuit(2)
        qc.append(QubitAlloc(), [qc.qubits[0]])
        qc.h(0)
        qc.append(QubitAlloc(), [qc.qubits[1]])
        assert asap_layers(qc) == [-1, 0, -1]


class TestAsapLayersBarriers:
    """Barriers constrain exactly the qubits they name."""

    def test_partial_barrier_does_not_delay_other_qubits(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(0)
        qc.h(1)
        # h(1) stays in layer 0: the barrier never named its qubit.
        assert asap_layers(qc) == [0, 1, 1, 0]

    def test_full_width_barrier_synchronises_everything(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.h(1)
        # h(1) is pushed past the barrier because the barrier names q1 too.
        assert asap_layers(qc) == [0, 1, 1]

    def test_barrier_opens_the_layer_it_fences_off(self):
        """The barrier's own layer is the layer of what follows it, not its own."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.barrier()
        qc.h(0)
        layers = asap_layers(qc)
        assert layers == [0, 1, 1]

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
        assert asap_layers(qc) == [0, 1, 1, 1]

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


class TestIntraLayerSubsteps:
    """Sub-steps decide how a layer is drawn, not what it means."""

    def test_disjoint_instructions_share_a_substep(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        assert intra_layer_substeps(qc, asap_layers(qc)) == [0, 0, 0]

    def test_reuse_of_a_qubit_advances_the_substep(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.append(StimNoiseGate("Z_ERROR", 0.5), [qc.qubits[0]])
        assert intra_layer_substeps(qc, asap_layers(qc)) == [0, 1, 2]

    def test_substeps_are_per_layer(self):
        """Each layer counts from zero again."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.x(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        assert asap_layers(qc) == [0, 0, 1, 1]
        assert intra_layer_substeps(qc, asap_layers(qc)) == [0, 1, 0, 1]

    def test_interleaved_chains_pack_in_parallel(self):
        """The two chains' gates share a sub-step, and so do their channels."""
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
        qc.cx(0, 1)
        qc.h(5)
        qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[5]])
        qc.cx(5, 4)

        layers = asap_layers(qc)
        substeps = intra_layer_substeps(qc, layers)
        assert layers == [0, 0, 1, 0, 0, 1]
        assert substeps == [0, 1, 0, 0, 1, 0]

    def test_clbits_count_towards_substeps(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(qc.qubits[0], qc.clbits[0])
        qc.parity([qc.clbits[0]])
        # The parity shares the clbit with the measurement, so it cannot be
        # drawn in the same column even though they share a layer.
        assert asap_layers(qc) == [0, 0]
        assert intra_layer_substeps(qc, asap_layers(qc)) == [0, 1]

    def test_resource_less_instruction_gets_substep_zero(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        assert intra_layer_substeps(qc, asap_layers(qc)) == [0]


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
