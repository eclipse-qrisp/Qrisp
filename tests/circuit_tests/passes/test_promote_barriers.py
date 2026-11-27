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
from qrisp.circuit.pass_management.passes import promote_barriers
from qrisp.circuit.pass_management.scheduling import asap_layers
from qrisp.circuit.quantum_circuit import is_full_width_barrier

pytest.importorskip("stim")


def _names(qc: QuantumCircuit) -> list[str]:
    return [instr.op.name for instr in qc.data]


class TestPromoteBarriersBasic:
    """Smoke tests - basic invariants."""

    def test_returns_quantum_circuit(self):
        qc = QuantumCircuit(2)
        qc.barrier([qc.qubits[0]])
        assert isinstance(promote_barriers(qc), QuantumCircuit)

    def test_empty_circuit(self):
        assert len(promote_barriers(QuantumCircuit(3)).data) == 0

    def test_circuit_without_barriers_is_unchanged(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(2)
        result = promote_barriers(qc)
        assert _names(result) == _names(qc)
        for before, after in zip(qc.data, result.data, strict=True):
            assert before.qubits == after.qubits

    def test_instruction_order_preserved(self):
        """The pass widens barriers; it must not reorder anything."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(1)
        qc.barrier([qc.qubits[1], qc.qubits[2]])
        qc.h(2)
        assert _names(promote_barriers(qc)) == ["h", "barrier", "h", "barrier", "h"]


class TestPromoteBarriersWidening:
    """The widening itself."""

    def test_partial_barrier_becomes_full_width(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.barrier([qc.qubits[0], qc.qubits[1]])
        result = promote_barriers(qc)
        assert is_full_width_barrier(result.data[1], result)
        assert len(result.data[1].qubits) == 4

    def test_every_barrier_is_promoted(self):
        qc = QuantumCircuit(3)
        qc.barrier([qc.qubits[0]])
        qc.barrier([qc.qubits[1], qc.qubits[2]])
        qc.barrier()
        result = promote_barriers(qc)
        assert all(is_full_width_barrier(instr, result) for instr in result.data)

    def test_full_width_barrier_stays_full_width(self):
        qc = QuantumCircuit(3)
        qc.barrier()
        result = promote_barriers(qc)
        assert is_full_width_barrier(result.data[0], result)

    def test_idempotent(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(1)
        once = promote_barriers(qc)
        twice = promote_barriers(once)
        assert _names(once) == _names(twice)
        for a, b in zip(once.data, twice.data, strict=True):
            assert a.qubits == b.qubits

    def test_input_circuit_not_mutated(self):
        qc = QuantumCircuit(3)
        qc.barrier([qc.qubits[0]])
        promote_barriers(qc)
        assert len(qc.data[0].qubits) == 1


class TestPromoteBarriersEffects:
    """What promotion buys, and what it costs."""

    def test_promotion_produces_a_tick(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.h(0)
        assert qc.to_stim().num_ticks == 0
        assert promote_barriers(qc).to_stim().num_ticks == 1

    def test_promotion_widens_the_schedule(self):
        """§2.8: promoting a chain-A fence costs the independent chain B two layers."""
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

        assert len(set(asap_layers(qc))) == 4
        assert len(set(asap_layers(promote_barriers(qc)))) == 6

    def test_promotion_is_a_no_op_when_barriers_already_global(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.barrier()
        qc.h(0)
        assert asap_layers(promote_barriers(qc)) == asap_layers(qc)

    def test_measurement_statistics_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier([qc.qubits[0]])
        qc.h(2)
        qc.measure(qc.qubits)
        assert promote_barriers.compare_measurement(qc)

    def test_unitary_preserved(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier([qc.qubits[0]])
        qc.cx(0, 1)
        qc.h(2)
        assert promote_barriers.compare_unitary(qc)
