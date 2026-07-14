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

from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management.passes.arrange_swaps import arrange_swaps


class TestArrangeSwaps:
    def test_swap_used_unused_reorders(self):
        """SWAP(used, unused) should be reordered to SWAP(unused, used)."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.swap(0, 2)

        result = arrange_swaps(qc)

        swap_instrs = [i for i in result.data if i.op.name == "swap"]
        assert len(swap_instrs) == 1
        assert result.qubits.index(swap_instrs[0].qubits[0]) == 2
        assert result.qubits.index(swap_instrs[0].qubits[1]) == 0

    def test_swap_both_used_unchanged(self):
        """SWAP(used, used) should remain in original order."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 1)

        result = arrange_swaps(qc)

        swap_instrs = [i for i in result.data if i.op.name == "swap"]
        assert len(swap_instrs) == 1
        assert result.qubits.index(swap_instrs[0].qubits[0]) == 0
        assert result.qubits.index(swap_instrs[0].qubits[1]) == 1

    def test_swap_both_unused_removed(self):
        """SWAP(unused, unused) should be filtered out entirely."""
        qc = QuantumCircuit(4)
        qc.swap(2, 3)

        result = arrange_swaps(qc)

        swap_instrs = [i for i in result.data if i.op.name == "swap"]
        assert len(swap_instrs) == 0

    def test_multiple_swaps_mixed(self):
        """Multiple SWAPs with mixed usage are each handled correctly."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 2)
        qc.swap(1, 3)

        result = arrange_swaps(qc)

        swap_instrs = [i for i in result.data if i.op.name == "swap"]
        assert len(swap_instrs) == 2

        assert result.qubits.index(swap_instrs[0].qubits[0]) == 2
        assert result.qubits.index(swap_instrs[0].qubits[1]) == 0

        assert result.qubits.index(swap_instrs[1].qubits[0]) == 3
        assert result.qubits.index(swap_instrs[1].qubits[1]) == 1

    def test_no_swaps_circuit_unchanged(self):
        """A circuit with no SWAPs passes through unchanged."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = arrange_swaps(qc)

        assert len(result.data) == len(qc.data)


# ---------------------------------------------------------------------------
# Correctness: compare_unitary
# ---------------------------------------------------------------------------


class TestArrangeSwapsCorrectness:
    """Verify arrange_swaps preserves the circuit unitary."""

    def test_unitary_preserved_used_unused_swap(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.swap(0, 2)
        assert arrange_swaps.compare_unitary(qc)

    def test_unitary_preserved_both_used_swap(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 1)
        assert arrange_swaps.compare_unitary(qc)

    def test_unitary_preserved_both_unused_identity(self):
        """Unused qubits never touched → pass is a no-op → unitary preserved."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        assert arrange_swaps.compare_unitary(qc)

    def test_unitary_preserved_mixed_swaps(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 2)
        qc.swap(1, 3)
        assert arrange_swaps.compare_unitary(qc)
