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

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.simulator.preprocessing.disentangling import _Disentangler, _insert_disentangling

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instr_key(qc: QuantumCircuit, instr) -> tuple:
    return (instr.op.name, tuple(qc.qubits.index(q) for q in instr.qubits))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInsertDisentanglingStructuralInvariants:
    """_insert_disentangling always appends one reset per qubit and only inserts
    _Disentangler markers, never altering the original computational instructions."""

    def test_reset_appended_per_qubit(self):
        """One reset instruction is appended for every qubit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.cx(1, 2)
        result = _insert_disentangling(qc)
        assert sum(1 for instr in result.data if instr.op.name == "reset") == 3

    def test_original_instructions_preserved_in_order(self):
        """Only reset/disentangle markers are inserted; original ops keep their order."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.p(0.3, 0)
        qc.cx(0, 1)
        original = [_instr_key(qc, instr) for instr in qc.data]

        result = _insert_disentangling(qc)
        filtered = [_instr_key(result, instr) for instr in result.data if instr.op.name not in ("reset", "disentangle")]
        assert filtered == original

    def test_qubit_and_clbit_sets_preserved(self):
        """Qubit/clbit registers are unchanged."""
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.measure(0, 0)
        result = _insert_disentangling(qc)
        assert set(result.qubits) == set(qc.qubits)
        assert set(result.clbits) == set(qc.clbits)

    def test_disentangler_ops_are_single_qubit(self):
        """Every inserted _Disentangler instruction acts on exactly one qubit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.p(0.3, 0)
        qc.cx(0, 1)
        result = _insert_disentangling(qc)
        for instr in result.data:
            if instr.op.name == "disentangle":
                assert len(instr.qubits) == 1

    def test_empty_circuit(self):
        """An empty circuit still gets a terminal reset per qubit."""
        qc = QuantumCircuit(2)
        result = _insert_disentangling(qc)
        assert sum(1 for instr in result.data if instr.op.name == "reset") == 2


class TestDisentangler:
    """_Disentangler is a 1-qubit marker operation with trivial permeability."""

    def test_name_and_qubit_count(self):
        """_Disentangler is named 'disentangle' and acts on a single qubit."""
        op = _Disentangler()
        assert op.name == "disentangle"
        assert op.num_qubits == 1

    def test_permeability_is_false(self):
        """_Disentangler is not permeable, so it blocks further grouping."""
        op = _Disentangler()
        assert op.permeability == {0: False}
