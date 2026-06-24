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

import numpy as np

from qrisp import PassManager, QuantumCircuit, decompose
from qrisp.circuit import GPhaseGate


class TestDecomposePass:
    """Test the :func:`decompose` pass."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_default_level_is_inf(self):
        """``decompose()`` with no arguments uses level=np.inf."""
        dec = decompose()
        assert dec.__name__ == "decompose(level=inf)"

    def test_explicit_level(self):
        """``decompose(level=3)`` sets the name accordingly."""
        dec = decompose(level=3)
        assert dec.__name__ == "decompose(level=3)"

    def test_level_zero(self):
        """``decompose(level=0)`` sets the name accordingly."""
        dec = decompose(level=0)
        assert dec.__name__ == "decompose(level=0)"

    def test_is_callable(self):
        """The result of ``decompose()`` is callable on a QuantumCircuit."""
        dec = decompose()
        qc = QuantumCircuit(2)
        result = dec(qc)
        assert isinstance(result, QuantumCircuit)

    # ------------------------------------------------------------------
    # Integration with PassManager
    # ------------------------------------------------------------------

    def test_pass_manager_integration(self):
        """``decompose`` works inside a :class:`PassManager` pipeline."""
        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)

        pm = PassManager()
        pm += decompose()
        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)
        assert len(result.data) > len(qc.data)

    # ------------------------------------------------------------------
    # Decomposition behaviour
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mcx_circuit(n_ctrl: int = 2) -> QuantumCircuit:
        """Return a circuit with a single MCX gate."""
        qc = QuantumCircuit(n_ctrl + 1)
        ctrl = list(range(n_ctrl))
        targ = n_ctrl
        qc.mcx(ctrl, targ)
        return qc

    # ------------------------------------------------------------------
    # Multi-level decomposition behaviour (MCX has a 2-level definition)
    # ------------------------------------------------------------------

    def test_level_zero_no_decomposition(self):
        """Level 0 leaves the circuit unchanged."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm = PassManager()
        pm += decompose(level=0)
        result = pm.run(qc)
        assert len(result.data) == 1

    def test_level_one_unwraps_one_layer(self):
        """Level 1 unwraps the MCX definition but ``gray multi cx``
        is not decomposed further at this recursion depth.
        """
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm = PassManager()
        pm += decompose(level=1)
        result = pm.run(qc)
        # gray multi cx stays as a single instruction at level 1
        assert len(result.data) == 1
        assert "gray" in result.data[0].op.name.lower()

    def test_level_two_fully_decomposes(self):
        """Level 2 decomposes MCX all the way to elementary gates."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm = PassManager()
        pm += decompose(level=2)
        result = pm.run(qc)
        assert len(result.data) > 1

    def test_decompose_predicate_blocks_decomposition(self):
        """A ``decompose_predicate`` that returns ``False`` prevents decomposition."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm = PassManager()
        pm += decompose(level=5, decompose_predicate=lambda op: False)
        result = pm.run(qc)
        # Nothing should be decomposed — the gate stays intact
        assert len(result.data) == 1
        assert "cx" in result.data[0].op.name

    # ------------------------------------------------------------------
    # collect_gphases
    # ------------------------------------------------------------------

    def test_collect_gphases_merges_multiple(self):
        """Multiple gphase gates are merged into a single one."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(GPhaseGate(0.5), [0])
        qc.cx(0, 1)
        qc.append(GPhaseGate(0.3), [1])
        qc.append(GPhaseGate(0.2), [0])

        pm = PassManager()
        pm += decompose(collect_gphases=True)
        result = pm.run(qc)

        gphase_ops = [instr for instr in result.data if instr.op.name == "gphase"]
        assert len(gphase_ops) == 1, f"Expected 1 gphase, got {len(gphase_ops)}"
        assert abs(gphase_ops[0].op.params[0] - 1.0) < 1e-10, (
            f"Expected accumulated phase 1.0, got {gphase_ops[0].op.params[0]}"
        )

    def test_collect_gphases_no_gphases(self):
        """Circuit with no gphase gates is unchanged."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pm = PassManager()
        pm += decompose(collect_gphases=True)
        result = pm.run(qc)

        gphase_ops = [instr for instr in result.data if instr.op.name == "gphase"]
        assert len(gphase_ops) == 0

    def test_collect_gphases_anchored_on_last_nonzero_qubit_op(self):
        """The merged gphase is placed on the first qubit of the last
        instruction with num_qubits > 0, not on a freshly allocated qubit.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(GPhaseGate(0.4), [0])
        qc.cx(0, 1)  # last op with num_qubits > 0
        qc.append(GPhaseGate(0.6), [1])

        pm = PassManager()
        pm += decompose(collect_gphases=True)
        result = pm.run(qc)

        gphase_ops = [instr for instr in result.data if instr.op.name == "gphase"]
        assert len(gphase_ops) == 1
        # The gphase should be on qubit 0 (first qubit of CX, the last num_qubits>0 instr)
        anchor_qubit = gphase_ops[0].qubits[0]
        assert anchor_qubit == qc.qubits[0], f"Expected gphase on qubit 0, got {anchor_qubit}"

    def test_collect_gphases_only_gphases(self):
        """Circuit with only gphase gates: no anchor → no gphase emitted."""
        qc = QuantumCircuit(1)
        qc.append(GPhaseGate(0.5), [0])
        qc.append(GPhaseGate(0.7), [0])

        pm = PassManager()
        pm += decompose(collect_gphases=True)
        result = pm.run(qc)

        # No non-gphase instruction → nowhere to anchor → empty circuit
        assert len(result.data) == 0

    def test_collect_gphases_preserves_unitary(self):
        """The unitary before and after gphase collection must match."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.append(GPhaseGate(0.5), [0])
        qc.cx(0, 1)
        qc.append(GPhaseGate(0.3), [1])
        qc.rz(0.7, 0)
        qc.append(GPhaseGate(0.2), [0])

        pm = PassManager()
        pm += decompose(collect_gphases=True)
        result = pm.run(qc)

        # Compare unitaries — the gphase gate carries the global phase
        U_orig = qc.get_unitary()
        U_result = result.get_unitary()
        assert np.allclose(U_orig, U_result, atol=1e-6), "Unitaries must match after gphase collection"

    def test_collect_gphases_default_false(self):
        """Default behaviour (collect_gphases=False) leaves gphase gates untouched."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(GPhaseGate(0.5), [0])

        pm = PassManager()
        pm += decompose()
        result = pm.run(qc)

        gphase_ops = [instr for instr in result.data if instr.op.name == "gphase"]
        assert len(gphase_ops) == 1, "Default decompose should preserve gphase gates"

    def test_decompose_predicate_selective(self):
        """A ``decompose_predicate`` can target specific gates by name."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm = PassManager()
        # The MCX name contains "cx" — predicate allows decomposition
        pm += decompose(level=2, decompose_predicate=lambda op: "cx" in op.name)
        result = pm.run(qc)
        assert len(result.data) > 1

    def test_elementary_gates_preserved(self):
        """Elementary gates (H, CX) are left unchanged at any level."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pm = PassManager()
        pm += decompose(level=3)
        result = pm.run(qc)

        # H and CX are elementary — they stay as-is
        names = [instr.op.name for instr in result.data]
        assert "h" in names
        assert "cx" in names

    def test_multi_ctrl_mcx_decomposes(self):
        """A 3-control MCX (cccX) decomposes at default level."""
        qc = self._make_mcx_circuit(n_ctrl=3)
        pm = PassManager()
        pm += decompose(level=2)
        result = pm.run(qc)
        assert len(result.data) > 1

    def test_empty_circuit(self):
        """Decomposing an empty circuit is a no-op."""
        qc = QuantumCircuit(2)
        pm = PassManager()
        pm += decompose(level=2)
        result = pm.run(qc)
        assert len(result.data) == 0

    def test_preserves_qubit_count(self):
        """Decomposition must not change the number of qubits."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm = PassManager()
        pm += decompose(level=2)
        result = pm.run(qc)
        assert result.num_qubits() == qc.num_qubits()

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_high_level_matches_max_decomposition(self):
        """A very high level is fine — decomposition saturates at elementary gates."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        pm_low = PassManager()
        pm_low += decompose(level=5)
        result_low = pm_low.run(qc)

        pm_high = PassManager()
        pm_high += decompose(level=50)
        result_high = pm_high.run(qc)

        # Once fully decomposed, higher level shouldn't change anything
        assert len(result_high.data) == len(result_low.data)

    def test_decompose_then_run_same_as_run_then_decompose(self):
        """Two decompose calls with the same level give the same result."""
        qc = self._make_mcx_circuit(n_ctrl=2)
        dec = decompose(level=2)

        result1 = dec(qc)
        result2 = dec(qc)
        assert len(result1.data) == len(result2.data)

    # ------------------------------------------------------------------
    # Correctness (unitary equivalence)
    # ------------------------------------------------------------------

    def test_unitary_equivalence_default_decompose(self):
        """Decomposing with default settings preserves the unitary."""
        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)
        qc.h(2)

        pm = PassManager()
        pm += decompose()
        results = pm.verify(qc, "unitary")
        assert all(passed for _, passed in results)

    def test_unitary_equivalence_with_predicate(self):
        """Selective decomposition via predicate preserves the unitary."""
        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)
        qc.h(0)

        pm = PassManager()
        pm += decompose(decompose_predicate=lambda op: "cx" in op.name)
        results = pm.verify(qc, "unitary")
        assert all(passed for _, passed in results)
