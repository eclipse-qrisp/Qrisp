"""
********************************************************************************
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

import sys

import pytest
from qrisp import QuantumCircuit, PassManager, visualize


class TestVisualizePass:
    """Test the :func:`visualize` pass."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_is_a_circuit_pass(self):
        """``visualize`` is a :class:`CircuitPass`."""
        from qrisp import CircuitPass
        assert isinstance(visualize, CircuitPass)

    def test_is_callable(self):
        """``visualize`` is callable on a QuantumCircuit."""
        qc = QuantumCircuit(2)
        result = visualize(qc)
        assert isinstance(result, QuantumCircuit)

    # ------------------------------------------------------------------
    # Identity behaviour
    # ------------------------------------------------------------------

    @staticmethod
    def _make_circuit() -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    def test_pass_is_identity(self):
        """The pass must not modify the circuit."""
        qc = self._make_circuit()
        result = visualize(qc)
        assert len(result.data) == len(qc.data)

    def test_preserves_qubit_count(self):
        """The pass must not change the qubit count."""
        qc = self._make_circuit()
        result = visualize(qc)
        assert result.num_qubits() == qc.num_qubits()

    # ------------------------------------------------------------------
    # Integration with PassManager
    # ------------------------------------------------------------------

    def test_pass_manager_integration(self):
        """``visualize`` works inside a :class:`PassManager` pipeline."""
        qc = self._make_circuit()
        pm = PassManager()
        pm += visualize
        result = pm.run(qc)
        assert len(result.data) == len(qc.data)

    def test_between_other_passes(self):
        """Placing visualize between other passes does not affect outcome."""
        from qrisp import decompose

        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)

        pm = PassManager()
        pm += decompose()
        pm += visualize
        pm += decompose(level=0)
        result = pm.run(qc)
        assert len(result.data) > 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def test_output_contains_circuit_diagram(self, capsys):
        """The pass prints the circuit to stdout."""
        qc = self._make_circuit()
        visualize(qc)
        captured = capsys.readouterr()
        assert "h" in captured.out.lower() or "cx" in captured.out.lower()

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_unitary_equivalence(self):
        """The visualize pass must preserve unitarity."""
        qc = self._make_circuit()
        pm = PassManager()
        pm += visualize
        results = pm.verify(qc, "unitary")
        assert all(passed for _, passed in results)
