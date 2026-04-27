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

import pytest
from qrisp import PassManager
from qrisp.circuit import QuantumCircuit
from qrisp.circuit.passes import PassManager as PassManagerFromCircuit


# ---------------------------------------------------------------------------
# Simple pass helpers used across tests
# ---------------------------------------------------------------------------

def identity_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Pass that returns the circuit unchanged."""
    return qc


def add_h_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Pass that appends an H gate on qubit 0."""
    qc.h(qc.qubits[0])
    return qc


def add_cx_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Pass that appends a CX gate on qubits 0 and 1."""
    qc.cx(qc.qubits[0], qc.qubits[1])
    return qc


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestPassManagerImports:
    def test_import_from_qrisp_top_level(self):
        """PassManager must be importable directly from qrisp."""
        from qrisp import PassManager as PM
        assert PM is PassManager

    def test_import_from_qrisp_circuit_passes(self):
        """PassManager must be importable from qrisp.circuit.passes."""
        assert PassManagerFromCircuit is PassManager

    def test_import_from_qrisp_circuit_passes_pass_manager(self):
        """PassManager must be importable from the leaf module."""
        from qrisp.circuit.passes.pass_manager import PassManager as PM
        assert PM is PassManager


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPassManagerConstruction:
    def test_default_construction(self):
        pm = PassManager()
        assert len(pm) == 0

    def test_construction_with_initial_passes(self):
        pm = PassManager(passes=[identity_pass, add_h_pass])
        assert len(pm) == 2

    def test_construction_with_none(self):
        pm = PassManager(passes=None)
        assert len(pm) == 0

    def test_from_list(self):
        pm = PassManager.from_list([identity_pass, add_h_pass])
        assert len(pm) == 2
        assert pm._passes[0] is identity_pass
        assert pm._passes[1] is add_h_pass


# ---------------------------------------------------------------------------
# Mutation methods
# ---------------------------------------------------------------------------

class TestPassManagerMutation:
    def test_add_pass_appends(self):
        pm = PassManager()
        pm.add_pass(identity_pass)
        assert len(pm) == 1
        assert pm._passes[0] is identity_pass

    def test_add_pass_returns_self_for_chaining(self):
        pm = PassManager()
        result = pm.add_pass(identity_pass)
        assert result is pm

    def test_method_chaining(self):
        pm = PassManager().add_pass(identity_pass).add_pass(add_h_pass)
        assert len(pm) == 2

    def test_insert_pass(self):
        pm = PassManager()
        pm.add_pass(identity_pass)
        pm.add_pass(add_cx_pass)
        pm.insert_pass(1, add_h_pass)

        assert len(pm) == 3
        assert pm._passes[0] is identity_pass
        assert pm._passes[1] is add_h_pass
        assert pm._passes[2] is add_cx_pass

    def test_insert_pass_returns_self(self):
        pm = PassManager()
        result = pm.insert_pass(0, identity_pass)
        assert result is pm

    def test_remove_pass(self):
        pm = PassManager()
        pm.add_pass(identity_pass)
        pm.add_pass(add_h_pass)
        pm.remove_pass(0)

        assert len(pm) == 1
        assert pm._passes[0] is add_h_pass

    def test_remove_pass_returns_self(self):
        pm = PassManager()
        pm.add_pass(identity_pass)
        result = pm.remove_pass(0)
        assert result is pm

    def test_clear(self):
        pm = PassManager()
        pm.add_pass(identity_pass)
        pm.add_pass(add_h_pass)
        pm.clear()
        assert len(pm) == 0

    def test_clear_returns_self(self):
        pm = PassManager()
        result = pm.clear()
        assert result is pm


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

class TestPassManagerRun:
    def _make_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(qc.qubits[0])
        return qc

    def test_run_empty_pass_list_returns_circuit(self):
        qc = self._make_circuit()
        pm = PassManager()
        result = pm.run(qc)
        assert result is qc

    def test_run_identity_pass(self):
        qc = self._make_circuit()
        pm = PassManager([identity_pass])
        result = pm.run(qc)
        assert result is qc

    def test_run_applies_passes_in_order(self):
        """Passes are applied sequentially; order of side-effects must be correct."""
        log = []

        def pass_a(qc):
            log.append("a")
            return qc

        def pass_b(qc):
            log.append("b")
            return qc

        pm = PassManager([pass_a, pass_b])
        pm.run(QuantumCircuit(1))

        assert log == ["a", "b"]

    def test_run_passes_output_to_next_input(self):
        """Each pass receives the output of the previous pass."""
        qc = QuantumCircuit(2)
        initial_gate_count = len(qc.data)

        pm = PassManager([add_h_pass, add_cx_pass])
        result = pm.run(qc)

        assert len(result.data) == initial_gate_count + 2


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------

class TestPassManagerDunder:
    def test_len(self):
        pm = PassManager([identity_pass, add_h_pass])
        assert len(pm) == 2

    def test_iter(self):
        pm = PassManager([identity_pass, add_h_pass])
        collected = list(pm)
        assert collected == [identity_pass, add_h_pass]

    def test_repr_contains_pass_names(self):
        pm = PassManager([identity_pass, add_h_pass])
        r = repr(pm)
        assert "identity_pass" in r
        assert "add_h_pass" in r

    def test_repr_empty(self):
        pm = PassManager()
        assert repr(pm) == "PassManager([])"
