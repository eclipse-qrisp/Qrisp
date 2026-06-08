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
from qrisp import PassManager, CircuitPass
from qrisp.circuit import QuantumCircuit
from qrisp.circuit.pass_management import PassManager as PassManagerFromCircuit


# ---------------------------------------------------------------------------
# Simple pass helpers used across tests
# ---------------------------------------------------------------------------

identity_pass = CircuitPass(lambda qc: qc)
identity_pass.__name__ = "identity_pass"
identity_pass.__doc__ = "Pass that returns the circuit unchanged."

add_h_pass = CircuitPass(lambda qc: qc.h(qc.qubits[0]) or qc)
add_h_pass.__name__ = "add_h_pass"
add_h_pass.__doc__ = "Pass that appends an H gate on qubit 0."

add_cx_pass = CircuitPass(lambda qc: qc.cx(qc.qubits[0], qc.qubits[1]) or qc)
add_cx_pass.__name__ = "add_cx_pass"
add_cx_pass.__doc__ = "Pass that appends a CX gate on qubits 0 and 1."


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestPassManagerImports:
    def test_import_from_qrisp_top_level(self):
        """PassManager must be importable directly from qrisp."""
        from qrisp import PassManager as PM
        assert PM is PassManager

    def test_import_from_qrisp_circuit_passes(self):
        """PassManager must be importable from qrisp.circuit.pass_management."""
        assert PassManagerFromCircuit is PassManager

    def test_import_from_qrisp_circuit_passes_pass_manager(self):
        """PassManager must be importable from the leaf module."""
        from qrisp.circuit.pass_management.pass_manager import PassManager as PM
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

        pass_a = CircuitPass(lambda qc: log.append("a") or qc)
        pass_a.__name__ = "pass_a"
        pass_b = CircuitPass(lambda qc: log.append("b") or qc)
        pass_b.__name__ = "pass_b"

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


# ---------------------------------------------------------------------------
# __iadd__ operator
# ---------------------------------------------------------------------------


class TestPassManagerIAdd:
    """Test the ``+=`` in-place operator on :class:`PassManager`."""

    def test_iadd_single_pass_appends(self):
        """``pm += circuit_pass`` must append a single pass."""
        pm = PassManager()
        pm += identity_pass
        assert len(pm) == 1
        assert pm._passes[0] is identity_pass

    def test_iadd_multiple_single_passes(self):
        """Repeated ``+=`` must accumulate passes in the correct order."""
        pm = PassManager()
        pm += identity_pass
        pm += add_h_pass
        pm += add_cx_pass
        assert len(pm) == 3
        assert pm._passes[0] is identity_pass
        assert pm._passes[1] is add_h_pass
        assert pm._passes[2] is add_cx_pass

    def test_iadd_pass_manager_extends(self):
        """``pm += other_pm`` must extend *pm* with all passes from *other_pm*."""
        pm = PassManager([identity_pass])
        other = PassManager([add_h_pass, add_cx_pass])
        pm += other
        assert len(pm) == 3
        assert pm._passes[0] is identity_pass
        assert pm._passes[1] is add_h_pass
        assert pm._passes[2] is add_cx_pass

    def test_iadd_pass_manager_returns_self(self):
        """``pm += other_pm`` must return self."""
        pm = PassManager([identity_pass])
        other = PassManager([add_h_pass])
        result = pm.__iadd__(other)
        assert result is pm

    def test_iadd_pass_manager_empty(self):
        """Extending with an empty PassManager is a no-op."""
        pm = PassManager([identity_pass])
        other = PassManager()
        pm += other
        assert len(pm) == 1
        assert pm._passes[0] is identity_pass

    def test_iadd_into_empty_from_pass_manager(self):
        """Extending an empty PassManager with a non-empty one works."""
        pm = PassManager()
        other = PassManager([add_h_pass, add_cx_pass])
        pm += other
        assert len(pm) == 2
        assert pm._passes[0] is add_h_pass
        assert pm._passes[1] is add_cx_pass

    def test_iadd_rejects_non_pass_non_passmanager(self):
        """``pm += x`` where *x* is neither a CircuitPass nor PassManager must raise TypeError."""
        pm = PassManager()
        with pytest.raises(TypeError, match="only accepts CircuitPass or PassManager"):
            pm += "not_a_pass"

    def test_iadd_rejects_int(self):
        """``pm += 42`` must raise TypeError."""
        pm = PassManager()
        with pytest.raises(TypeError):
            pm += 42

    def test_iadd_rejects_none(self):
        """``pm += None`` must raise TypeError."""
        pm = PassManager()
        with pytest.raises(TypeError):
            pm += None


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class TestPassManagerVerify:
    """Test PassManager.verify with unitary and measurement verification."""

    # --- Helper: bad passes ---

    @staticmethod
    def _bad_unitary_pass() -> CircuitPass:
        def _bad(qc: QuantumCircuit) -> QuantumCircuit:
            new_qc = qc.copy()
            new_qc.x(0)
            return new_qc

        cp = CircuitPass(_bad)
        cp.__name__ = "bad_unitary"
        return cp

    @staticmethod
    def _make_ghz_measured(n: int = 3) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        for _ in range(n):
            qc.add_clbit()
        qc.h(qc.qubits[0])
        for i in range(n - 1):
            qc.cx(qc.qubits[i], qc.qubits[i + 1])
        qc.measure(qc.qubits, qc.clbits)
        return qc

    # --- Unitary verification ---

    def test_unitary_all_pass(self):
        from qrisp.circuit.pass_management.passes.cancel_inverses import cancel_inverses

        pm = PassManager([cancel_inverses, identity_pass])
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        results = pm.verify(qc, "unitary")
        assert results == [("cancel_inverses", True), ("identity_pass", True)]

    def test_unitary_one_fails(self):
        from qrisp.circuit.pass_management.passes.cancel_inverses import cancel_inverses

        pm = PassManager([cancel_inverses, self._bad_unitary_pass()])
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        results = pm.verify(qc, "unitary")
        assert results == [("cancel_inverses", True), ("bad_unitary", False)]

    def test_unitary_first_fails(self):
        pm = PassManager([self._bad_unitary_pass(), identity_pass])
        qc = QuantumCircuit(2)
        qc.h(0)
        results = pm.verify(qc, "unitary")
        assert results == [("bad_unitary", False), ("identity_pass", True)]

    def test_unitary_all_fail(self):
        pm = PassManager([self._bad_unitary_pass(), self._bad_unitary_pass()])
        qc = QuantumCircuit(2)
        results = pm.verify(qc, "unitary")
        assert results == [("bad_unitary", False), ("bad_unitary", False)]

    def test_unitary_empty_manager(self):
        pm = PassManager()
        qc = QuantumCircuit(2)
        results = pm.verify(qc, "unitary")
        assert results == []

    def test_unitary_passes_kwargs(self):
        """verify forwards kwargs like precision to compare_unitary."""
        bad = self._bad_unitary_pass()
        pm = PassManager([bad])
        qc = QuantumCircuit(2)
        qc.h(0)
        # Very loose precision: X should be treated as identity-ish
        results = pm.verify(qc, "unitary", precision=-5)
        assert results == [("bad_unitary", True)]

    def test_unitary_returns_correct_length(self):
        from qrisp.circuit.pass_management.passes.cancel_inverses import cancel_inverses
        from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates

        pm = PassManager([cancel_inverses, combine_single_qubit_gates, identity_pass])
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        qc.z(0)
        results = pm.verify(qc, "unitary")
        assert len(results) == 3

    # --- Measurement verification ---

    def test_measurement_all_pass(self):
        qc = self._make_ghz_measured(3)
        pm = PassManager([identity_pass, identity_pass])
        results = pm.verify(qc, "measurements")
        assert results == [("identity_pass", True), ("identity_pass", True)]

    def test_measurement_one_fails(self):
        def add_x_before_measure(qc: QuantumCircuit) -> QuantumCircuit:
            new_qc = qc.clearcopy()
            for instr in qc.data:
                if instr.op.name == "measure":
                    new_qc.x(0)
                new_qc.append(instr.op, instr.qubits, instr.clbits)
            return new_qc

        bad_meas = CircuitPass(add_x_before_measure)
        bad_meas.__name__ = "bad_meas"

        qc = self._make_ghz_measured(3)
        pm = PassManager([identity_pass, bad_meas])
        results = pm.verify(qc, "measurements")
        assert results == [("identity_pass", True), ("bad_meas", False)]

    # --- Error handling ---

    def test_unknown_verification_type_raises(self):
        pm = PassManager([identity_pass])
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="Unknown verification_type"):
            pm.verify(qc, "bogus")

    # --- visualize_failures ---

    def test_visualize_failures_does_not_crash(self, capsys):
        """visualize_failures calls visualize on failing passes
        without raising exceptions."""
        pm = PassManager([self._bad_unitary_pass(), identity_pass])
        qc = QuantumCircuit(2)
        qc.h(0)
        results = pm.verify(qc, "unitary", visualize_failures=True)
        assert results == [("bad_unitary", False), ("identity_pass", True)]
        captured = capsys.readouterr()
        # visualize should have produced output for the bad pass
        assert "bad_unitary" in captured.out
        assert "Before" in captured.out
        assert "After" in captured.out

    def test_visualize_failures_only_bad_passes(self, capsys):
        """Only failing passes should be visualized."""
        from qrisp.circuit.pass_management.passes.cancel_inverses import cancel_inverses

        pm = PassManager([cancel_inverses, self._bad_unitary_pass()])
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        results = pm.verify(qc, "unitary", visualize_failures=True)
        assert results == [("cancel_inverses", True), ("bad_unitary", False)]
        captured = capsys.readouterr()
        assert "cancel_inverses" not in captured.out
        assert "bad_unitary" in captured.out
