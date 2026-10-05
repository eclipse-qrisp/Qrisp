# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for the QuantumVariable class in qrisp.core.quantum_variable."""

import pytest

from qrisp import QuantumSession, QuantumVariable, h, x
from qrisp.jasp import jaspify
from qrisp.jasp.tracing_logic.tracing_quantum_session import TracingModeError


def _dealloc_count(qv: QuantumVariable) -> int:
    return sum(instr.op.name == "qb_dealloc" and instr.qubits[0] in qv.reg for instr in qv.qs.data)


def _is_live(qv: QuantumVariable) -> bool:
    return any(ref() is qv for ref in QuantumVariable.live_qvs)


def _assert_not_deleted(qv: QuantumVariable) -> None:
    assert qv.name in [existing.name for existing in qv.qs.qv_list]
    assert qv.name not in [existing.name for existing in qv.qs.deleted_qv_list]
    assert _is_live(qv)
    if isinstance(qv.qs, QuantumSession):
        assert not qv.is_deleted()
        assert all(qb.allocated for qb in qv.reg)
        assert _dealloc_count(qv) == 0


def _assert_deleted(qv: QuantumVariable) -> None:
    assert qv.name not in [existing.name for existing in qv.qs.qv_list]
    assert qv.name in [existing.name for existing in qv.qs.deleted_qv_list]
    assert not _is_live(qv)
    if isinstance(qv.qs, QuantumSession):
        assert qv.is_deleted()
        assert all(not qb.allocated for qb in qv.reg)
        assert _dealloc_count(qv) == len(qv.reg)


@pytest.fixture(autouse=True)
def _isolated_live_qvs():
    """Give each test an empty QuantumVariable.live_qvs and restore it afterwards.

    live_qvs lives on the QuantumVariable class and is shared process-wide, so
    without this, variables left over from other tests would leak into the checks.
    """
    original_live_qvs = QuantumVariable.live_qvs
    QuantumVariable.live_qvs = []
    yield
    QuantumVariable.live_qvs = original_live_qvs


class TestDelete:
    """Test the basic behavior of :meth:`QuantumVariable.delete`."""

    def test_delete_frees_qubits_and_unregisters_variable(self):
        qv = QuantumVariable(3)
        qv.delete()
        _assert_deleted(qv)

    def test_delete_in_tracing_mode_unregisters_variable(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            qv.delete()
            _assert_deleted(qv)
            return 0

        main()

    def test_delete_leaves_other_variables_untouched(self):
        qs = QuantumSession()
        kept = QuantumVariable(2, qs=qs)
        to_be_deleted = QuantumVariable(2, qs=qs)
        to_be_deleted.delete()
        _assert_not_deleted(kept)
        _assert_deleted(to_be_deleted)
        assert len(QuantumVariable.live_qvs) == 1

    def test_operation_on_deleted_qubit_raises(self):
        qv = QuantumVariable(2)
        _assert_not_deleted(qv)
        qv.delete()
        _assert_deleted(qv)
        with pytest.raises(Exception, match="unallocated qubit"):
            x(qv[0])


class TestDeleteTwice:
    """Test calling :meth:`QuantumVariable.delete` on an already deleted variable."""

    def test_delete_twice_in_static_mode_is_a_no_op(self):
        qs = QuantumSession()
        qv = QuantumVariable(2, qs=qs)
        qv.delete()
        live_after_first_delete = list(QuantumVariable.live_qvs)
        qv.delete()

        _assert_deleted(qv)
        assert sum(existing.name == qv.name for existing in qs.deleted_qv_list) == 1
        assert QuantumVariable.live_qvs == live_after_first_delete

    # In tracing mode is_deleted() cannot be evaluated, so delete() has no early
    # return and the second call fails to find the variable in the session.
    def test_delete_twice_in_tracing_mode_raises(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            qs = qv.qs
            qv.delete()
            live_after_first_delete = list(QuantumVariable.live_qvs)
            with pytest.raises(Exception, match="non existent quantum variable"):
                qv.delete()
            _assert_deleted(qv)
            assert sum(existing.name == qv.name for existing in qs.deleted_qv_list) == 1
            assert QuantumVariable.live_qvs == live_after_first_delete
            return 0

        main()


class TestDeleteVerify:
    """Test the ``verify`` option of :meth:`QuantumVariable.delete`."""

    def test_delete_verify_accepts_uncomputed_qubits(self):
        qv = QuantumVariable(2)
        x(qv[0])
        h(qv[1])
        x(qv[0])
        h(qv[1])
        _assert_not_deleted(qv)
        qv.delete(verify=True)
        _assert_deleted(qv)

    @pytest.mark.parametrize(
        "gate",
        [pytest.param(x, id="ket_one"), pytest.param(h, id="ket_plus")],
    )
    def test_delete_verify_rejects_nonzero_state_and_keeps_variable(self, gate):
        qv = QuantumVariable(2)
        gate(qv[1])
        with pytest.raises(Exception, match=r"not in \|0> state"):
            qv.delete(verify=True, recompute=True)

        _assert_not_deleted(qv)
        assert not any(getattr(qb, "recompute", False) for qb in qv.reg)

    def test_delete_verify_in_tracing_mode_raises_and_keeps_variable(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            with pytest.raises(TracingModeError, match="verify deletion in tracing mode"):
                qv.delete(verify=True)
            _assert_not_deleted(qv)
            return 0

        main()


class TestDeleteRecompute:
    """Test the ``recompute`` option of :meth:`QuantumVariable.delete`."""

    @pytest.mark.parametrize("recompute", [True, False])
    def test_delete_recompute_marks_qubits(self, recompute):
        qv = QuantumVariable(2)
        qv.delete(recompute=recompute)
        _assert_deleted(qv)
        assert all(getattr(qb, "recompute", False) == recompute for qb in qv.reg)


class TestTracingModeError:
    """Test that operations unsupported in tracing mode raise :class:`TracingModeError`."""

    def test_get_measurement_in_tracing_mode_raises(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            with pytest.raises(TracingModeError, match="Tried to get measurement of a QuantumVariable in tracing mode"):
                qv.get_measurement()
            return 0

        main()

    def test_uncompute_in_tracing_mode_raises_and_keeps_variable(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            with pytest.raises(TracingModeError, match="Tried to uncompute a QuantumVariable in tracing mode"):
                qv.uncompute()
            _assert_not_deleted(qv)
            return 0

        main()

    def test_static_iteration_in_tracing_mode_raises(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            with pytest.raises(
                TracingModeError, match="Tried to perform a static iteration on a dynamic QuantumVariable"
            ):
                iter(qv)
            return 0

        main()

    def test_init_from_in_tracing_mode_raises(self):
        @jaspify
        def main():
            source = QuantumVariable(2)
            target = QuantumVariable(2)
            with pytest.raises(
                TracingModeError, match="Tried to initialize a QuantumVariable from another in tracing mode"
            ):
                target.init_from(source)
            return 0

        main()

    def test_duplicate_with_init_in_tracing_mode_raises(self):
        @jaspify
        def main():
            qv = QuantumVariable(2)
            with pytest.raises(
                TracingModeError, match="Tried to initialize a QuantumVariable from another in tracing mode"
            ):
                qv.duplicate(init=True)
            return 0

        main()
