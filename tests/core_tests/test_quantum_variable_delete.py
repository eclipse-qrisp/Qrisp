"""********************************************************************************
# Copyright (c) 2026 the Qrisp authors
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# with the GNU Classpath Exception which is
# available at https://www.gnu.org/software/classpath/license.html.
#
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import pytest

from qrisp import QuantumSession, QuantumVariable, h, x
from qrisp.jasp import jaspify
from qrisp.jasp.tracing_logic.tracing_quantum_session import TracingModeError


# QuantumVariable.__eq__ builds a QuantumBool, so plain `in` is not a reliable
# membership test; compare by identity instead.
def _contains(qvs: list[QuantumVariable], qv: QuantumVariable) -> bool:
    return any(existing is qv for existing in qvs)


def _dealloc_count(qs: QuantumSession) -> int:
    return sum(instr.op.name == "qb_dealloc" for instr in qs.data)


def _is_live(qv: QuantumVariable) -> bool:
    return any(ref() is qv for ref in QuantumVariable.live_qvs)


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


def test_delete_frees_qubits_and_unregisters_variable():
    qs = QuantumSession()
    qv = QuantumVariable(3, qs=qs)
    qv.delete()

    assert qv.is_deleted()
    assert all(not qb.allocated for qb in qv.reg)
    assert qv.name not in [existing.name for existing in qs.qv_list]
    assert _contains(qs.deleted_qv_list, qv)
    assert not _is_live(qv)
    assert _dealloc_count(qs) == 3


def test_delete_leaves_other_variables_untouched():
    qs = QuantumSession()
    kept = QuantumVariable(2, qs=qs)
    deleted = QuantumVariable(2, qs=qs)
    deleted.delete()

    assert not kept.is_deleted()
    assert _contains(qs.qv_list, kept)
    assert _is_live(kept)
    assert not _is_live(deleted)


def test_delete_twice_in_static_mode_is_a_no_op():
    qs = QuantumSession()
    qv = QuantumVariable(2, qs=qs)
    qv.delete()
    qv.delete()

    assert _dealloc_count(qs) == 2
    assert sum(existing is qv for existing in qs.deleted_qv_list) == 1


# In tracing mode is_deleted() cannot be evaluated, so delete() has no early
# return and the second call fails to find the variable in the session.
def test_delete_twice_in_tracing_mode_raises():
    @jaspify
    def main():
        qv = QuantumVariable(2)
        qv.delete()
        qv.delete()
        return 0

    with pytest.raises(Exception, match="non existent quantum variable"):
        main()


def test_operation_on_deleted_qubit_raises():
    qv = QuantumVariable(2)
    qv.delete()
    with pytest.raises(Exception, match="unallocated qubit"):
        x(qv[0])


def test_delete_verify_accepts_uncomputed_qubits():
    qv = QuantumVariable(2)
    x(qv[0])
    h(qv[1])
    x(qv[0])
    h(qv[1])
    qv.delete(verify=True)
    assert qv.is_deleted()


@pytest.mark.parametrize(
    "gate",
    [pytest.param(x, id="basis_state"), pytest.param(h, id="superposition")],
)
def test_delete_verify_rejects_nonzero_state_and_keeps_variable(gate):
    qs = QuantumSession()
    qv = QuantumVariable(2, qs=qs)
    gate(qv[1])
    with pytest.raises(Exception, match=r"not in \|0> state"):
        qv.delete(verify=True)

    assert not qv.is_deleted()
    assert _contains(qs.qv_list, qv)
    assert _is_live(qv)
    assert _dealloc_count(qs) == 0


def test_delete_recompute_marks_qubits():
    qv = QuantumVariable(2)
    qv.delete(recompute=True)
    assert all(qb.recompute for qb in qv.reg)


def test_delete_without_recompute_leaves_qubits_unmarked():
    qv = QuantumVariable(2)
    qv.delete()
    assert not any(getattr(qb, "recompute", False) for qb in qv.reg)


def test_delete_in_tracing_mode_unregisters_variable():
    observed = {}

    @jaspify
    def main():
        qv = QuantumVariable(2)
        qv.delete()
        observed["in_qv_list"] = _contains(qv.qs.qv_list, qv)
        observed["in_deleted_qv_list"] = _contains(qv.qs.deleted_qv_list, qv)
        observed["is_live"] = _is_live(qv)
        return 0

    main()
    assert observed["in_qv_list"] is False
    assert observed["in_deleted_qv_list"] is True
    assert observed["is_live"] is False


def test_delete_verify_in_tracing_mode_raises():
    @jaspify
    def main():
        qv = QuantumVariable(2)
        qv.delete(verify=True)
        return 0

    with pytest.raises(TracingModeError, match="verify deletion in tracing mode"):
        main()
