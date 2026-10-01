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

"""Tests for the QuantumSession class in qrisp.core.quantum_session."""

import weakref
from dataclasses import dataclass

import pytest

from qrisp import QuantumSession, QuantumVariable, QuantumVariableNamingError


@dataclass
class _SessionState:
    qv_names: list[str]
    deleted_qv_names: list[str]
    qubit_count: int
    live_qvs: list[weakref.ref]
    creation_counter: int


def _capture_state(qs: QuantumSession) -> _SessionState:
    return _SessionState(
        qv_names=[existing.name for existing in qs.qv_list],
        deleted_qv_names=[existing.name for existing in qs.deleted_qv_list],
        qubit_count=len(qs.qubits),
        live_qvs=list(QuantumVariable.live_qvs),
        creation_counter=QuantumVariable.creation_counter,
    )


def _assert_registered(qv: QuantumVariable, qs: QuantumSession, before: _SessionState) -> None:
    assert [existing.name for existing in qs.qv_list] == [*before.qv_names, qv.name]
    assert [existing.name for existing in qs.deleted_qv_list] == before.deleted_qv_names
    assert len(QuantumVariable.live_qvs) == len(before.live_qvs) + 1
    assert QuantumVariable.live_qvs[-1]() is qv
    assert qv.creation_time == before.creation_counter
    assert QuantumVariable.creation_counter == before.creation_counter + 1


def _assert_unchanged(qs: QuantumSession, before: _SessionState) -> None:
    assert [existing.name for existing in qs.qv_list] == before.qv_names
    assert [existing.name for existing in qs.deleted_qv_list] == before.deleted_qv_names
    assert len(qs.qubits) == before.qubit_count
    assert QuantumVariable.live_qvs == before.live_qvs
    assert QuantumVariable.creation_counter == before.creation_counter


@pytest.fixture(autouse=True)
def _isolated_quantum_variable_class_state():
    """Give each test empty QuantumVariable class state and restore it afterwards.

    live_qvs and creation_counter live on the QuantumVariable class and are shared
    process-wide, so without this, other tests would leak into the checks.
    """
    original_live_qvs = QuantumVariable.live_qvs
    original_creation_counter = QuantumVariable.creation_counter
    QuantumVariable.live_qvs = []
    QuantumVariable.creation_counter = 0
    yield
    QuantumVariable.live_qvs = original_live_qvs
    QuantumVariable.creation_counter = original_creation_counter


class TestRegisterQv:
    """Test :meth:`QuantumSession.register_qv`."""

    def test_register_qv_with_size_allocates_and_tracks_variable(self):
        qv = QuantumVariable(1, qs=QuantumSession(), name="alice")
        qs = QuantumSession()
        before_state = _capture_state(qs)

        qs.register_qv(qv, 3)

        _assert_registered(qv, qs, before_state)
        assert all(qb in qs.qubits for qb in qv.reg)
        assert [qb.identifier for qb in qv.reg] == ["alice.0", "alice.1", "alice.2"]
        assert len(qs.qubits) == before_state.qubit_count + 3

    def test_register_qv_without_size_keeps_existing_register(self):
        qv = QuantumVariable(2, qs=QuantumSession(), name="alice")
        original_reg = qv.reg
        qs = QuantumSession()
        before_state = _capture_state(qs)

        qs.register_qv(qv, None)

        _assert_registered(qv, qs, before_state)
        assert qv.reg is original_reg
        assert len(qs.qubits) == before_state.qubit_count

    @pytest.mark.parametrize("delete_existing", [False, True], ids=["active", "deleted"])
    def test_register_qv_name_collision_raises_without_side_effects(self, delete_existing):
        qs = QuantumSession()
        existing_qv = QuantumVariable(1, qs=qs, name="alice")
        if delete_existing:
            existing_qv.delete()
        new_qv = QuantumVariable(1, qs=QuantumSession(), name="alice")
        original_reg = new_qv.reg
        before_state = _capture_state(qs)

        with pytest.raises(QuantumVariableNamingError, match="Variable name alice already exists"):
            qs.register_qv(new_qv, 2)

        _assert_unchanged(qs, before_state)
        assert new_qv.reg is original_reg
