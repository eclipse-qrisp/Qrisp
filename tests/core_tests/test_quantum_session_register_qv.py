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

from qrisp import QuantumSession, QuantumVariable, QuantumVariableNamingError


# QuantumVariable.__eq__ builds a QuantumBool, so plain `in` is not a reliable
# membership test; compare by identity instead.
def _contains(qvs: list[QuantumVariable], qv: QuantumVariable) -> bool:
    return any(existing is qv for existing in qvs)


def _is_live(qv: QuantumVariable) -> bool:
    return any(ref() is qv for ref in QuantumVariable.live_qvs)


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


# QuantumVariable.__init__ already registers into its own session, so the tests
# register an existing variable into a second session, as duplicate() does.


def test_register_qv_with_size_allocates_and_tracks_variable():
    qv = QuantumVariable(1, qs=QuantumSession(), name="alice")
    qs = QuantumSession()
    counter_before = QuantumVariable.creation_counter

    qs.register_qv(qv, 3)

    assert len(qv.reg) == 3
    assert all(qb in qs.qubits for qb in qv.reg)
    assert [qb.identifier for qb in qv.reg] == ["alice.0", "alice.1", "alice.2"]
    assert _contains(qs.qv_list, qv)
    assert _is_live(qv)
    assert qv.creation_time == counter_before
    assert QuantumVariable.creation_counter == counter_before + 1


def test_register_qv_without_size_keeps_existing_register():
    qv = QuantumVariable(2, qs=QuantumSession(), name="alice")
    original_reg = qv.reg
    qs = QuantumSession()
    qubit_count_before = len(qs.qubits)

    qs.register_qv(qv, None)

    assert qv.reg is original_reg
    assert len(qs.qubits) == qubit_count_before
    assert _contains(qs.qv_list, qv)


@pytest.mark.parametrize("delete_existing", [False, True], ids=["active", "deleted"])
def test_register_qv_name_collision_raises_without_side_effects(delete_existing):
    qs = QuantumSession()
    existing = QuantumVariable(1, qs=qs, name="alice")
    if delete_existing:
        existing.delete()
    newcomer = QuantumVariable(1, qs=QuantumSession(), name="alice")
    original_reg = newcomer.reg
    qv_list_before = list(qs.qv_list)
    qubit_count_before = len(qs.qubits)
    counter_before = QuantumVariable.creation_counter

    with pytest.raises(QuantumVariableNamingError, match="alice"):
        qs.register_qv(newcomer, 2)

    assert newcomer.reg is original_reg
    assert qs.qv_list == qv_list_before
    assert len(qs.qubits) == qubit_count_before
    assert QuantumVariable.creation_counter == counter_before
