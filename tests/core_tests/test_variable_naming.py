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

import pytest

from qrisp import QuantumSession, QuantumVariable, QuantumVariableNamingError


# Utilized to fail code-introspection when generating name.
# The bare `return` (not an assignment) is one frame deeper than a normal call
# site, so introspection finds no `var = Quantum...` line and name generation
# falls back.
def _make_unnamed_qv(qs: QuantumSession):
    return QuantumVariable(2, qs=qs)


@pytest.fixture(autouse=True)
def _isolated_quantum_variable_naming_state():
    """Save/reset/restore QuantumVariable's naming class state around each test.

    name_tracker, live_qvs and creation_counter live on the QuantumVariable class
    itself, shared process-wide across every QuantumSession, so without this a
    test's result depends on what other tests happened to run first.
    """
    original_name_tracker = QuantumVariable.name_tracker
    original_live_qvs = QuantumVariable.live_qvs
    original_creation_counter = QuantumVariable.creation_counter
    QuantumVariable.name_tracker = {}
    QuantumVariable.live_qvs = []
    QuantumVariable.creation_counter = 0
    try:
        yield
    finally:
        QuantumVariable.name_tracker = original_name_tracker
        QuantumVariable.live_qvs = original_live_qvs
        QuantumVariable.creation_counter = original_creation_counter


def test_name_generation_explicit_fresh_is_fixed():
    qs = QuantumSession()
    placeholder = QuantumVariable(1, qs=qs)
    name, is_fixed_name = qs.generate_name("alice", placeholder, 0)
    assert name == "alice"
    assert is_fixed_name is True


def test_name_generation_explicit_collision_raises():
    qs = QuantumSession()
    placeholder = QuantumVariable(1, qs=qs)
    _ = QuantumVariable(1, qs=qs, name="alice")
    with pytest.raises(QuantumVariableNamingError):
        _ = qs.generate_name("alice", placeholder, 0)


def test_name_generation_wildcard_fresh_is_not_fixed():
    qs = QuantumSession()
    placeholder = QuantumVariable(1, qs=qs)
    name, is_fixed_name = qs.generate_name("alice*", placeholder, 0)
    assert name == "alice"
    assert is_fixed_name is False


def test_name_generation_wildcard_collision_suffix_starts_at_creation_counter():
    qs = QuantumSession()
    placeholder = QuantumVariable(1, qs=qs)
    _ = QuantumVariable(1, qs=qs, name="alice")
    # Bump the global counter with a few extra QuantumVariables so the suffix
    # wouldn't coincidentally look 0-based.
    for _ in range(3):
        _ = QuantumVariable(1, qs=qs, name="filler*")
    expected_suffix = QuantumVariable.creation_counter
    name, is_fixed_name = qs.generate_name("alice*", placeholder, 0)
    assert name == f"alice_{expected_suffix}"
    assert is_fixed_name is False


def test_name_generation_duplicated_name_fresh_is_not_fixed():
    qs = QuantumSession()
    placeholder = QuantumVariable(1, qs=qs)
    name, is_fixed_name = qs.generate_name("alice", placeholder, 0, is_duplicated_name=True)
    assert name == "alice_dupl"
    assert is_fixed_name is False


def test_name_generation_duplicated_name_collision_suffix_starts_at_creation_counter():
    qs = QuantumSession()
    placeholder = QuantumVariable(1, qs=qs)
    _ = QuantumVariable(1, qs=qs, name="alice_dupl")
    # Bump the global counter with a few extra QuantumVariables so the suffix
    # wouldn't coincidentally look 0-based.
    for i in range(3):
        _ = QuantumVariable(1, qs=qs, name=f"filler_{i}")
    expected_suffix = QuantumVariable.creation_counter
    name, is_fixed_name = qs.generate_name("alice", placeholder, 0, is_duplicated_name=True)
    assert name == f"alice_dupl_{expected_suffix}"
    assert is_fixed_name is False


def test_name_generation_introspection_infers_python_variable_name():
    qs = QuantumSession()
    qv = QuantumVariable(2, qs=qs)
    assert qv.name == "qv"
    assert qv.is_fixed_name is False


# Code introspection always numbers its suffix starting from 0, independent of
# how many other QuantumVariables (same or different names) already exist.
def test_name_generation_introspection_collision_suffix_starts_at_zero():
    qs = QuantumSession()
    qv = QuantumVariable(2, qs=qs)
    first_qv = qv
    _ = QuantumVariable(1, qs=qs, name="filler_a")
    _ = QuantumVariable(1, qs=qs, name="filler_b")
    _ = QuantumVariable(1, qs=qs, name="filler*")
    assert QuantumVariable.creation_counter > 0
    qv = QuantumVariable(2, qs=qs)
    assert first_qv.name == "qv"
    assert qv.name == "qv_0"
    assert qv.is_fixed_name is False


# get_unique_name() is called when introspection fails.


def test_name_generation_falls_back_when_introspection_fails():
    qs = QuantumSession()
    next_number = QuantumVariable.name_tracker.get("qv", 0)
    quantum_variable = _make_unnamed_qv(qs)
    assert quantum_variable.name == f"qv_{next_number}"
    assert quantum_variable.is_fixed_name is False


def test_name_generation_falls_back_returns_sequential_names():
    qs = QuantumSession()
    next_number = QuantumVariable.name_tracker.get("qv", 0)
    first_qv = _make_unnamed_qv(qs)
    second_qv = _make_unnamed_qv(qs)
    assert first_qv.name == f"qv_{next_number}"
    assert second_qv.name == f"qv_{next_number + 1}"


# On collision, get_unique_name() appends "_0" to the colliding candidate
# rather than incrementing a numerical suffix.
def test_name_generation_falls_back_retries_past_a_live_variables_name():
    qs = QuantumSession()
    next_number = QuantumVariable.name_tracker.get("qv", 0)
    colliding_name = f"qv_{next_number}"
    _ = QuantumVariable(1, qs=qs, name=colliding_name)
    placeholder = QuantumVariable(1, qs=qs)
    name, is_fixed_name = qs.generate_name(None, placeholder, 0)
    assert name == f"{colliding_name}_0"
    assert is_fixed_name is False


# .delete() drops the name from live_qvs (global) but keeps it in
# deleted_qv_list (session-local), so get_unique_name() alone can miss it.
# generate_name's outer retry then calls get_unique_name() again, getting a
# plain increment instead of the nested suffix from the test above.
def test_name_generation_falls_back_retries_past_a_deleted_variables_name():
    qs = QuantumSession()
    next_number = QuantumVariable.name_tracker.get("qv", 0)
    blocker = QuantumVariable(1, qs=qs, name=f"qv_{next_number}")
    blocker.delete()
    placeholder = QuantumVariable(1, qs=qs)
    name, is_fixed_name = qs.generate_name(None, placeholder, 0)
    assert name == f"qv_{next_number + 1}"
    assert is_fixed_name is False
