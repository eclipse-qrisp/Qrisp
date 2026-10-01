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

"""Tests the different cases for variable naming."""

import pytest

from qrisp import QuantumSession, QuantumVariable, QuantumVariableNamingError
from qrisp.core.session_merging_tools import resolve_naming_collisions
from qrisp.jasp import jaspify


# Utilized to fail code-introspection when generating name.
# The bare `return` (not an assignment) is one frame deeper than a normal call
# site, so introspection finds no `var = Quantum...` line and name generation
# falls back.
def _make_unnamed_qv(qs: QuantumSession):
    return QuantumVariable(2, qs=qs)


@pytest.fixture(autouse=True)
def _isolated_quantum_variable_naming_state():
    """Save, reset, and restore QuantumVariable's naming class state around each test.

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
    yield
    QuantumVariable.name_tracker = original_name_tracker
    QuantumVariable.live_qvs = original_live_qvs
    QuantumVariable.creation_counter = original_creation_counter


class TestRequestedName:
    """Test :meth:`QuantumSession.generate_name` with an explicit, wildcard or duplicated name."""

    @pytest.mark.parametrize(
        ("requested_name", "is_duplicated_name", "expected_name", "expected_is_fixed_name"),
        [
            pytest.param("alice", False, "alice", True, id="explicit"),
            pytest.param("alice*", False, "alice", False, id="wildcard"),
            pytest.param("alice", True, "alice_dupl", False, id="duplicated"),
        ],
    )
    def test_name_generation_without_collision(
        self, requested_name: str, is_duplicated_name: bool, expected_name: str, expected_is_fixed_name: bool
    ):
        qs = QuantumSession()
        placeholder = QuantumVariable(1, qs=qs)
        name, is_fixed_name = qs.generate_name(requested_name, placeholder, 0, is_duplicated_name=is_duplicated_name)
        assert name == expected_name
        assert is_fixed_name is expected_is_fixed_name

    def test_name_generation_explicit_collision_raises(self):
        qs = QuantumSession()
        placeholder = QuantumVariable(1, qs=qs)
        _ = QuantumVariable(1, qs=qs, name="alice")
        with pytest.raises(QuantumVariableNamingError):
            _ = qs.generate_name("alice", placeholder, 0)

    @pytest.mark.parametrize(
        ("requested_name", "is_duplicated_name", "colliding_name"),
        [
            pytest.param("alice*", False, "alice", id="wildcard"),
            pytest.param("alice", True, "alice_dupl", id="duplicated"),
        ],
    )
    def test_name_generation_collision_suffix_starts_at_creation_counter(
        self, requested_name: str, is_duplicated_name: bool, colliding_name: str
    ):
        qs = QuantumSession()
        placeholder = QuantumVariable(1, qs=qs)
        _ = QuantumVariable(1, qs=qs, name=colliding_name)
        # Bump the global counter with a few extra QuantumVariables so the suffix
        # wouldn't coincidentally look 0-based.
        for i in range(3):
            _ = QuantumVariable(1, qs=qs, name=f"filler_{i}")
        expected_suffix = QuantumVariable.creation_counter
        name, is_fixed_name = qs.generate_name(requested_name, placeholder, 0, is_duplicated_name=is_duplicated_name)
        assert name == f"{colliding_name}_{expected_suffix}"
        assert is_fixed_name is False


class TestIntrospectedName:
    """Test names inferred from the Python variable a QuantumVariable is assigned to."""

    def test_name_generation_introspection_infers_python_variable_name(self):
        qs = QuantumSession()
        qv = QuantumVariable(2, qs=qs)
        assert qv.name == "qv"
        assert qv.is_fixed_name is False

    # Code introspection always numbers its suffix starting from 0, independent of
    # how many other QuantumVariables (same or different names) already exist.
    def test_name_generation_introspection_collision_suffix_starts_at_zero(self):
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
class TestFallbackName:
    """Test the generated names used when no name is given and introspection fails."""

    def test_name_generation_falls_back_returns_sequential_names(self):
        qs = QuantumSession()
        next_number = QuantumVariable.name_tracker.get("qv", 0)
        first_qv = _make_unnamed_qv(qs)
        second_qv = _make_unnamed_qv(qs)
        assert first_qv.name == f"qv_{next_number}"
        assert first_qv.is_fixed_name is False
        assert second_qv.name == f"qv_{next_number + 1}"
        assert second_qv.is_fixed_name is False

    # On collision, get_unique_name() appends "_0" to the colliding candidate
    # rather than incrementing a numerical suffix.
    def test_name_generation_falls_back_retries_past_a_live_variables_name(self):
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
    def test_name_generation_falls_back_retries_past_a_deleted_variables_name(self):
        qs = QuantumSession()
        next_number = QuantumVariable.name_tracker.get("qv", 0)
        blocker = QuantumVariable(1, qs=qs, name=f"qv_{next_number}")
        blocker.delete()
        placeholder = QuantumVariable(1, qs=qs)
        name, is_fixed_name = qs.generate_name(None, placeholder, 0)
        assert name == f"qv_{next_number + 1}"
        assert is_fixed_name is False


def _make_colliding_pair(qv_0_fixed: bool, qv_1_fixed: bool, qv_0_newer: bool):
    """Create same-named QuantumVariables in two separate QuantumSessions.

    Returns ``(qs_0, qv_0, qs_1, qv_1)``; which variable is created first
    determines which one has the older ``creation_time``.
    """
    qs_0, qs_1 = QuantumSession(), QuantumSession()
    name_0 = "alice" if qv_0_fixed else "alice*"
    name_1 = "alice" if qv_1_fixed else "alice*"
    if qv_0_newer:
        qv_1 = QuantumVariable(1, qs=qs_1, name=name_1)
        qv_0 = QuantumVariable(1, qs=qs_0, name=name_0)
    else:
        qv_0 = QuantumVariable(1, qs=qs_0, name=name_0)
        qv_1 = QuantumVariable(1, qs=qs_1, name=name_1)
    return qs_0, qv_0, qs_1, qv_1


class TestResolveNamingCollisions:
    """Test :func:`resolve_naming_collisions` for same-named variables in two sessions."""

    # The newer variable is renamed unless its name is fixed, in which case the
    # older one is renamed instead. A fixed name is never renamed.
    @pytest.mark.parametrize(
        ("qv_0_fixed", "qv_1_fixed", "qv_0_newer", "renamed"),
        [
            pytest.param(False, False, True, "qv_0", id="neither_fixed-qv_0_newer"),
            pytest.param(False, False, False, "qv_1", id="neither_fixed-qv_1_newer"),
            pytest.param(False, True, True, "qv_0", id="qv_1_fixed-qv_0_newer"),
            pytest.param(False, True, False, "qv_0", id="qv_1_fixed-qv_1_newer"),
            pytest.param(True, False, True, "qv_1", id="qv_0_fixed-qv_0_newer"),
            pytest.param(True, False, False, "qv_1", id="qv_0_fixed-qv_1_newer"),
        ],
    )
    def test_resolve_naming_collisions_renames_expected_variable(
        self, qv_0_fixed: bool, qv_1_fixed: bool, qv_0_newer: bool, renamed: str
    ):
        qs_0, qv_0, qs_1, qv_1 = _make_colliding_pair(qv_0_fixed, qv_1_fixed, qv_0_newer)
        resolve_naming_collisions(qs_0, qs_1)
        renamed_qv, kept_qv = (qv_0, qv_1) if renamed == "qv_0" else (qv_1, qv_0)
        assert kept_qv.name == "alice"
        assert renamed_qv.name == "alice_1"
        assert renamed_qv.reg[0].identifier == "alice_1.0"

    def test_resolve_naming_collisions_both_fixed_raises(self):
        qs_0, _, qs_1, _ = _make_colliding_pair(qv_0_fixed=True, qv_1_fixed=True, qv_0_newer=True)
        with pytest.raises(Exception, match="identically named QuantumVariables alice"):
            resolve_naming_collisions(qs_0, qs_1)


class TestDuplicateNaming:
    """Test the names :meth:`QuantumVariable.duplicate` gives in tracing mode."""

    @pytest.mark.parametrize(
        ("requested_name", "expected_name", "expected_is_fixed_name"),
        [
            pytest.param(None, "alice_dupl", False, id="no_name"),
            pytest.param("bob", "bob", True, id="explicit_name"),
            pytest.param("bob*", "bob", False, id="wildcard_name"),
        ],
    )
    def test_duplicate_naming_in_tracing_mode(
        self, requested_name: str | None, expected_name: str, expected_is_fixed_name: bool
    ):
        @jaspify
        def main():
            qv = QuantumVariable(2, name="alice")
            duplicate = qv.duplicate(name=requested_name)
            assert duplicate.name == expected_name
            assert duplicate.is_fixed_name is expected_is_fixed_name
            return 0

        main()
