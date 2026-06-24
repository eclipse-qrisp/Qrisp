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

import types

import pytest


from qrisp import *
from qrisp.jasp import *
from qrisp.jasp.tracing_logic.tracing_quantum_session import (
    _instance,
)


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------


class _StaleAbs:
    """Fake abstract quantum state whose _trace never equals the live JAX trace."""

    _trace = object()


@pytest.fixture
def session():
    """Return a fresh, uninitialised TracingQuantumSession."""
    return TracingQuantumSession()


@pytest.fixture
def stale_session():
    """Return a TracingQuantumSession whose abs_qst is permanently out of scope."""
    s = TracingQuantumSession()
    s.abs_qst = _StaleAbs()
    return s


# NOTE: It is not really a singleton since it is technically possible
# to create two distinct instances of TracingQuantumSession,
# but the module-level instance is the only one that should be used in practice.
class TestSingleton:
    """TracingQuantumSession exposes a module-level singleton via get_instance()."""

    def test_get_instance_is_idempotent(self):
        """Repeated calls to get_instance() return the identical object."""
        assert TracingQuantumSession.get_instance() is TracingQuantumSession.get_instance()

    def test_get_instance_returns_module_level_instance(self):
        """get_instance() returns the _instance created at import time."""
        assert TracingQuantumSession.get_instance() is _instance


def test_init_default_state(session):
    """All instance attributes start at their zero/empty defaults after __init__."""
    assert session.abs_qst is None
    assert session.qv_list == []
    assert session.deleted_qv_list == []
    assert session.qubit_cache is None
    assert session.abs_qst_stack == []
    assert session.qubit_cache_stack == []
    assert session.qv_stack == []


class TestTracingLifecycle:
    """start_tracing and conclude_tracing correctly push and pop all tracing state."""

    def test_start_tracing_sets_abs_qst(self, session):
        """start_tracing() makes the provided abs_qst the active one."""
        fake = object()
        session.start_tracing(fake)
        assert session.abs_qst is fake
        session.conclude_tracing()

    def test_start_tracing_initialises_empty_qubit_cache(self, session):
        """start_tracing() resets qubit_cache to a fresh empty dict."""
        session.start_tracing(object())
        assert session.qubit_cache == {} and isinstance(session.qubit_cache, dict)
        session.conclude_tracing()

    def test_start_tracing_pushes_previous_abs_qst(self, session):
        """The previous abs_qst is saved on the stack before being replaced."""
        session.abs_qst = sentinel = object()
        session.start_tracing(object())
        assert session.abs_qst_stack[-1] is sentinel
        session.conclude_tracing()

    def test_start_tracing_pushes_previous_qubit_cache(self, session):
        """The previous qubit_cache is saved on the stack before being replaced."""
        session.qubit_cache = prev = {"k": "v"}
        session.start_tracing(object())
        assert session.qubit_cache_stack[-1] is prev
        session.conclude_tracing()

    def test_start_tracing_resets_qv_lists(self, session):
        """Both qv_list and deleted_qv_list are cleared to [] on start_tracing()."""
        session.qv_list = ["a"]
        session.deleted_qv_list = ["b"]
        session.start_tracing(object())
        assert session.qv_list == [] and session.deleted_qv_list == []
        session.conclude_tracing()

    def test_start_tracing_pushes_old_qv_lists_onto_stack(self, session):
        """The old qv_list and deleted_qv_list are pushed onto qv_stack as a pair."""
        session.qv_list = old_qv = ["x"]
        session.deleted_qv_list = old_del = ["y"]
        session.start_tracing(object())
        pushed_qv, pushed_del = session.qv_stack[-1]
        assert pushed_qv is old_qv and pushed_del is old_del
        session.conclude_tracing()

    def test_conclude_tracing_returns_active_abs_qst(self, session):
        """conclude_tracing() returns the abs_qst that was active in the just-closed scope."""
        inner = object()
        session.start_tracing(inner)
        assert session.conclude_tracing() is inner

    def test_conclude_tracing_restores_abs_qst(self, session):
        """conclude_tracing() pops abs_qst back to the value held before start_tracing."""
        session.abs_qst = outer = object()
        session.start_tracing(object())
        session.conclude_tracing()
        assert session.abs_qst is outer

    def test_conclude_tracing_restores_qubit_cache(self, session):
        """conclude_tracing() pops qubit_cache back to the value held before start_tracing."""
        session.qubit_cache = outer_cache = {"outer": True}
        session.start_tracing(object())
        session.conclude_tracing()
        assert session.qubit_cache is outer_cache

    def test_conclude_tracing_restores_qv_lists(self, session):
        """conclude_tracing() restores both qv_list and deleted_qv_list from the stack."""
        session.qv_list = outer_qvs = ["a"]
        session.deleted_qv_list = outer_del = ["b"]
        session.start_tracing(object())
        session.conclude_tracing()
        assert session.qv_list is outer_qvs and session.deleted_qv_list is outer_del

    def test_nested_tracing_pushes_and_pops_all_layers(self, session):
        """Two levels of start_tracing/conclude_tracing restore the full outer state."""
        outer, inner, innermost = object(), object(), object()
        session.abs_qst = outer

        session.start_tracing(inner)
        assert session.abs_qst is inner

        session.start_tracing(innermost)
        assert session.abs_qst is innermost
        assert session.abs_qst_stack[-1] is inner

        session.conclude_tracing()
        assert session.abs_qst is inner

        session.conclude_tracing()
        assert session.abs_qst is outer


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: s.append(None), id="append"),
        pytest.param(lambda s: s.register_qv(None, None), id="register_qv"),
        pytest.param(lambda s: s.delete_qv(None), id="delete_qv"),
    ],
)
def test_stale_abs_qst_raises_runtime_error(stale_session, call):
    """append, register_qv and delete_qv all raise RuntimeError when abs_qst is out of scope."""
    with pytest.raises(RuntimeError, match="Lost track"):
        call(stale_session)


class TestAppend:
    """append() dispatches to the correct code path based on qubits[0] type."""

    def test_raises_value_error_for_non_empty_clbits(self):
        """Passing non-empty clbits raises ValueError; JAX mode has no classical bits."""

        def circuit():
            TracingQuantumSession.get_instance().append(None, clbits=[1])

        with pytest.raises(ValueError, match="non-zero classical bits"):
            make_jaspr(circuit)()

    def test_converts_none_qubits_to_empty_tuple(self):
        """qubits=None is normalised to () before dispatch; ()[0] then raises IndexError."""

        def circuit():
            TracingQuantumSession.get_instance().append(None, qubits=None)

        with pytest.raises(IndexError):
            make_jaspr(circuit)()

    @pytest.mark.parametrize(
        "target_fn,expected",
        [
            pytest.param(lambda qf: qf, 3, id="quantum_variable"),
            pytest.param(lambda qf: qf.reg, 3, id="dynamic_qubit_array"),
            pytest.param(lambda qf: qf[0], 1, id="individual_qubit"),
        ],
    )
    def test_x_gate_dispatch_branches(self, target_fn, expected):
        """QuantumVariable/DynamicQubitArray enter branch 1; individual qubit falls through to bind.

        Expected result after X: 3 (both bits flipped) for QV/DQA, 1 (LSB only) for a single qubit.
        """

        @boolean_simulation
        def circuit():
            qf = QuantumFloat(2)
            x(target_fn(qf))
            return measure(qf)

        assert circuit() == expected

    def test_dispatches_list_branch(self):
        """A Python list as qubits[0] enters the list-of-qubits batch dispatch path."""

        @boolean_simulation
        def circuit():
            ctrl, tgt = QuantumFloat(1), QuantumFloat(1)
            x(ctrl[0])
            # cx([ctrl_qb], [tgt_qb]) → qubits[0] is a list → list branch
            cx([ctrl.reg[0]], [tgt.reg[0]])
            return measure(ctrl), measure(tgt)

        ctrl_res, tgt_res = circuit()
        assert ctrl_res == 1 and tgt_res == 1

    def test_dispatches_quantum_array_branch(self):
        """A QuantumArray as qubits[0] enters the QuantumArray flattening dispatch path."""

        @boolean_simulation
        def circuit():
            qa = QuantumArray(qtype=QuantumFloat(1), shape=(2,))
            x(qa)
            return measure(qa)

        assert circuit() is not None

    def test_raises_type_error_for_mixed_quantum_array_types(self):
        """Mixing a QuantumArray with a QuantumFloat raises TypeError in the QuantumArray branch."""

        def circuit():
            cx(QuantumArray(qtype=QuantumFloat(1), shape=(2,)), QuantumFloat(1))

        with pytest.raises(TypeError, match="mixed qubit"):
            make_jaspr(circuit)()

    def test_raises_value_error_for_quantum_arrays_with_differing_shapes(self):
        """Two QuantumArrays of incompatible shapes raise ValueError before any gate is applied."""

        def circuit():
            cx(
                QuantumArray(qtype=QuantumFloat(1), shape=(2,)),
                QuantumArray(qtype=QuantumFloat(1), shape=(3,)),
            )

        with pytest.raises(ValueError, match="differing shape"):
            make_jaspr(circuit)()


class TestRegisterQvAndRequestQubits:
    """register_qv records a QuantumVariable in the session; request_qubits allocates qubits."""

    def test_register_qv_adds_qv_to_list_and_sets_qs(self):
        """register_qv appends the QV to qv_list and sets qv.qs to the session."""
        captured = {}

        def circuit():
            qs = TracingQuantumSession.get_instance()
            before = len(qs.qv_list)
            qf = QuantumFloat(2)
            captured["added"] = len(qs.qv_list) == before + 1
            captured["in_list"] = any(qv.name == qf.name for qv in qs.qv_list)
            captured["qs_set"] = qf.qs is qs
            return measure(qf)

        make_jaspr(circuit)()
        assert captured["added"] and captured["in_list"] and captured["qs_set"]

    def test_register_qv_with_nonzero_size_allocates_dqa(self):
        """Passing a non-None size triggers qubit allocation and assigns a DynamicQubitArray."""
        captured = {}

        def circuit():
            qf = QuantumFloat(3)
            captured["is_dqa"] = isinstance(qf.reg, DynamicQubitArray)
            return measure(qf)

        make_jaspr(circuit)()
        assert captured["is_dqa"]

    def test_register_qv_with_none_size_skips_allocation(self):
        """size=None skips qubit allocation; the existing register is kept unchanged."""
        captured = {}

        def circuit():
            qs = TracingQuantumSession.get_instance()
            qf = QuantumFloat(2)
            original_reg = qf.reg
            qs.qv_list = [qv for qv in qs.qv_list if qv.name != qf.name]
            qs.register_qv(qf, None)
            captured["reg_unchanged"] = qf.reg is original_reg
            return measure(qf)

        make_jaspr(circuit)()
        assert captured["reg_unchanged"]

    def test_request_qubits_returns_dqa_and_updates_abs_qst(self):
        """request_qubits returns a DynamicQubitArray and produces a new abstract quantum state."""
        captured = {}

        def circuit():
            qs = TracingQuantumSession.get_instance()
            before = qs.abs_qst
            dqa = qs.request_qubits(3)
            captured["is_dqa"] = isinstance(dqa, DynamicQubitArray)
            captured["abs_qst_changed"] = qs.abs_qst is not before
            qs.clear_qubits(dqa)
            return 0

        make_jaspr(circuit)()
        assert captured["is_dqa"] and captured["abs_qst_changed"]


class TestDeleteQvAndClearQubits:
    """delete_qv removes a QV from the live list, frees its qubits, and records the deletion."""

    def test_raises_not_implemented_error_when_verify_true(self):
        """Verification during deletion is not supported in tracing mode."""

        def circuit():
            qs = TracingQuantumSession.get_instance()
            qs.delete_qv(QuantumFloat(2), verify=True)

        with pytest.raises(NotImplementedError, match="verify deletion"):
            make_jaspr(circuit)()

    def test_raises_value_error_for_nonexistent_qv(self):
        """Attempting to delete a QV not registered in the session raises ValueError."""
        mock_qv = types.SimpleNamespace(name="no_such_qv_xyz")

        def circuit():
            qs = TracingQuantumSession.get_instance()
            QuantumFloat(2)
            qs.delete_qv(mock_qv)

        with pytest.raises(ValueError, match="non existent"):
            make_jaspr(circuit)()

    def test_moves_qv_from_qv_list_to_deleted_qv_list(self):
        """delete_qv removes the QV from qv_list and appends it to deleted_qv_list."""
        captured = {}

        def circuit():
            qs = TracingQuantumSession.get_instance()
            qf = QuantumFloat(2)
            name = qf.name
            captured["before"] = any(qv.name == name for qv in qs.qv_list)
            qs.delete_qv(qf)
            captured["after"] = any(qv.name == name for qv in qs.qv_list)
            captured["in_deleted"] = any(qv.name == name for qv in qs.deleted_qv_list)
            return 0

        make_jaspr(circuit)()
        assert captured["before"] and not captured["after"] and captured["in_deleted"]

    def test_clear_qubits_is_exercised_end_to_end(self):
        """clear_qubits is called by delete_qv; verify the full deallocation path runs."""

        @boolean_simulation
        def circuit():
            qf = QuantumFloat(2)
            x(qf[0])
            return measure(qf)  # triggers delete_qv → clear_qubits

        assert circuit() == 1
