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

from collections import deque
from unittest.mock import MagicMock

import pytest

from qrisp import QuantumVariable
from qrisp.jasp.primitives.abstract_quantum_state import AbstractQuantumState
from qrisp.jasp.tracing_logic.tracing_quantum_session import (
    TracingQuantumSession,
    check_for_tracing_mode,
)


def _make_abs_qst() -> AbstractQuantumState:
    """Return a real AbstractQuantumState."""
    return AbstractQuantumState()


def _make_qv(size: int, name: str = "qv") -> QuantumVariable:
    """Return a real QuantumVariable registered in its own QuantumSession."""
    return QuantumVariable(size=size, name=name)


def _clear_stacks() -> None:
    """Drain all stacks between tests to avoid cross-test contamination."""
    TracingQuantumSession.abs_qst_stack.clear()
    TracingQuantumSession.qubit_cache_stack.clear()
    # qv_stack is instance-level; reset it on the singleton directly.
    TracingQuantumSession().qv_stack = []


class TestSingleton:
    def test_same_instance_on_repeated_calls(self):
        """Calling TracingQuantumSession() twice must return the same object."""
        assert TracingQuantumSession() is TracingQuantumSession()

    def test_get_instance_returns_singleton(self):
        """get_instance() must return the same object as direct construction."""
        session = TracingQuantumSession()
        TracingQuantumSession.tr_qs_instance = session
        assert TracingQuantumSession.get_instance() is session

    def test_get_instance_raises_when_released(self):
        """get_instance() must raise RuntimeError after release()."""
        TracingQuantumSession.release()
        with pytest.raises(RuntimeError, match="No active TracingQuantumSession"):
            TracingQuantumSession.get_instance()
        # Restore for subsequent tests
        TracingQuantumSession.tr_qs_instance = TracingQuantumSession()

    def test_release_clears_instance(self):
        """release() must set tr_qs_instance to None."""
        TracingQuantumSession.release()
        assert TracingQuantumSession.tr_qs_instance is None
        # Restore
        TracingQuantumSession.tr_qs_instance = TracingQuantumSession()


class TestTracingStack:
    def setup_method(self):
        _clear_stacks()
        self.session = TracingQuantumSession()
        self.session.abs_qst = None
        self.session.qubit_cache = None
        self.session.qv_list = []
        self.session.deleted_qv_list = []
        self.session.qv_stack = []

    def test_start_tracing_sets_abs_qst(self):
        qst = _make_abs_qst()
        self.session.start_tracing(qst)
        assert self.session.abs_qst is qst
        self.session.conclude_tracing()

    def test_conclude_tracing_returns_current_qst(self):
        qst = _make_abs_qst()
        self.session.start_tracing(qst)
        returned = self.session.conclude_tracing()
        assert returned is qst

    def test_conclude_tracing_restores_previous_qst(self):
        outer_qst = _make_abs_qst()
        inner_qst = _make_abs_qst()
        self.session.abs_qst = outer_qst

        self.session.start_tracing(inner_qst)
        self.session.conclude_tracing()

        assert self.session.abs_qst is outer_qst

    def test_conclude_tracing_restores_none_when_no_outer_qst(self):
        self.session.abs_qst = None
        self.session.start_tracing(_make_abs_qst())
        self.session.conclude_tracing()
        assert self.session.abs_qst is None

    def test_start_tracing_gives_fresh_qubit_cache(self):
        original_cache = {"q0": object()}
        self.session.qubit_cache = original_cache

        self.session.start_tracing(_make_abs_qst())
        assert self.session.qubit_cache == {}
        assert self.session.qubit_cache is not original_cache
        self.session.conclude_tracing()

    def test_conclude_tracing_restores_qubit_cache(self):
        original_cache = {"q0": object()}
        self.session.qubit_cache = original_cache

        self.session.start_tracing(_make_abs_qst())
        self.session.conclude_tracing()

        assert self.session.qubit_cache is original_cache

    def test_start_tracing_resets_qv_list(self):
        self.session.qv_list = [_make_qv(1, "existing")]
        self.session.start_tracing(_make_abs_qst())
        assert self.session.qv_list == []
        self.session.conclude_tracing()

    def test_conclude_tracing_restores_qv_list(self):
        outer_qv = _make_qv(1, "outer_qv")
        self.session.qv_list = [outer_qv]

        self.session.start_tracing(_make_abs_qst())
        self.session.qv_list.append(_make_qv(2, "inner_qv"))
        self.session.conclude_tracing()

        assert len(self.session.qv_list) == 1
        assert self.session.qv_list[0] is outer_qv

    def test_conclude_without_start_raises(self):
        """Popping an empty stack must raise IndexError."""
        _clear_stacks()
        with pytest.raises(IndexError):
            self.session.conclude_tracing()


class TestNestedTracing:
    def setup_method(self):
        _clear_stacks()
        self.session = TracingQuantumSession()
        self.session.abs_qst = None
        self.session.qubit_cache = None
        self.session.qv_list = []
        self.session.deleted_qv_list = []
        self.session.qv_stack = []

    def test_triple_nesting_restores_each_level(self):
        qst_a, qst_b, qst_c = _make_abs_qst(), _make_abs_qst(), _make_abs_qst()

        self.session.start_tracing(qst_a)
        assert self.session.abs_qst is qst_a

        self.session.start_tracing(qst_b)
        assert self.session.abs_qst is qst_b

        self.session.start_tracing(qst_c)
        assert self.session.abs_qst is qst_c

        returned_c = self.session.conclude_tracing()
        assert returned_c is qst_c
        assert self.session.abs_qst is qst_b

        returned_b = self.session.conclude_tracing()
        assert returned_b is qst_b
        assert self.session.abs_qst is qst_a

        returned_a = self.session.conclude_tracing()
        assert returned_a is qst_a
        assert self.session.abs_qst is None

    def test_stack_depth_reflects_nesting(self):
        """Class-level stack length should match nesting depth."""
        _clear_stacks()
        assert len(TracingQuantumSession.abs_qst_stack) == 0

        self.session.start_tracing(_make_abs_qst())
        assert len(TracingQuantumSession.abs_qst_stack) == 1

        self.session.start_tracing(_make_abs_qst())
        assert len(TracingQuantumSession.abs_qst_stack) == 2

        self.session.conclude_tracing()
        assert len(TracingQuantumSession.abs_qst_stack) == 1

        self.session.conclude_tracing()
        assert len(TracingQuantumSession.abs_qst_stack) == 0

    def test_qv_lists_are_isolated_per_nesting_level(self):
        """QVs registered at inner scope must not leak into the outer scope."""
        outer_qv = _make_qv(1, "outer")
        inner_qv = _make_qv(1, "inner")

        self.session.qv_list = [outer_qv]
        self.session.start_tracing(_make_abs_qst())

        self.session.qv_list.append(inner_qv)
        assert any(v is inner_qv for v in self.session.qv_list)

        self.session.conclude_tracing()

        # QuantumVariable.__eq__ returns a QuantumBool, not a plain bool, so
        # `in` / `not in` cannot be used on lists of real QuantumVariables.
        # Use identity checks via `is` throughout this file instead.
        assert not any(v is inner_qv for v in self.session.qv_list)
        assert any(v is outer_qv for v in self.session.qv_list)


class TestClassLevelStackSurvivesReinit:
    """
    Guard against the regression introduced in PR #478 where moving
    abs_qst_stack to instance level caused conclude_tracing() to raise
    IndexError after any re-initialisation of the singleton, permanently
    preventing JAX from populating its JIT cache.
    """

    def setup_method(self):
        _clear_stacks()
        self.session = TracingQuantumSession()
        self.session.abs_qst = None
        self.session.qubit_cache = None
        self.session.qv_list = []
        self.session.deleted_qv_list = []
        self.session.qv_stack = []

    def test_stacks_are_class_attributes(self):
        """abs_qst_stack and qubit_cache_stack must live on the class, not the instance."""
        assert "abs_qst_stack" in TracingQuantumSession.__dict__
        assert "qubit_cache_stack" in TracingQuantumSession.__dict__

    def test_stacks_are_deques(self):
        assert isinstance(TracingQuantumSession.abs_qst_stack, deque)
        assert isinstance(TracingQuantumSession.qubit_cache_stack, deque)

    def test_qv_stack_is_wiped_by_reinit_bug(self):
        """
        Document a known secondary bug: qv_stack is instance-level and IS wiped
        by __init__, so conclude_tracing() will IndexError on self.qv_stack.pop()
        even when abs_qst_stack and qubit_cache_stack (class-level) survive fine.

        This test exists as a canary. If it starts *passing* unexpectedly, that
        means qv_stack was promoted to class-level (or otherwise guarded), which
        is the correct fix — and this test can then be replaced by a passing
        version matching test_conclude_tracing_still_works_after_reinit below.
        """
        self.session.start_tracing(_make_abs_qst())
        self.session.__init__()  # wipes self.qv_stack = []

        with pytest.raises(IndexError):
            self.session.conclude_tracing()

        # Restore a clean state for subsequent tests in this session.
        _clear_stacks()

    def test_abs_qst_stack_survives_reinit(self):
        """
        The class-level abs_qst_stack must not be wiped by __init__.
        This guards the core PR #478 fix: abs_qst_stack and qubit_cache_stack
        are class attributes and survive any re-initialisation of the instance.
        """
        self.session.start_tracing(_make_abs_qst())
        self.session.start_tracing(_make_abs_qst())
        depth_before = len(TracingQuantumSession.abs_qst_stack)

        self.session.__init__()  # only class attributes survive

        assert len(TracingQuantumSession.abs_qst_stack) == depth_before, (
            "abs_qst_stack was wiped by __init__. It must be a class-level "
            "attribute to survive singleton re-initialisation (PR #478 regression)."
        )

        # Restore: qv_stack was wiped by __init__ so replenish it before
        # calling conclude_tracing, otherwise it will IndexError there.
        for _ in range(depth_before):
            self.session.qv_stack.append(([], []))
        for _ in range(depth_before):
            self.session.conclude_tracing()


class TestDeleteQv:
    def setup_method(self):
        _clear_stacks()
        self.session = TracingQuantumSession()
        self.session.qv_list = []
        self.session.deleted_qv_list = []
        self.session.qv_stack = []

        # _check_trace requires a live JAX Tracer — patch it for unit isolation.
        self.session._check_trace = MagicMock()

        # _clear_qubits calls qubits.tracer, which requires a DynamicQubitArray.
        # Outside tracing mode, qv.reg is a plain list[Qubit], so patch it out.
        self.session._clear_qubits = MagicMock()

    def test_verify_true_raises_value_error(self):
        qv = _make_qv(1, "x")
        with pytest.raises(ValueError, match="verify deletion in tracing mode"):
            self.session.delete_qv(qv, verify=True)

    def test_delete_unknown_qv_raises_value_error(self):
        qv = _make_qv(1, "nonexistent")
        self.session.qv_list = [_make_qv(1, "other")]
        with pytest.raises(ValueError, match="non-existent quantum variable"):
            self.session.delete_qv(qv)

    def test_delete_known_qv_removes_it_from_list(self):
        qv = _make_qv(1, "target")
        other = _make_qv(1, "other")
        self.session.qv_list = [other, qv]

        self.session.delete_qv(qv)

        assert not any(v is qv for v in self.session.qv_list)
        assert any(v is other for v in self.session.qv_list)

    def test_delete_qv_appends_to_deleted_list(self):
        qv = _make_qv(1, "target")
        self.session.qv_list = [qv]

        self.session.delete_qv(qv)

        assert any(v is qv for v in self.session.deleted_qv_list)

    def test_delete_only_first_occurrence_when_duplicates(self):
        """If somehow the same name appears twice, only the first is removed."""
        qv1 = _make_qv(1, "dup")
        qv2 = _make_qv(2, "dup")
        self.session.qv_list = [qv1, qv2]

        self.session.delete_qv(qv1)

        assert len(self.session.qv_list) == 1
        assert self.session.qv_list[0] is qv2


class TestCheckForTracingMode:
    def test_returns_false_outside_trace(self):
        """Outside of a jit/make_jaxpr call there is no frame attribute."""
        assert check_for_tracing_mode() is False

    def test_returns_true_inside_make_jaxpr(self):
        """Inside make_jaxpr, the trace context has a frame attribute."""
        import jax
        import jax.numpy as jnp

        result_holder = []

        def probe(x):
            result_holder.append(check_for_tracing_mode())
            return x

        jax.make_jaxpr(probe)(jnp.array(0))
        assert result_holder == [True]
