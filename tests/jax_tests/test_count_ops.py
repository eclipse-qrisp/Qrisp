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

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jax.tree_util import register_pytree_node_class

from qrisp import (
    BigInteger,
    QuantumFloat,
    QuantumModulus,
    QuantumVariable,
    control,
    cx,
    h,
    invert,
    measure,
    parity,
    rx,
    rz,
    u3,
    x,
    y,
)
from qrisp.jasp import count_ops, expectation_value, jrange, q_cond, qache
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    always_one,
    always_zero,
    meas_rng,
)
from qrisp.operators import X, Z


class TestCountOpsSingleQubit:
    """Test that count_ops is correctly computed for single-qubit quantum primitives."""

    def test_one_qubit_h_static(self):
        """Test count_ops of a single Hadamard gate on one qubit (static case)."""

        @count_ops(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            h(qf[0])
            return measure(qf[0])

        assert main() == {"h": 1, "measure": 1}

    def test_one_qubit_h(self):
        """Test count_ops of a single Hadamard gate on one qubit (dynamic case)."""

        @count_ops(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)
            h(qf[0])
            return measure(qf[0])

        assert main(1) == {"h": 1, "measure": 1}

    @pytest.mark.parametrize("qubit_indices", [[0, 1, 2], [0, 0, 0]])
    def test_count_independent_of_scheduling(self, qubit_indices):
        """Total gate count must not depend on parallel vs. sequential scheduling."""

        @count_ops(meas_behavior="0")
        def main(num_qubits, qubit_indices):
            qf = QuantumFloat(num_qubits)
            h(qf[qubit_indices[0]])
            h(qf[qubit_indices[1]])
            h(qf[qubit_indices[2]])
            return measure(qf[qubit_indices[-1]])

        assert main(3, qubit_indices) == {"h": 3, "measure": 1}

    def test_rotation_gates_counted_by_name(self):
        """Test that different rotation gates are counted under their own, distinct names."""

        @count_ops(meas_behavior="0")
        def main():
            qf = QuantumFloat(3)
            rx(0.5, qf[0])
            rz(0.3, qf[1])
            u3(0.1, 0.2, 0.3, qf[2])
            return measure(qf)

        assert main() == {"rx": 1, "rz": 1, "u3": 1, "measure": 3}


class TestCountOpsMultiQubit:
    """Test that count_ops is correctly computed for multi-qubit gates."""

    def test_bell_pair(self):
        """Test count_ops of a two-qubit Bell state preparation."""

        @count_ops(meas_behavior="0")
        def main():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            return measure(qv)

        assert main() == {"h": 1, "cx": 1, "measure": 2}

    def test_ghz_state(self):
        """Test count_ops of an n-qubit GHZ state preparation via a jrange loop."""

        @count_ops(meas_behavior="0")
        def main(n):
            qv = QuantumVariable(n)
            h(qv[0])
            for i in jrange(1, n):
                cx(qv[0], qv[i])
            return measure(qv)

        assert main(5) == {"h": 1, "cx": 4, "measure": 5}


class TestCountOpsControlStructures:
    """Test that count_ops is correctly computed for control structures."""

    @pytest.mark.parametrize("selector", [0, -1])
    def test_conditional_1(self, selector):
        """Test count_ops computation in a conditional structure with q_cond."""

        @count_ops(meas_behavior="0")
        def main(num_qubits, selector):
            qf = QuantumFloat(num_qubits)

            def true_fn():
                h(qf[0])
                h(qf[2])
                cx(qf[0], qf[1])
                h(qf[1])

            def false_fn():
                h(qf[0])
                h(qf[2])
                cx(qf[0], qf[1])
                h(qf[2])
                h(qf[1])
                h(qf[1])
                cx(qf[2], qf[3])

            q_cond(selector >= 0, true_fn, false_fn)

            return measure(qf[0])

        expected = {"h": 3, "cx": 1, "measure": 1} if selector >= 0 else {"h": 5, "cx": 2, "measure": 1}
        assert main(4, selector) == expected

    @pytest.mark.parametrize("selector", [0, -1])
    def test_conditional_2(self, selector):
        """Test count_ops computation in a conditional structure with control context.

        Same circuit as ``test_conditional_1``, expressed with ``with control(...)``
        instead of ``q_cond``, to confirm both APIs are counted identically.
        """

        @count_ops(meas_behavior="0")
        def main(num_qubits, selector):
            qf = QuantumFloat(num_qubits)

            with control(selector >= 0):
                h(qf[0])
                h(qf[2])
                cx(qf[0], qf[1])
                h(qf[1])
            with control(selector < 0):
                h(qf[0])
                h(qf[2])
                cx(qf[0], qf[1])
                h(qf[2])
                h(qf[1])
                h(qf[1])
                cx(qf[2], qf[3])

            return measure(qf[0])

        expected = {"h": 3, "cx": 1, "measure": 1} if selector >= 0 else {"h": 5, "cx": 2, "measure": 1}
        assert main(4, selector) == expected

    def test_loop_count_independent_of_pattern(self):
        """Sequential (same-qubit) and parallel (distinct-qubit) loops must give the same total count."""

        @count_ops(meas_behavior="0")
        def sequential(num_qubits):
            qf = QuantumFloat(num_qubits)
            for _ in jrange(qf.size):
                h(qf[0])
            return measure(qf[0])

        @count_ops(meas_behavior="0")
        def parallel(num_qubits):
            qf = QuantumFloat(num_qubits)
            for i in jrange(qf.size):
                h(qf[i])
            return measure(qf[0])

        expected = {"h": 4, "measure": 1}
        assert sequential(4) == expected
        assert parallel(4) == expected

    def test_create_qubits_called_in_conditional(self):
        """Test count_ops computation when create_qubits is called inside a conditional."""

        @count_ops(meas_behavior="0")
        def main(num_qubits):
            qv = QuantumFloat(num_qubits)
            h(qv[0])
            m = measure(qv[0])
            with control(m == 0):
                qv_inner = QuantumFloat(num_qubits)
                h(qv_inner[0])
            return m

        assert main(1) == {"h": 2, "measure": 1}


class TestCountOpsMeasurementBehavior:
    """Test that different measurement behaviors affect count_ops computation correctly."""

    def test_error_on_invalid_measurement_behavior(self):
        """Test that an invalid measurement behavior raises ValueError."""

        @count_ops(meas_behavior="invalid_behavior")
        def main():
            pass

        with pytest.raises(
            ValueError,
            match="Don't know how to compute required resources via method invalid_behavior",
        ):
            main()

    def test_error_on_non_boolean_measurement(self):
        """Test that a non-boolean measurement result raises ValueError."""

        def invalid_meas_behavior(_):
            return 42

        @count_ops(meas_behavior=invalid_meas_behavior)
        def main():
            qf = QuantumFloat(1)
            return measure(qf[0])

        with pytest.raises(ValueError, match="Measurement behavior must return a boolean, got 42"):
            main()

    @pytest.mark.parametrize(
        "meas_behavior,expect_h",
        [
            (always_zero, False),
            (always_one, True),
        ],
    )
    def test_count_ops_controlled_by_measurement(self, meas_behavior, expect_h):
        """Test count_ops computation when operations are controlled by measurement outcomes."""

        @count_ops(meas_behavior=meas_behavior)
        def main(num_qubits):
            qv = QuantumFloat(num_qubits)
            m = measure(qv[0])
            with control(m == 1):
                h(qv[0])
            return m

        result = main(1)
        assert ("h" in result) == expect_h  # type: ignore[operator]
        assert result["measure"] == 1  # type: ignore[index]

    @pytest.mark.parametrize("selector", [0, 1])
    def test_count_ops_rng_is_deterministic(self, selector):
        """Test that count_ops computation with rng-based measurement behavior is deterministic."""

        @count_ops(meas_behavior=meas_rng)
        def main(i):
            qv = QuantumFloat(1)
            m = measure(qv[0])
            with control(m == i):
                h(qv[0])
            return m

        first = main(selector)
        second = main(selector)
        assert first == second

    def test_custom_rng_callable_matches_string_modes(self):
        """A custom RNG-based callable must agree with the "0"/"1" string modes.

        With the default (deterministic) PRNG key, this specific callable always
        resolves ``meas_res == 0`` for both qubits, matching ``meas_behavior="0"``
        exactly and diverging from ``meas_behavior="1"``.
        """

        def meas_behavior(key):
            return jnp.bool(random.randint(key, (1,), 0, 1)[0])

        def main():
            qv = QuantumVariable(2)
            meas_res = measure(qv)
            with control(meas_res == 0):
                x(qv)
            return measure(qv)

        expected_x_gates = 2
        rng_result = count_ops(meas_behavior=meas_behavior)(main)()
        zero_result = count_ops(meas_behavior="0")(main)()
        assert rng_result["x"] == expected_x_gates  # type: ignore[index]
        assert zero_result == rng_result
        assert "x" not in count_ops(meas_behavior="1")(main)()

    def test_sim_mode_matches_deterministic_circuit(self):
        """meas_behavior="sim" must give an exact count when there is no measurement-dependent branching."""

        @count_ops(meas_behavior="sim")
        def main():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            x(qv[1])
            return measure(qv)

        assert main() == {"h": 1, "cx": 1, "x": 1, "measure": 2}

    def test_kernelization_raises_precise_error(self):
        """count_ops on a kernelized function must raise NotImplementedError, not an arbitrary exception."""

        def state_prep():
            qf = QuantumFloat(3)
            h(qf)
            return qf

        @count_ops(meas_behavior="0")
        def main():
            return expectation_value(state_prep, 10)()

        with pytest.raises(
            NotImplementedError,
            match="Quantum kernel creation not yet supported in profiling interpreter",
        ):
            main()


class TestCountOpsGroundTruth:
    """Cross-check count_ops against the classical QuantumCircuit.count_ops(), the actual ground truth."""

    def test_matches_classical_count_ops_with_invert_and_jrange(self):
        """The Jasp-mode count_ops must agree with the classical circuit's own count_ops.

        Builds the circuit outside of Jasp tracing (via ``main(i)``, a plain call
        that returns a real ``QuantumVariable``), compiles it classically, and
        compares its exact gate counts against the Jasp-traced count_ops result
        for the same function and inputs.
        """

        def main(size):
            qf = QuantumFloat(size)
            for idx in jrange(qf.size):
                x(qf[idx])

            with invert():
                y(qf)

            return qf

        for i in range(1, 10):
            classical_counts = main(i).qs.compile().count_ops()
            assert classical_counts == count_ops(meas_behavior="0")(main)(i)

    def test_scales_to_large_registers(self):
        """count_ops must remain tractable for registers far too large to simulate."""

        @count_ops(meas_behavior="0")
        def main(i):
            a = QuantumFloat(i)
            b = QuantumFloat(i)
            c = a * b
            return measure(c)

        small_result = main(5)
        large_result = main(5000)
        assert small_result
        assert large_result


class TestCountOpsArguments:
    """Test count_ops with various kinds of function arguments."""

    def test_static_function_argument(self):
        """Test that a plain (non-quantum-typed) function can be passed as a static argument."""

        def state_prep():
            return QuantumVariable(3)

        @count_ops(meas_behavior="0")
        def main(state_prep):
            qv = state_prep()
            return measure(qv)

        assert main(state_prep) == {"measure": 3}

    def test_same_type_different_shape_caching(self):
        """Test that BigInteger/QuantumModulus arguments of different sizes are traced independently."""

        @count_ops(meas_behavior="0")
        def main(i: BigInteger):
            r = QuantumModulus(i)
            r[:] = 1

        main(BigInteger.create_static(1, 1))
        main(BigInteger.create_static(5, 2))

    def test_custom_pytree_argument_regression(self):
        """Regression test for https://github.com/eclipse-qrisp/Qrisp/issues/281.

        A custom, user-registered pytree class passed as an argument must not
        break tracing, even when the traced function never touches any quantum
        primitive.
        """

        @register_pytree_node_class
        @dataclass(frozen=True)
        class CustomPytree:
            digits: jnp.ndarray

            def tree_flatten(self):
                return (self.digits,), None

            @classmethod
            def tree_unflatten(cls, aux_data, children):
                return cls(*children)

        @count_ops(meas_behavior="0")
        def main(n):
            return n.digits[0]

        assert main(CustomPytree(jnp.array([0, 1]))) == {}

    def test_operator_trotterization_gate_counts(self):
        """Test count_ops on a circuit built from an Operator's trotterization."""

        @count_ops(meas_behavior="1")
        def main(hamiltonian):
            qv = QuantumVariable(2)
            trotter_step = hamiltonian.trotterization()
            trotter_step(qv, 1)

        assert main(X(0) * Z(1)) == {"cx": 2, "rz": 1, "h": 2}


class TestCountOpsParity:
    """Test that the parity classical post-processing helper does not affect gate counts."""

    def test_parity_not_counted_as_operation(self):
        """parity() itself must never appear in the op counts, and must not change them."""

        # use_parity must stay a plain Python bool (not a traced argument):
        # a Python-level `if` on a traced value would fail to trace at all.
        def build_circuit(use_parity):
            qv = QuantumVariable(3)
            x(qv[0])
            x(qv[2])

            m1 = measure(qv[0])
            m2 = measure(qv[1])
            m3 = measure(qv[2])

            if use_parity:
                return parity(m1, m2, m3)
            return m1

        expected_x_gates = 2
        expected_measurements = 3
        with_parity = count_ops(meas_behavior="0")(lambda: build_circuit(True))()
        without_parity = count_ops(meas_behavior="0")(lambda: build_circuit(False))()

        assert with_parity["x"] == expected_x_gates  # type: ignore[index]
        assert with_parity["measure"] == expected_measurements  # type: ignore[index]
        assert "parity" not in with_parity  # type: ignore[operator]
        assert with_parity == without_parity

    def test_parity_with_expectation_parameter(self):
        """Test count_ops when parity() is called with an explicit expectation parameter."""

        @count_ops(meas_behavior="1")
        def main():
            qv = QuantumVariable(2)
            x(qv[0])

            m1 = measure(qv[0])
            m2 = measure(qv[1])

            # Parity is True (one 1), expectation is False
            return parity(m1, m2, expectation=0)

        expected_x_gates = 1
        expected_measurements = 2
        result = main()
        assert result["x"] == expected_x_gates  # type: ignore[index]
        assert result["measure"] == expected_measurements  # type: ignore[index]

    def test_parity_with_array_inputs_triggers_while(self):
        """Test that parity() with array inputs (which triggers a while loop) is profiled correctly."""

        @count_ops(meas_behavior="0")
        def main():
            qv0 = QuantumVariable(3)
            qv1 = QuantumVariable(3)

            x(qv0[0])  # 2 x gates total
            x(qv0[2])
            x(qv1[1])  # 3 x gates total

            m0 = jnp.array([measure(qv0[0]), measure(qv0[1]), measure(qv0[2])])
            m1 = jnp.array([measure(qv1[0]), measure(qv1[1]), measure(qv1[2])])

            return parity(m0, m1)

        expected_x_gates = 3
        expected_measurements = 6
        result = main()
        assert result["x"] == expected_x_gates  # type: ignore[index]
        assert result["measure"] == expected_measurements  # type: ignore[index]
        assert "parity" not in result  # type: ignore[operator]


class TestCountOpsControlFlowRegressions:
    """Regression tests for specific control-flow primitives interacting with count_ops."""

    def test_scan_with_bare_carry(self):
        """Test that a jax.lax.scan with a single (non-tuple) carry is profiled by count_ops.

        Regression test for two bugs found and fixed in evaluate_scan_under_trace
        (control_flow_interpretation.py), both specific to num_carry == 1 (a bare,
        non-tuple carry): an unguarded list(carry) call that crashed with
        "TypeError: iteration over a 0-d array", and a carry/pytree structure
        mismatch between scan's input and output that crashed with a jax.lax.scan
        structure-mismatch error.
        """

        @count_ops(meas_behavior="0")
        def main():
            qv = QuantumVariable(3)
            x(qv[0])
            x(qv[2])

            m0 = measure(qv[0])
            m1 = measure(qv[1])
            m2 = measure(qv[2])

            init_carry = jnp.int64(m0) + jnp.int64(m1) + jnp.int64(m2)
            xs = jnp.array([1, 2, 3], dtype=jnp.int64)

            def body(carry, xi):
                return carry + xi, carry

            # init_carry is a bare scalar, not a tuple -> num_carry == 1
            final_carry, _ = jax.lax.scan(body, init_carry, xs)
            return final_carry

        expected_x_gates = 2
        expected_measurements = 3
        result: dict[str, int] = main()
        assert result["x"] == expected_x_gates
        assert result["measure"] == expected_measurements


class TestCountOpsEdgeCases:
    """Test count_ops on edge cases not covered by the more targeted classes above."""

    def test_empty_circuit(self):
        """Test that a function with no quantum primitives at all returns an empty dict."""

        @count_ops(meas_behavior="0")
        def main():
            pass

        assert main() == {}

    def test_delete_zero_state_register_is_free(self):
        """Deleting a register still in its freshly-allocated |0> state must add no gates."""

        @count_ops(meas_behavior="0")
        def main():
            qv = QuantumVariable(2)
            qv.delete()

        assert main() == {}

    def test_delete_does_not_add_uncomputation_gates(self):
        """Deleting a register that is not in |0> must not silently inflate the gate count.

        Regression test locking in the current, observed behavior: ``delete()``
        does not add any gates of its own to the count, whatever state the
        register is in when deleted.
        """

        @count_ops(meas_behavior="0")
        def main():
            qv = QuantumVariable(2)
            h(qv[0])
            qv.delete()

        assert main() == {"h": 1}

    def test_nested_jrange_loops(self):
        """Test count_ops with a jrange loop nested inside another jrange loop."""

        @count_ops(meas_behavior="0")
        def main(n, m):
            qv = QuantumFloat(n * m)
            for i in jrange(n):
                for j in jrange(m):
                    h(qv[i * m + j])
            return measure(qv)

        assert main(3, 4) == {"h": 12, "measure": 12}


def _assert_thresholds_agree(
    make_circuit: Callable,
    thresholds: tuple[int | None, ...] = (0, 500, 10**9),
    meas_behaviors: tuple[str, ...] = ("0", "1"),
) -> dict[str, int]:
    """Assert that every given callback_threshold yields the same count_ops result as the default.

    Shared helper for the callback_threshold tests below: ``callback_threshold``
    controls whether ``jax.pure_callback`` wrapping is used for reused
    sub-jaxprs to prevent XLA compilation blowup, and must never change the
    profiling result itself.

    Returns
    -------
    dict
        The baseline (default threshold, first meas_behavior) result, for
        callers that want to make additional assertions about its content.

    """
    baseline = None
    for meas_behavior in meas_behaviors:
        this_baseline: dict[str, int] = count_ops(meas_behavior=meas_behavior)(make_circuit)()
        if baseline is None:
            baseline = this_baseline
        for threshold in thresholds:
            result: dict[str, int] = count_ops(meas_behavior=meas_behavior, callback_threshold=threshold)(
                make_circuit
            )()
            assert result == this_baseline, (
                f"meas_behavior={meas_behavior!r}, callback_threshold={threshold} diverged:\n"
                f"  baseline={this_baseline}\n  got={result}"
            )
    assert baseline is not None
    return baseline


def test_callback_threshold_reused_qache_subroutine():
    """Test callback_threshold with a qache'd subroutine reused multiple times."""

    @qache
    def add_controlled_h(qv):
        """Apply H gates controlled on the first qubit to two other qubits."""
        with control(qv[0]):
            h(qv[1])
            h(qv[2])

    def make_circuit():
        qv = QuantumFloat(3)
        h(qv[0])
        add_controlled_h(qv)
        add_controlled_h(qv)
        add_controlled_h(qv)
        x(qv)
        return measure(qv)

    baseline = _assert_thresholds_agree(make_circuit)
    for gate in ("h", "cx", "x", "measure"):
        assert gate in baseline, f"Expected {gate!r} in count_ops result, got {baseline}"


def test_callback_threshold_nested_qache_subroutines():
    """Test callback_threshold with nested @qache'd subroutines, both reused multiple times."""

    @qache
    def inner_sub(qv):
        """Inner subroutine: apply an H gate controlled on qv[0]."""
        with control(qv[0]):
            h(qv[1])

    @qache
    def outer_sub(qv):
        """Outer subroutine: calls inner_sub and adds its own gates."""
        with control(qv[1]):
            h(qv[2])
        inner_sub(qv)
        inner_sub(qv)

    def make_circuit():
        qv = QuantumFloat(4)
        h(qv[0])
        outer_sub(qv)
        outer_sub(qv)
        outer_sub(qv)
        x(qv)
        return measure(qv)

    _assert_thresholds_agree(make_circuit)


def test_callback_threshold_with_jrange_loop():
    """Test callback_threshold with a jrange loop calling a qache'd subroutine many times."""

    @qache
    def iterated_sub(qv, n):
        """Subroutine called inside a jrange loop."""
        with control(qv[0]):
            h(qv[n % 3])
        with control(qv[1]):
            x(qv[n % 3])

    def make_circuit():
        qv = QuantumFloat(4)
        h(qv[0])
        for i in jrange(20):
            iterated_sub(qv, i)
        x(qv)
        return measure(qv)

    baseline = _assert_thresholds_agree(make_circuit)
    assert baseline.get("h", 0) > 0, "Expected some H gates"
    assert baseline.get("x", 0) > 0, "Expected some X gates"


def test_callback_threshold_edge_cases():
    """Test edge cases of callback_threshold: threshold=1, determinism, and no reuse at all."""

    @qache
    def tiny_sub(qv):
        """A very small subroutine (few equations)."""
        x(qv[0])

    def make_circuit():
        qv = QuantumFloat(3)
        h(qv[0])
        tiny_sub(qv)
        tiny_sub(qv)
        tiny_sub(qv)
        return measure(qv)

    # threshold=1: even tiny reused sub-jaxprs should be wrapped
    baseline = _assert_thresholds_agree(make_circuit, thresholds=(1,), meas_behaviors=("0",))

    # Determinism: same threshold, multiple calls give the same result
    for _ in range(3):
        r: dict[str, int] = count_ops(meas_behavior="0", callback_threshold=0)(make_circuit)()
        assert r == baseline, f"Determinism check failed: expected {baseline}, got {r}"

    # Single-call subroutine (no reuse) -- callback_threshold should have no effect
    @qache
    def single_use_sub(qv):
        with control(qv[0]):
            h(qv[1])

    def single_call_circuit():
        qv = QuantumFloat(3)
        h(qv[0])
        single_use_sub(qv)  # only called once -- no reuse
        x(qv)
        return measure(qv)

    _assert_thresholds_agree(single_call_circuit, thresholds=(0,), meas_behaviors=("0",))
