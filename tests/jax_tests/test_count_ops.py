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

from jax import random

from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class, tree_flatten

from qrisp import *
from qrisp.jasp import *


def test_count_ops():

    # Test general pipeline
    @count_ops(meas_behavior="0")
    def main(i):

        a = QuantumFloat(i)
        b = QuantumFloat(i)

        c = a * b

        return measure(c)

    print(main(5))
    print(main(5000))

    # Test pipeline correctness

    def main(i):

        qf = QuantumFloat(i)
        for i in jrange(qf.size):
            x(qf[i])

        with invert():
            y(qf)

        return qf

    for i in range(1, 10):
        assert main(i).qs.compile().count_ops() == count_ops(meas_behavior="0")(main)(i)

    def meas_behavior(key):
        return jnp.bool(random.randint(key, (1,), 0, 1)[0])

    def main():

        qv = QuantumVariable(2)
        meas_res = measure(qv)

        with control(meas_res == 0):
            x(qv)

        return measure(qv)

    assert count_ops(meas_behavior=meas_behavior)(main)()["x"] == 2
    assert count_ops(meas_behavior="0")(main)()["x"] == 2
    assert "x" not in count_ops(meas_behavior="1")(main)()

    # Test passing static arguments
    def state_prep():
        return QuantumVariable(3)

    @count_ops(meas_behavior="0")
    def main(state_prep):
        qv = state_prep()
        return measure(qv)

    assert main(state_prep) == {"measure": 3}

    # Test same type - different shape caching behavior

    @count_ops(meas_behavior="0")
    def main(i: BigInteger):
        r = QuantumModulus(i)
        r[:] = 1

    main(BigInteger.create_static(1, 1))
    main(BigInteger.create_static(5, 2))

    # Test https://github.com/eclipse-qrisp/Qrisp/issues/281
    @register_pytree_node_class
    @dataclass(frozen=True)
    class TestClass:
        digits: jnp.ndarray

        def tree_flatten(self):
            return (self.digits,), None

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    @count_ops(meas_behavior="0")
    def ttt(N):
        return N.digits[0]

    assert ttt(TestClass(jnp.array([0, 1]))) == {}

    # Test Operators as input arguments
    from qrisp.operators import X, Z

    @count_ops(meas_behavior="1")
    def main(H):
        qv = QuantumVariable(2)
        U = H.trotterization()
        U(qv, 1)

    assert main(X(0) * Z(1)) == {"cx": 2, "rz": 1, "h": 2}

    # Test kernilization error message
    def state_prep():
        qf = QuantumFloat(3)
        h(qf)
        return qf

    @count_ops(meas_behavior="0")
    def main():
        return expectation_value(state_prep, 10)()

    try:
        main()
    except Exception as e:
        if "kernel" in str(e):
            return
        else:
            assert False


def test_parity_count_ops():
    """Test that parity primitive works with count_ops profiling."""

    # Test basic parity - should not add to op counts
    @count_ops(meas_behavior="0")
    def test_parity_basic():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])

        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])

        result = parity(m1, m2, m3)
        return result

    ops = test_parity_basic()
    # Should count x gates and measurements, but not parity
    assert ops["x"] == 2, f"Expected 2 x gates, got {ops.get('x', 0)}"
    assert ops["measure"] == 3, f"Expected 3 measurements, got {ops.get('measure', 0)}"
    assert "parity" not in ops, "Parity should not be counted as an operation"

    # Test that parity doesn't affect operation counts compared to without parity
    @count_ops(meas_behavior="0")
    def test_without_parity():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])

        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])

        return m1  # Just return measurement, no parity

    ops_without = test_without_parity()
    ops_with = test_parity_basic()

    # Operation counts should be identical (parity doesn't add quantum operations)
    assert ops_with == ops_without, (
        f"Operations should match: with parity {ops_with} vs without {ops_without}"
    )

    # Test with expectation parameter
    @count_ops(meas_behavior="1")
    def test_parity_expectation():
        qv = QuantumVariable(2)
        x(qv[0])

        m1 = measure(qv[0])
        m2 = measure(qv[1])

        # Parity is True (one 1), expectation is False
        result = parity(m1, m2, expectation=0)
        return result

    ops = test_parity_expectation()
    assert ops["x"] == 1, f"Expected 1 x gate, got {ops.get('x', 0)}"
    assert ops["measure"] == 2, f"Expected 2 measurements, got {ops.get('measure', 0)}"


def test_parity_count_ops_in_while():
    """Test that parity with array inputs (while primitive) works with count_ops profiling."""
    import jax.numpy as jnp

    @count_ops(meas_behavior="0")
    def test_array_parity():
        qv0 = QuantumVariable(3)
        qv1 = QuantumVariable(3)

        # Set specific states
        x(qv0[0])  # 2 x gates total
        x(qv0[2])
        x(qv1[1])  # 3 x gates total

        # Measure individual qubits - 6 measurements
        m0_0 = measure(qv0[0])
        m0_1 = measure(qv0[1])
        m0_2 = measure(qv0[2])

        m1_0 = measure(qv1[0])
        m1_1 = measure(qv1[1])
        m1_2 = measure(qv1[2])

        # Create arrays and compute parity (triggers while)
        meas_array_0 = jnp.array([m0_0, m0_1, m0_2])
        meas_array_1 = jnp.array([m1_0, m1_1, m1_2])

        result = parity(meas_array_0, meas_array_1)
        return result

    ops = test_array_parity()
    # Should count x gates and measurements, but not parity
    assert ops["x"] == 3, f"Expected 3 x gates, got {ops.get('x', 0)}"
    assert ops["measure"] == 6, f"Expected 6 measurements, got {ops.get('measure', 0)}"
    assert "parity" not in ops, "Parity should not be counted as an operation"


def test_callback_threshold_count_ops():
    """Test that count_ops with callback_threshold produces correct results.

    The callback_threshold parameter controls whether ``jax.pure_callback``
    wrapping is used for reused sub-jaxprs to prevent XLA compilation blowup.
    Different threshold values should all produce the same profiling results.
    """

    # A qache'd subroutine that will be reused multiple times.
    # The reuse triggers the call-graph analysis and callback wrapping.
    @qache
    def add_controlled_h(qv):
        """Apply H gates controlled on the first qubit to two other qubits."""
        with control(qv[0]):
            h(qv[1])
            h(qv[2])

    def make_circuit():
        qv = QuantumFloat(3)
        h(qv[0])

        # Call the qache'd subroutine multiple times so it gets reused
        add_controlled_h(qv)
        add_controlled_h(qv)
        add_controlled_h(qv)

        x(qv)
        return measure(qv)

    # Baseline: no callbacks (default, fastest execution)
    baseline = count_ops(meas_behavior="0")(make_circuit)()

    # callback_threshold=0: wrap every reused sub-jaxpr (fastest compilation)
    result_0 = count_ops(meas_behavior="0", callback_threshold=0)(make_circuit)()
    assert result_0 == baseline, (
        f"callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"
    )

    # callback_threshold=500: suggested middle ground
    result_500 = count_ops(meas_behavior="0", callback_threshold=500)(make_circuit)()
    assert result_500 == baseline, (
        f"callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"
    )

    # callback_threshold with a very large value: no sub-jaxpr should be wrapped
    result_large = count_ops(meas_behavior="0", callback_threshold=10**9)(
        make_circuit
    )()
    assert result_large == baseline, (
        f"callback_threshold=10**9 diverged:\n  baseline={baseline}\n  got={result_large}"
    )

    # Also test with meas_behavior="1"
    baseline_1 = count_ops(meas_behavior="1")(make_circuit)()
    result_1_0 = count_ops(meas_behavior="1", callback_threshold=0)(make_circuit)()
    assert result_1_0 == baseline_1, (
        f"meas_behavior='1', callback_threshold=0 diverged:\n  baseline={baseline_1}\n  got={result_1_0}"
    )

    # Verify that the results contain expected gates
    assert "h" in baseline, f"Expected 'h' in count_ops result, got {baseline}"
    assert "cx" in baseline, f"Expected 'cx' in count_ops result, got {baseline}"
    assert "x" in baseline, f"Expected 'x' in count_ops result, got {baseline}"
    assert "measure" in baseline, (
        f"Expected 'measure' in count_ops result, got {baseline}"
    )


def test_callback_threshold_nested_qache():
    """Test callback_threshold with nested @qache'd subroutines.

    An outer qache'd function calls an inner qache'd function.  Both are
    reused multiple times.  The callback wrapping must handle the nested
    case without corrupting profiling results.
    """

    @qache
    def inner_sub(qv):
        """Inner subroutine: apply H gates controlled on qv[0]."""
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

        # Reuse both outer_sub (and transitively inner_sub) multiple times
        outer_sub(qv)
        outer_sub(qv)
        outer_sub(qv)

        x(qv)
        return measure(qv)

    baseline = count_ops(meas_behavior="0")(make_circuit)()
    result_0 = count_ops(meas_behavior="0", callback_threshold=0)(make_circuit)()
    assert result_0 == baseline, (
        f"Nested qache with callback_threshold=0 diverged:\n"
        f"  baseline={baseline}\n  got={result_0}"
    )

    result_500 = count_ops(meas_behavior="0", callback_threshold=500)(make_circuit)()
    assert result_500 == baseline, (
        f"Nested qache with callback_threshold=500 diverged:\n"
        f"  baseline={baseline}\n  got={result_500}"
    )

    # Nested qache with meas_behavior="1"
    baseline_1 = count_ops(meas_behavior="1")(make_circuit)()
    result_1_0 = count_ops(meas_behavior="1", callback_threshold=0)(make_circuit)()
    assert result_1_0 == baseline_1, (
        f"Nested qache meas_behavior='1', callback_threshold=0 diverged:\n"
        f"  baseline={baseline_1}\n  got={result_1_0}"
    )


def test_callback_threshold_with_jrange():
    """Test callback_threshold with a jrange loop calling a qache'd subroutine.

    The jrange loop creates many call sites for the same qache'd function,
    which is exactly the scenario that callback wrapping is designed to
    optimize.  All threshold values must produce identical results.
    """

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

    baseline = count_ops(meas_behavior="0")(make_circuit)()
    result_0 = count_ops(meas_behavior="0", callback_threshold=0)(make_circuit)()
    assert result_0 == baseline, (
        f"jrange loop with callback_threshold=0 diverged:\n"
        f"  baseline={baseline}\n  got={result_0}"
    )

    result_500 = count_ops(meas_behavior="0", callback_threshold=500)(make_circuit)()
    assert result_500 == baseline, (
        f"jrange loop with callback_threshold=500 diverged:\n"
        f"  baseline={baseline}\n  got={result_500}"
    )

    # Verify that the gate counts are as expected (20 iterations of iterated_sub)
    assert baseline.get("h", 0) > 0, "Expected some H gates"
    assert baseline.get("x", 0) > 0, "Expected some X gates"


def test_callback_threshold_edge_cases():
    """Test edge cases of callback_threshold.

    - threshold=1: every reused sub-jaxpr (call_count > 1) gets wrapped,
      regardless of its size.
    - Determinism: running the same circuit multiple times with the same
      threshold yields identical results.
    """

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

    baseline = count_ops(meas_behavior="0")(make_circuit)()

    # threshold=1: even tiny reused sub-jaxprs should be wrapped
    result_1 = count_ops(meas_behavior="0", callback_threshold=1)(make_circuit)()
    assert result_1 == baseline, (
        f"callback_threshold=1 diverged:\n  baseline={baseline}\n  got={result_1}"
    )

    # Determinism: same threshold, multiple calls give same result
    for _ in range(3):
        r = count_ops(meas_behavior="0", callback_threshold=0)(make_circuit)()
        assert r == baseline, f"Determinism check failed: expected {baseline}, got {r}"

    # Single-call subroutine (no reuse) — callback_threshold should have no effect
    @qache
    def single_use_sub(qv):
        with control(qv[0]):
            h(qv[1])

    def single_call_circuit():
        qv = QuantumFloat(3)
        h(qv[0])
        single_use_sub(qv)  # only called once — no reuse
        x(qv)
        return measure(qv)

    s_baseline = count_ops(meas_behavior="0")(single_call_circuit)()
    s_result_0 = count_ops(meas_behavior="0", callback_threshold=0)(
        single_call_circuit
    )()
    assert s_result_0 == s_baseline, (
        f"Single-call with callback_threshold=0 diverged:\n"
        f"  baseline={s_baseline}\n  got={s_result_0}"
    )
