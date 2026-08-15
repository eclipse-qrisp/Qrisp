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

import jax.numpy as jnp
from jax import lax

from qrisp import QuantumFloat, QuantumVariable, control, measure, x
from qrisp.jasp import boolean_simulation, jrange, qache


def test_boolean_simulation():

    @boolean_simulation
    def main(i, j):

        a = QuantumFloat(10)

        b = QuantumFloat(10)

        a[:] = i
        b[:] = j

        c = QuantumFloat(30)

        for i in jrange(150):
            c += a * b

        return measure(c)

    for i in range(5):
        for j in range(5):
            assert main(i, j) == 150 * i * j

    # Test multi switch

    @boolean_simulation
    def main():

        def case0(x):
            return x + 1

        def case1(x):
            return x + 2

        def case2(x):
            return x + 3

        def case3(x):
            return x + 4

        def compute(index, x):
            return lax.switch(index, [case0, case1, case2, case3], x)

        qf = QuantumFloat(2)
        qf[:] = 3
        ind = jnp.int8(measure(qf))

        res = compute(ind, jnp.int32(0))

        return ind, res

    assert main() == (3, 4)

    ## Test qubit array fusion

    @boolean_simulation
    def main():

        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7

        return measure(a.reg + b.reg)

    assert main() == 63

    @boolean_simulation
    def main():

        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7

        return measure(a.reg + b[0])

    assert main() == 15

    @boolean_simulation
    def main():

        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7

        return measure(a[0] + b.reg)

    assert main() == 15

    @boolean_simulation
    def main():

        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7

        return measure(a[0] + b[0])

    assert main() == 3


def test_parity_boolean_simulation():
    from qrisp import boolean_simulation, measure, x
    from qrisp.jasp import parity

    # Test scalar parity with boolean simulation
    @boolean_simulation
    def test_scalar_parity():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])

        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])

        result = parity(m1, m2, m3)
        return result

    # Parity of (True, False, True) = False (even number of Trues)
    result = test_scalar_parity()
    assert result == False, f"Expected False, got {result}"

    # Test with expectation parameter
    @boolean_simulation
    def test_parity_expectation():
        qv = QuantumVariable(2)
        x(qv[0])

        m1 = measure(qv[0])
        m2 = measure(qv[1])

        # Parity is True, expectation is False -> mismatch = True
        result = parity(m1, m2, expectation=0)
        return result

    result = test_parity_expectation()
    assert result == True, f"Expected True (mismatch), got {result}"

    # Test matching expectation
    @boolean_simulation
    def test_parity_match():
        qv = QuantumVariable(2)

        m1 = measure(qv[0])
        m2 = measure(qv[1])

        # Parity is False, expectation is False -> match = False
        result = parity(m1, m2, expectation=0)
        return result

    result = test_parity_match()
    assert result == False, f"Expected False (match), got {result}"


def test_parity_boolean_simulation_inside_loop():
    """Test parity with array inputs (scan primitive) in boolean simulation."""
    import jax.numpy as jnp

    from qrisp import boolean_simulation, measure, x
    from qrisp.jasp import parity

    @boolean_simulation
    def test_array_parity():
        qv0 = QuantumVariable(3)
        qv1 = QuantumVariable(3)

        # Set specific states
        x(qv0[0])  # qv0 = [1, 0, 1] after x on [0] and [2]
        x(qv0[2])
        x(qv1[1])  # qv1 = [0, 1, 0]

        # Measure individual qubits
        m0_0 = measure(qv0[0])
        m0_1 = measure(qv0[1])
        m0_2 = measure(qv0[2])

        m1_0 = measure(qv1[0])
        m1_1 = measure(qv1[1])
        m1_2 = measure(qv1[2])

        # Create arrays and compute parity (triggers scan)
        meas_array_0 = jnp.array([m0_0, m0_1, m0_2])
        meas_array_1 = jnp.array([m1_0, m1_1, m1_2])

        result = parity(meas_array_0, meas_array_1)
        return result

    result = test_array_parity()
    # Expected: [1 XOR 0, 0 XOR 1, 1 XOR 0] = [1, 1, 1]
    expected = jnp.array([1, 1, 1])
    assert jnp.array_equal(result, expected), f"Expected {expected}, got {result}"

    # Test with all zeros
    @boolean_simulation
    def test_array_parity_zeros():
        qv0 = QuantumVariable(3)
        qv1 = QuantumVariable(3)

        m0_0 = measure(qv0[0])
        m0_1 = measure(qv0[1])
        m0_2 = measure(qv0[2])

        m1_0 = measure(qv1[0])
        m1_1 = measure(qv1[1])
        m1_2 = measure(qv1[2])

        meas_array_0 = jnp.array([m0_0, m0_1, m0_2])
        meas_array_1 = jnp.array([m1_0, m1_1, m1_2])

        result = parity(meas_array_0, meas_array_1)
        return result

    result = test_array_parity_zeros()
    expected = jnp.array([0, 0, 0])
    assert jnp.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_boolean_simulation_pytree():
    """Test that boolean_simulation preserves PyTree structure in return values."""

    # Test 1: Returning a dictionary
    @boolean_simulation
    def dict_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 5
        b[:] = 3
        return {"first": measure(a), "second": measure(b)}

    result = dict_return()
    assert isinstance(result, dict), "Should return a dict"
    assert "first" in result and "second" in result, "Should have correct keys"
    assert result["first"] == 5, "first should be 5"
    assert result["second"] == 3, "second should be 3"

    # Test 2: Returning a list
    @boolean_simulation
    def list_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        c = QuantumFloat(4)
        a[:] = 1
        b[:] = 2
        c[:] = 3
        return [measure(a), measure(b), measure(c)]

    result = list_return()
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 3, "Should have 3 elements"
    assert result == [1, 2, 3], "Should have correct values"

    # Test 3: Returning a tuple (should still work as before)
    @boolean_simulation
    def tuple_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        a[:] = 7
        b[:] = 9
        return (measure(a), measure(b))

    result = tuple_return()
    assert isinstance(result, tuple), "Should return a tuple"
    assert result == (7, 9), "Should have correct values"

    # Test 4: Returning nested structure
    @boolean_simulation
    def nested_return():
        a = QuantumFloat(4)
        b = QuantumFloat(4)
        c = QuantumFloat(4)
        a[:] = 1
        b[:] = 2
        c[:] = 3
        return {"pair": (measure(a), measure(b)), "single": measure(c)}

    result = nested_return()
    assert isinstance(result, dict), "Should return a dict"
    assert isinstance(result["pair"], tuple), "Nested tuple should be preserved"
    assert result["pair"] == (1, 2), "pair should be (1, 2)"
    assert result["single"] == 3, "single should be 3"

    # Test 5: Single return value (scalar)
    @boolean_simulation
    def single_return():
        a = QuantumFloat(4)
        a[:] = 11
        return measure(a)

    result = single_return()
    # Single values should be returned as-is
    assert result == 11, "Should return the scalar value"

    # Test 6: No return value
    @boolean_simulation
    def no_return():
        a = QuantumFloat(4)
        a[:] = 5

    result = no_return()
    assert result is None, "Should return None"

    # Test 7: Returning dict with computed values
    @boolean_simulation
    def computed_dict(x, y):
        a = QuantumFloat(8)
        b = QuantumFloat(8)
        a[:] = x
        b[:] = y
        c = a + b
        d = a * b
        return {"sum": measure(c), "product": measure(d)}

    result = computed_dict(3, 4)
    assert result["sum"] == 7, "sum should be 7"
    assert result["product"] == 12, "product should be 12"


def test_callback_threshold_boolean_simulation():
    """Test that boolean_simulation with callback_threshold produces correct results.

    The callback_threshold parameter controls whether ``jax.pure_callback``
    wrapping is used for reused sub-jaxprs to prevent XLA compilation blowup.
    Different threshold values should all produce the same simulation results.
    """

    # A qache'd subroutine that will be reused multiple times.
    @qache
    def boolean_sub(qf):
        """A subroutine with multiple boolean operations."""
        with control(qf[0]):
            x(qf[1])
        with control(qf[1]):
            x(qf[2])
        return measure(qf[0])

    def make_circuit(i):
        qf = QuantumFloat(4)
        qf[:] = i

        # Call the qache'd subroutine multiple times so it gets reused
        m1 = boolean_sub(qf)
        m2 = boolean_sub(qf)
        m3 = boolean_sub(qf)

        final_res = measure(qf)
        return final_res, m1, m2, m3

    # Baseline: no callbacks (default)
    baseline_func = boolean_simulation(make_circuit)
    baseline = baseline_func(7)

    # callback_threshold=0: wrap every reused sub-jaxpr
    result_0_func = boolean_simulation(make_circuit, callback_threshold=0)
    result_0 = result_0_func(7)
    assert result_0 == baseline, f"callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"

    # callback_threshold=500: middle ground
    result_500_func = boolean_simulation(make_circuit, callback_threshold=500)
    result_500 = result_500_func(7)
    assert result_500 == baseline, f"callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"

    # Very large threshold
    result_large_func = boolean_simulation(make_circuit, callback_threshold=10**9)
    result_large = result_large_func(7)
    assert result_large == baseline, f"callback_threshold=10**9 diverged:\n  baseline={baseline}\n  got={result_large}"

    # Also test the decorator syntax with callback_threshold
    @boolean_simulation(callback_threshold=0)
    def decorated_circuit(i):
        qf = QuantumFloat(4)
        qf[:] = i
        boolean_sub(qf)
        boolean_sub(qf)
        boolean_sub(qf)
        return measure(qf)

    result_decorated = decorated_circuit(7)

    @boolean_simulation
    def baseline_decorated(i):
        qf = QuantumFloat(4)
        qf[:] = i
        boolean_sub(qf)
        boolean_sub(qf)
        boolean_sub(qf)
        return measure(qf)

    baseline_decorated_result = baseline_decorated(7)
    assert result_decorated == baseline_decorated_result, (
        f"decorator syntax diverged:\n  baseline={baseline_decorated_result}\n  got={result_decorated}"
    )

    # Test simple circuit with no qache'd subroutines (should still work)
    @boolean_simulation(callback_threshold=0)
    def simple_circuit(x, y):
        a = QuantumFloat(8)
        b = QuantumFloat(8)
        a[:] = x
        b[:] = y
        c = a * b
        return measure(c)

    assert simple_circuit(3, 4) == 12, "Simple multiply with callback_threshold=0 failed"
    assert simple_circuit(7, 8) == 56, "Simple multiply with callback_threshold=0 failed"


def test_callback_threshold_boolean_simulation_nested():
    """Test callback_threshold with nested @qache'd subroutines in boolean_simulation.

    An outer qache'd function calls an inner qache'd function.  Both are
    reused multiple times.  The callback wrapping must preserve correctness
    across all threshold values.
    """

    @qache
    def inner_bool_sub(qf):
        """Inner subroutine: controlled-X operations."""
        with control(qf[0]):
            x(qf[1])
        return measure(qf[0])

    @qache
    def outer_bool_sub(qf):
        """Outer subroutine: calls inner_bool_sub and adds its own gates."""
        with control(qf[1]):
            x(qf[2])
        m1 = inner_bool_sub(qf)
        m2 = inner_bool_sub(qf)
        return m1, m2

    def make_circuit(i):
        qf = QuantumFloat(4)
        qf[:] = i

        outer_bool_sub(qf)
        outer_bool_sub(qf)
        outer_bool_sub(qf)

        final_res = measure(qf)
        return final_res

    baseline_func = boolean_simulation(make_circuit)
    baseline = baseline_func(7)

    result_0_func = boolean_simulation(make_circuit, callback_threshold=0)
    result_0 = result_0_func(7)
    assert result_0 == baseline, (
        f"Nested qache boolean_simulation callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"
    )

    result_500_func = boolean_simulation(make_circuit, callback_threshold=500)
    result_500 = result_500_func(7)
    assert result_500 == baseline, (
        f"Nested qache boolean_simulation callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"
    )


def test_callback_threshold_boolean_simulation_jrange():
    """Test callback_threshold with a jrange loop in boolean_simulation.

    The jrange loop calls a qache'd subroutine many times, creating many
    call sites — the exact scenario callback wrapping optimizes for.
    """

    @qache
    def iterated_bool_sub(qf, n):
        """Subroutine called inside a jrange loop."""
        with control(qf[0]):
            x(qf[n % 3])
        return measure(qf[0])

    def make_circuit(i):
        qf = QuantumFloat(4)
        qf[:] = i

        for idx in jrange(30):
            iterated_bool_sub(qf, idx)

        final_res = measure(qf)
        return final_res

    baseline_func = boolean_simulation(make_circuit)
    baseline = baseline_func(5)

    result_0_func = boolean_simulation(make_circuit, callback_threshold=0)
    result_0 = result_0_func(5)
    assert result_0 == baseline, (
        f"jrange boolean_simulation callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"
    )

    result_500_func = boolean_simulation(make_circuit, callback_threshold=500)
    result_500 = result_500_func(5)
    assert result_500 == baseline, (
        f"jrange boolean_simulation callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"
    )

    # threshold=1 edge case
    result_1_func = boolean_simulation(make_circuit, callback_threshold=1)
    result_1 = result_1_func(5)
    assert result_1 == baseline, (
        f"jrange boolean_simulation callback_threshold=1 diverged:\n  baseline={baseline}\n  got={result_1}"
    )
