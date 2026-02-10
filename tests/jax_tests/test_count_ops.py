"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
    assert ops_with == ops_without, f"Operations should match: with parity {ops_with} vs without {ops_without}"
    
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
