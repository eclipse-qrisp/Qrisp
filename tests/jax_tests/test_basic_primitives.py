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

from qrisp import *
from jax import make_jaxpr

def test_basic_primitives():

    def test_function():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        res_bl = measure(qv[0])
        return res_bl

    compare_jaxpr(make_jaspr(test_function)(), 
            ['jasp.create_qubits',
             'jasp.get_qubit',
             'jasp.quantum_gate',
             'jasp.get_qubit',
             'jasp.quantum_gate',
             'jasp.measure',
             ])
    
    def test_function():
        qv = QuantumVariable(2)
        
        with QuantumEnvironment():
            h(qv[0])
            cx(qv[0], qv[1])
        
        res_bl = measure(qv[0])
        return res_bl
    
    compare_jaxpr(make_jaspr(test_function)(), 
                ['jasp.create_qubits',
                 'jasp.get_qubit',
                 'jasp.quantum_gate',
                 'jasp.get_qubit',
                 'jasp.quantum_gate',
                 'jasp.get_qubit',
                 'jasp.measure'])    

    def test_function():
        qv = QuantumVariable(2)
        
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[1], qv[0])
        h(qv[1])
        res_bl = measure(qv[0])
        qv.delete()
        return res_bl

    compare_jaxpr(make_jaspr(test_function)(), 
                ['jasp.create_qubits',
                 'jasp.get_qubit',
                 'jasp.quantum_gate',
                 'jasp.get_qubit',
                 'jasp.quantum_gate',
                 'jasp.quantum_gate',
                 'jasp.quantum_gate',
                 'jasp.measure',
                 'jasp.delete_qubits'])

    def test_function(a):
        qv = QuantumVariable(2)
        
        rz(np.pi, qv[0])
        p(a, qv[0])
        res_bl = measure(qv[0])
        qv.delete()
        return res_bl

    print(make_jaspr(test_function)(2.))
    compare_jaxpr(make_jaspr(test_function)(2.), 
                ['jasp.create_qubits',
                 'jasp.get_qubit',
                 'jasp.quantum_gate',
                 'jasp.quantum_gate',
                 'jasp.measure',
                 'jasp.delete_qubits'])
    

def test_qc_loss_error_message():
    # Test    
    @jaspify
    def main():
    
    
        def body_fun(k, state):
            
            qv = QuantumFloat(2)
            qv.delete()
        
            return state
        
        state = fori_loop(0,5,body_fun,1)
    
        return state
    
    try:
        main()
    except Exception as e:
        assert "quantum_kernel" in str(e)
        return
    
    assert False
    
def test_list_appending():
    
    @jaspify
    def main():
        
        qv_a = QuantumFloat(5)
        
        qubit_list_a = [qv_a[i] for i in range(5)]
        
        x(qubit_list_a)
        
        qv_b = QuantumFloat(5)
        
        qubit_list_b = [qv_b[i] for i in range(5)]
        
        cx(qubit_list_a, qubit_list_b)
        
        return measure(qv_a), measure(qv_a)

    assert main() == (31, 31)
    
def test_quantum_array_appending():
    
    # Test QuantumArray gate application behavior
    @jaspify
    def test():
        qa = QuantumArray(qtype=QuantumFloat(1), shape=(3,2))
        qb = QuantumArray(qtype=QuantumFloat(1), shape=(3,2))
        x(qa)
        cx(qa, qb)
        return measure(qb)

    assert test()[0,0] == 1
    
    @jaspify
    def test():
        qa = QuantumArray(qtype=QuantumFloat(1), shape=(3,2))
        qb = QuantumArray(qtype=QuantumFloat(1), shape=(3,1))
        x(qa)
        cx(qa, qb)
        return measure(qb)

    try:
        test()
    except Exception as e:
        assert "shape" in str(e)
        
        
    @jaspify
    def test():
        qa = QuantumArray(qtype=QuantumFloat(1), shape=(3,2))
        qb = QuantumArray(qtype=QuantumFloat(1), shape=(3,2))
        x(qa)
        cx(qa, qb[0,1])
        return measure(qb)

    try:
        test()
    except Exception as e:
        assert "mixed" in str(e)


def test_parity_scalar():
    """Test parity function with scalar boolean inputs."""
    
    # Test basic scalar parity (XOR)
    def test_function():
        qv = QuantumVariable(4)
        h(qv[0])
        
        meas_a = measure(qv[0])
        meas_b = measure(qv[1])
        meas_c = measure(qv[2])
        meas_d = measure(qv[3])
        
        # Test parity with different numbers of inputs
        result_2 = parity(meas_a, meas_b)
        result_3 = parity(meas_a, meas_b, meas_c)
        result_4 = parity(meas_a, meas_b, meas_c, meas_d)
        
        return result_2, result_3, result_4
    
    # Check that parity primitive is in the jaxpr
    jaspr = make_jaspr(test_function)()
    jaxpr_str = str(jaspr.jaxpr)
    assert 'parity' in jaxpr_str
    
    # Test with expectation parameter
    def test_function_with_expectation():
        qv = QuantumVariable(2)
        
        meas_a = measure(qv[0])
        meas_b = measure(qv[1])
        
        # Test with None (returns raw parity)
        result_none = parity(meas_a, meas_b, observable=True)
        # Test with False expectation
        result_false = parity(meas_a, meas_b, expectation=0)
        
        return result_none, result_false
    
    jaspr = make_jaspr(test_function_with_expectation)()
    assert 'parity' in str(jaspr.jaxpr)


def test_parity_array():
    """Test parity function with array inputs."""
    import jax.numpy as jnp
    
    # Test element-wise parity on arrays
    def test_function():
        qv = QuantumVariable(3)
        
        # Create arrays of measurements
        meas_a = jnp.array([True, False, True, False])
        meas_b = jnp.array([False, False, True, True])
        meas_c = jnp.array([True, True, False, False])
        
        # Test parity with arrays
        result_2 = parity(meas_a, meas_b)
        result_3 = parity(meas_a, meas_b, meas_c)
        
        return result_2, result_3
    
    jaspr = make_jaspr(test_function)()
    
    jaxpr_str = str(jaspr.jaxpr)
    assert 'while' in jaxpr_str
    assert 'parity' in jaxpr_str
    
    # Test shape mismatch error
    def test_shape_mismatch():
        meas_a = jnp.array([True, False, True])
        meas_b = jnp.array([False, False])  # Different shape!
        
        return parity(meas_a, meas_b)
    
    try:
        make_jaspr(test_shape_mismatch)()
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "same shape" in str(e)
    
    # Test mixing scalar and array error
    def test_mixed_scalar_array():
        qv = QuantumVariable(2)
        meas_scalar = measure(qv[0])
        meas_array = jnp.array([True, False])
        
        return parity(meas_scalar, meas_array)
    
    try:
        make_jaspr(test_mixed_scalar_array)()
        assert False, "Should have raised ValueError for mixing scalar and array"
    except ValueError as e:
        assert "scalar and array" in str(e)
    
    
def test_redundant_allocation_removal():
    
    @make_jaspr
    def main():
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        return a.size

    jaspr_str = str(main())
    assert jaspr_str.count("jasp.create_qubits") == 0

    @make_jaspr
    def main():
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        res = a.size
        a.delete()
        return res

    jaspr_str = str(main())
    assert jaspr_str.count("jasp.create_qubits") == 0

    
    @make_jaspr
    def main():
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        a = QuantumFloat(5)
        return measure(a)

    jaspr_str = str(main())
    assert jaspr_str.count("jasp.create_qubits") == 1
    
    @make_jaspr
    def main():
        qa = QuantumArray(qtype = QuantumFloat(5), shape = (5,5))
        return measure(qa[0, 1])

    jaspr_str = str(main())
    assert jaspr_str.count("jasp.create_qubits") == 1


def test_make_jaxpr_mod():
    """Test the make_jaxpr_mod function with return_shape parameter."""
    from qrisp.jasp import make_jaxpr_mod
    from jax.tree_util import tree_unflatten
    import jax.numpy as jnp
    
    # Test 1: Basic function without return_shape (should behave like make_jaxpr)
    def simple_func(x):
        return x + 1
    
    jaxpr = make_jaxpr_mod(simple_func)(1.0)
    assert hasattr(jaxpr, 'jaxpr'), "Should return a ClosedJaxpr"
    
    # Test 2: Function with return_shape=True returning a simple value
    def simple_func(x):
        return x * 2
    
    jaxpr, out_tree = make_jaxpr_mod(simple_func, return_shape=True)(1.0)
    assert hasattr(jaxpr, 'jaxpr'), "Should return a ClosedJaxpr"
    # Verify out_tree can reconstruct the structure
    result = tree_unflatten(out_tree, [42.0])
    assert result == 42.0, "Should reconstruct scalar correctly"
    
    # Test 3: Function returning a dictionary (PyTree)
    def dict_func(x):
        return {"a": x + 1, "b": x * 2}
    
    jaxpr, out_tree = make_jaxpr_mod(dict_func, return_shape=True)(1.0)
    result = tree_unflatten(out_tree, [2.0, 2.0])
    assert isinstance(result, dict), "Should reconstruct dict"
    assert "a" in result and "b" in result, "Should have correct keys"
    
    # Test 4: Function returning a tuple
    def tuple_func(x, y):
        return (x + y, x - y, x * y)
    
    jaxpr, out_tree = make_jaxpr_mod(tuple_func, return_shape=True)(3.0, 2.0)
    result = tree_unflatten(out_tree, [5.0, 1.0, 6.0])
    assert isinstance(result, tuple), "Should reconstruct tuple"
    assert len(result) == 3, "Should have 3 elements"
    assert result == (5.0, 1.0, 6.0), "Should have correct values"
    
    # Test 5: Function returning nested structure
    def nested_func(x):
        return {"values": (x, x + 1), "squared": x ** 2}
    
    jaxpr, out_tree = make_jaxpr_mod(nested_func, return_shape=True)(2.0)
    result = tree_unflatten(out_tree, [2.0, 3.0, 4.0])
    assert isinstance(result, dict), "Should reconstruct nested dict"
    assert isinstance(result["values"], tuple), "Should have tuple nested"
    
    # Test 6: Function with static_argnums
    def static_func(x, n):
        return x * n
    
    jaxpr, out_tree = make_jaxpr_mod(static_func, static_argnums=(1,), return_shape=True)(2.0, 3)
    assert hasattr(jaxpr, 'jaxpr'), "Should work with static_argnums"
    
    # Test 7: Function returning a list
    def list_func(x):
        return [x, x + 1, x + 2]
    
    jaxpr, out_tree = make_jaxpr_mod(list_func, return_shape=True)(0.0)
    result = tree_unflatten(out_tree, [0.0, 1.0, 2.0])
    assert isinstance(result, list), "Should reconstruct list"
    assert len(result) == 3, "Should have 3 elements"


def test_make_jaspr_return_shape():
    """Test make_jaspr with return_shape parameter for quantum functions."""
    from jax.tree_util import tree_unflatten
    
    # Test 1: Quantum function returning a single measurement
    def single_measurement():
        qv = QuantumVariable(2)
        h(qv[0])
        return measure(qv[0])
    
    jaspr, out_tree = make_jaspr(single_measurement, return_shape=True)()
    assert hasattr(jaspr, 'jaxpr'), "Should return a Jaspr"
    # Single bool result
    result = tree_unflatten(out_tree, [True])
    assert result == True, "Should reconstruct scalar"
    
    # Test 2: Quantum function returning a tuple of measurements
    def tuple_measurement():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return (measure(qv[0]), measure(qv[1]))
    
    jaspr, out_tree = make_jaspr(tuple_measurement, return_shape=True)()
    result = tree_unflatten(out_tree, [True, True])
    assert isinstance(result, tuple), "Should reconstruct tuple"
    assert len(result) == 2, "Should have 2 elements"
    
    # Test 3: Quantum function returning a dictionary
    def dict_measurement():
        qv = QuantumVariable(2)
        h(qv[0])
        return {"first": measure(qv[0]), "second": measure(qv[1])}
    
    jaspr, out_tree = make_jaspr(dict_measurement, return_shape=True)()
    result = tree_unflatten(out_tree, [False, False])
    assert isinstance(result, dict), "Should reconstruct dict"
    assert "first" in result and "second" in result, "Should have correct keys"
    
    # Test 4: Quantum function with parameters
    def param_function(angle):
        qv = QuantumVariable(1)
        rz(angle, qv[0])
        return measure(qv[0])
    
    jaspr, out_tree = make_jaspr(param_function, return_shape=True)(0.5)
    assert hasattr(jaspr, 'jaxpr'), "Should work with parameters"
    
    # Test 5: Verify that flatten_envs still works with return_shape
    def env_function():
        qv = QuantumVariable(2)
        with QuantumEnvironment():
            h(qv[0])
        return measure(qv[0])
    
    jaspr, out_tree = make_jaspr(env_function, flatten_envs=True, return_shape=True)()
    assert hasattr(jaspr, 'jaxpr'), "Should work with flatten_envs"