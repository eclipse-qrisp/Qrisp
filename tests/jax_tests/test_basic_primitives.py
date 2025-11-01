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
             'jasp.reset',
             'jasp.delete_qubits',
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
                 'jasp.measure',
                 'jasp.reset',
                 'jasp.delete_qubits'])    

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
        
    
        
    
    