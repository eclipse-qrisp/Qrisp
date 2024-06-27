"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
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

    assert str(make_jaxpr(test_function)()) == '{ \x1b[34m\x1b[22m\x1b[1mlambda \x1b[39m\x1b[22m\x1b[22m; . \x1b[34m\x1b[22m\x1b[1mlet\n    \x1b[39m\x1b[22m\x1b[22ma\x1b[35m:QuantumCircuit\x1b[39m = qdef \n    b\x1b[35m:QuantumCircuit\x1b[39m c\x1b[35m:QubitArray\x1b[39m = create_qubits a 2\n    d\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    e\x1b[35m:QuantumCircuit\x1b[39m = h b d\n    f\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    g\x1b[35m:Qubit\x1b[39m = get_qubit c 1\n    h\x1b[35m:QuantumCircuit\x1b[39m = cx e f g\n    i\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    _\x1b[35m:QuantumCircuit\x1b[39m j\x1b[35m:bool[]\x1b[39m = measure h i\n  \x1b[34m\x1b[22m\x1b[1min \x1b[39m\x1b[22m\x1b[22m(j,) }'
    
    
    def test_function():
        qv = QuantumVariable(2)
        
        with QuantumEnvironment():
            h(qv[0])
            cx(qv[0], qv[1])
        
        res_bl = measure(qv[0])
        return res_bl
    
    assert str(make_jaxpr(test_function)()) == '{ \x1b[34m\x1b[22m\x1b[1mlambda \x1b[39m\x1b[22m\x1b[22m; . \x1b[34m\x1b[22m\x1b[1mlet\n    \x1b[39m\x1b[22m\x1b[22ma\x1b[35m:QuantumCircuit\x1b[39m = qdef \n    b\x1b[35m:QuantumCircuit\x1b[39m c\x1b[35m:QubitArray\x1b[39m = create_qubits a 2\n    d\x1b[35m:QuantumCircuit\x1b[39m = quantumenvironment[stage=enter] b\n    e\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    f\x1b[35m:QuantumCircuit\x1b[39m = h d e\n    g\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    h\x1b[35m:Qubit\x1b[39m = get_qubit c 1\n    i\x1b[35m:QuantumCircuit\x1b[39m = cx f g h\n    j\x1b[35m:QuantumCircuit\x1b[39m = quantumenvironment[stage=exit] i\n    k\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    _\x1b[35m:QuantumCircuit\x1b[39m l\x1b[35m:bool[]\x1b[39m = measure j k\n  \x1b[34m\x1b[22m\x1b[1min \x1b[39m\x1b[22m\x1b[22m(l,) }'


    def test_function():
        qv = QuantumVariable(2)
        
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[1], qv[0])
        h(qv[1])
        res_bl = measure(qv[0])
        qv.delete()
        return res_bl
    
    assert str(make_jaxpr(test_function)()) == '{ \x1b[34m\x1b[22m\x1b[1mlambda \x1b[39m\x1b[22m\x1b[22m; . \x1b[34m\x1b[22m\x1b[1mlet\n    \x1b[39m\x1b[22m\x1b[22ma\x1b[35m:QuantumCircuit\x1b[39m = qdef \n    b\x1b[35m:QuantumCircuit\x1b[39m c\x1b[35m:QubitArray\x1b[39m = create_qubits a 2\n    d\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    e\x1b[35m:QuantumCircuit\x1b[39m = h b d\n    f\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    g\x1b[35m:Qubit\x1b[39m = get_qubit c 1\n    h\x1b[35m:QuantumCircuit\x1b[39m = cx e f g\n    i\x1b[35m:Qubit\x1b[39m = get_qubit c 1\n    j\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    k\x1b[35m:QuantumCircuit\x1b[39m = cx h i j\n    l\x1b[35m:Qubit\x1b[39m = get_qubit c 1\n    m\x1b[35m:QuantumCircuit\x1b[39m = h k l\n    n\x1b[35m:Qubit\x1b[39m = get_qubit c 0\n    o\x1b[35m:QuantumCircuit\x1b[39m p\x1b[35m:bool[]\x1b[39m = measure m n\n    _\x1b[35m:QuantumCircuit\x1b[39m = delete_qubits o c\n  \x1b[34m\x1b[22m\x1b[1min \x1b[39m\x1b[22m\x1b[22m(p,) }'