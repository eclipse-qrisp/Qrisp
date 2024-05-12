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

from qrisp.jax import qfunc_def
from qrisp import *

def test_function_1(i):
    
    @qfunc_def
    def fan_out(a, b):
        
        cx(a[0], b[0])
        cx(a[0], b[1])
        rz(0.5, a[0])

    a = QuantumVariable(i)
    b = QuantumVariable(i+1)
    
    fan_out(a,b)
    fan_out(a,b)
    
    return measure(b[0])

from jax import make_jaxpr

jaxpr = make_jaxpr(test_function_1)(4)

# Jaxpr darstellen
print(jaxpr)
"""
{ lambda ; a:i32[]. let
    b:QuantumCircuit = qdef 
    c:QuantumCircuit d:QubitArray = create_qubits b a
    e:i32[] = add a 1
    f:QuantumCircuit g:QubitArray = create_qubits c e
    h:QuantumCircuit i:QubitArray j:QubitArray = qdef[num_args=2] 
    k:Qubit = get_qubit i 0
    l:Qubit = get_qubit j 0
    m:QuantumCircuit = cx h k l
    n:Qubit = get_qubit i 0
    o:Qubit = get_qubit j 1
    p:QuantumCircuit = cx m n o
    q:QuantumCircuit = qcall f p d g
    r:QuantumCircuit = qcall q p d g
    s:Qubit = get_qubit g 0
    _:QuantumCircuit t:bool[] = measure r s
  in (t,) }
"""

# Liste von eqns
eqns = jaxpr.eqns

# Einzelne equation
eqn = eqns[7]
print(eqn)
# a:QuantumCircuit = cx b c d

invars = eqn.invars
print(invars)
# [h, k, l]

invars = eqn.outvars

primitive = eqn.primitive
print(primitive)
# cx

# Gatter-Primitive sind qrisp.Operation Objekte => Unitary via get_unitary
print(primitive.get_unitary())
"""
[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
 [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
 [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
 [0.+0.j 0.+0.j 1.+0.j 0.+0.j]]
"""
pass