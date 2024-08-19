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

from jax import make_jaxpr
from qrisp import QuantumVariable, cx, QuantumCircuit
from qrisp.jisp import qache, flatten_pjit, make_jispr

def test_qc_converter():
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        cx(qv[0], qv[1])
        cx(qv[1], qv[2])
        cx(qv[1], qv[2])
    
    for i in range(3, 7):
        
        jispr = make_jispr(test_function)(i)
        qc = jispr.extract_qc(i)
        
        comparison_qc = QuantumCircuit(i)
        comparison_qc.cx(0, 1)
        comparison_qc.cx(1, 2)
        comparison_qc.cx(1, 2)
        
        assert qc.compare_unitary(comparison_qc)
    
    @qache
    def inner_function(qv, i):
        cx(qv[i], qv[i+1])
        cx(qv[i+2], qv[i+1])
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        inner_function(qv, 0)
        inner_function(qv, 1)
        inner_function(qv, 2)
        
    
    jispr = make_jispr(test_function)(5)
    qc = jispr.extract_qc(5)
    
    assert len(qc.data) == 3 + len(qc.qubits)
    print(qc)
    comparison_qc = QuantumCircuit(5)
    
    comparison_qc.cx(0, 1)
    comparison_qc.cx(2, 1)
    comparison_qc.cx(1, 2)
    comparison_qc.cx(3, 2)
    comparison_qc.cx(2, 3)
    comparison_qc.cx(4, 3)
    
    print(comparison_qc)
    
    assert qc.compare_unitary(comparison_qc)
    
    jispr = make_jispr(test_function)(5)
    flattened_jispr = flatten_pjit(jispr)
    
    qc = flattened_jispr.extract_qc(5)
    
    assert qc.compare_unitary(comparison_qc)
    
    
    
    