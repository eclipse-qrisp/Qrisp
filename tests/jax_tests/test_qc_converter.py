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
from qrisp.jax import jaxpr_to_qc, qache

def test_converter():
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        cx(qv[0], qv[1])
        cx(qv[1], qv[2])
        cx(qv[1], qv[2])
    
    for i in range(3, 7):
        
        jaxpr = make_jaxpr(test_function)(i)
        
        qc, = jaxpr_to_qc(jaxpr)(i)
        
        comparison_qc = QuantumCircuit(i)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(1, 2)
        
        assert qc.compare_unitary(comparison_qc)
    
    @qache
    def inner_function(qv):
        cx(qv[0], qv[1])
        cx(qv[2], qv[1])
    
    def test_function(i):
        qv = QuantumVariable(i)
        
        inner_function(qv)
        inner_function(qv)
        inner_function(qv)
        
        
    jaxpr = make_jaxpr(test_function)(i)
    
    qc, = jaxpr_to_qc(jaxpr)(i)
    
    assert len(qc.data) == 3