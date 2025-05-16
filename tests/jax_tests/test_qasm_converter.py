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

def test_qasm_converter():

    @qache
    def inner_function(qv):
        if isinstance(qv, QuantumFloat):
            x(qv)
        if isinstance(qv, QuantumBool):
            t(qv)

    def main(i):
        qf = QuantumFloat(i)
        qbl = QuantumBool()

        inner_function(qf)
        inner_function(qbl)
        
        with invert():    
            inner_function(qf)
            inner_function(qbl)
            
        inner_function(qf)
        inner_function(qbl)

        return measure(qf), measure(qbl)

    t0 = time.time()
    jaspr = make_jaspr(main)(1)
    
    qasm_str = jaspr.to_qasm(3)
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    assert qc.run() == {'0111': 1.0}
    
    qasm_str = jaspr.to_qasm(5)
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    assert qc.run() == {'011111': 1.0}

    