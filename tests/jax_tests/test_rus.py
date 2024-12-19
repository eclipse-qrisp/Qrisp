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
from qrisp.jasp import *
from jax import make_jaxpr

def test_rus():
    
    @RUS
    def rus_trial_function():
        qf = QuantumFloat(5)
        h(qf[0])

        for i in range(1, 5):
            cx(qf[0], qf[i])

        cancelation_bool = measure(qf[0])
        return cancelation_bool, qf

    def call_RUS_example():

        qf = rus_trial_function()

        return measure(qf)

    jaspr = make_jaspr(call_RUS_example)()
    assert jaspr() == 31
    # Yields, 31 which is the decimal version of 11111
    
    # More complicated example
    
    def test_function():
        
        @RUS
        def trial_function():
            a = QuantumFloat(5)
            b = QuantumFloat(5)
            qbl = QuantumBool()
            a[:] = 10
            h(qbl[0])
            
            with control(qbl[0]):
                jasp_mod_adder(a, b, 7, inpl_adder = jasp_fourier_adder)
            
            return measure(qbl[0]), b
        
        
        res = trial_function()
        jasp_fourier_adder(5, res)
        
        return measure(res)
    jaspr = make_jaspr(test_function)()
    
    assert jaspr() == 8
    



