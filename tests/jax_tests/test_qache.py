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
from qrisp.jax import *
from jax import make_jaxpr

def test_qache():
    
    class TracingCounter:
        def __init__(self):
            self.count = 0
    
        def increment(self):
            self.count += 1

    counter = TracingCounter()
    
    @qache
    def inner_function(qv):
        counter.increment()
        h(qv[0])
        cx(qv[0], qv[1])
        res_bl = measure(qv[0])
        return res_bl

    def outer_function():
        qv_0 = QuantumVariable(2)
        qv_1 = QuantumVariable(2)
        
        temp_0 = inner_function(qv_0)
        temp_1 = inner_function(qv_1)
        temp_2 = inner_function(qv_0)
        return temp_0 & temp_1 & temp_2
    
    print(make_jaxpr(outer_function)())
    
    assert counter.count == 1
    



