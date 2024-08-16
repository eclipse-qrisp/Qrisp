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

def test_environment_compilation():
    
    def outer_function(x):
        qv = QuantumVariable(x)
        with invert():
            with invert():
                cx(qv[0], qv[1])
                h(qv[0])
        return qv

    jispr = make_jispr(outer_function)(2)
    jispr = flatten_environments(jispr)
    
    jisp_function_test(outer_function)
    
    @qache
    def inner_function(qv):
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[0], qv[qv.size-1])
        return qv.size
    
    def outer_function(x):
        qv = QuantumVariable(x)
        
        with invert():
            inner_function(qv)
            with invert():
                inner_function(qv)
        temp_1 = inner_function(qv)
        temp_2 = inner_function(qv)
        return qv
    
    testing_function = jisp_function_test(outer_function)
    
    assert testing_function(5)
    assert testing_function(6)
    assert testing_function(7)
    assert testing_function(8)
    
    