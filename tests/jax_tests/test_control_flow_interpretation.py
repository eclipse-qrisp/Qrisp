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

def test_control_flow_interpretation():
    
    @qache
    def inner_function(qf, i):
        qf[:] = i

    def test_f(i):
        a = QuantumFloat(5, -1)
        with invert():
            with QuantumEnvironment():
                with invert():
                    inner_function(a, i)
        b = measure(a)
        b += 4
        return b
    
    jisp_program = make_jispr(test_f)(0.5)
    
    eval_function = jisp_interpreter(jisp_program)
    
    for i in range(5):
        res = eval_function(i + 0.5)
        
        assert res == i + 4.5
