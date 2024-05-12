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
    with invert():
        cx(a[0], b[0])
    
    return measure(b[0])

from jax import make_jaxpr

jaxpr = make_jaxpr(test_function_1)(4)

# Jaxpr darstellen
print(jaxpr)
