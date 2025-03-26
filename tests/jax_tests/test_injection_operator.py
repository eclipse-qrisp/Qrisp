"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import *
from qrisp.jasp import *
from jax import make_jaxpr

def test_injection_operator():
    
    @jaspify
    def main(i):
        
        a = QuantumFloat(i)
        b = QuantumFloat(i)
        
        h(a)
        h(b)
        
        s = a*b
        
        with invert():
            (s << (lambda a, b : a*b))(a,b)
        
        return measure(s)
    
    for i in range(2, 6):
        assert main(i) == 0
        
    def AND(a, b):
        res = QuantumBool()
        mcx([a, b], res)
        return res

    @jaspify
    def main():
        a = QuantumBool()
        b = QuantumBool()

        tar = QuantumBool()

        (tar << AND)(a,b)

        res = measure(tar)
        return res

    main()
