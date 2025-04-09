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

def test_template():
    
    @jaspify
    def main():
        
        qv = QuantumFloat(4, 1)
        template = qv.template()
        
        @quantum_kernel
        def inner_function(template):
            qv = template.construct()
            x(qv[0])
            return measure(qv)
        
        return inner_function(template)

    assert main() == 2

    @jaspify
    def main():
        
        qv = QuantumFloat(4, 1)
        template = qv.template()
        
        @quantum_kernel
        def inner_function():
            qv = template.construct()
            x(qv[0])
            return measure(qv)
        
        return inner_function()

    assert main() == 2

    qv = QuantumFloat(4, 1)
    template = qv.template()

    @jaspify
    def main():
        
        @quantum_kernel
        def inner_function():
            qv = template.construct()
            x(qv[0])
            return measure(qv)
        
        return inner_function()

    assert main() == 2