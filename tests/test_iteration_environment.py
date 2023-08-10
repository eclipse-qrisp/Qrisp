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

from qrisp import IterationEnvironment, auto_uncompute, z, h, QuantumFloat
from qrisp.grover import diffuser
def test_iteration_env():
    
    @auto_uncompute
    def sqrt_oracle(qf):
        temp_qbool = (qf*qf == 0.25)
        z(temp_qbool)

    
    
    n = 6
    iterations = int((2**n/2)**0.5)

    qf = QuantumFloat(n-1, -1, signed = True)
    h(qf)

    with IterationEnvironment(qf.qs, iterations):
        sqrt_oracle(qf)
        diffuser(qf)
        
    mes_res = list(qf.get_measurement().keys())
    assert set([mes_res[0], mes_res[1]]) == {0.5, -0.5}
        
    qf = QuantumFloat(n-1, -1, signed = True)
    h(qf)

    with IterationEnvironment(qf.qs, iterations, precompile = True):
        sqrt_oracle(qf)
        diffuser(qf)
        
    mes_res = list(qf.get_measurement().keys())
    assert set([mes_res[0], mes_res[1]]) == {0.5, -0.5}
    