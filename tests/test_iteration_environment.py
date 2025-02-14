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
import numpy as np

from qrisp import IterationEnvironment, auto_uncompute, z, h, QuantumBool, z, ry, QuantumFloat, merge, IterationEnvironment, recursive_qs_search, control
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
    
    
    # Test case from GitLab issue https://gitlab.fokus.fraunhofer.de/qompiler/qrisp/-/issues/96

    def oracle_function(qf):
        z(qf[0])


    def test(qf, oracle_function, iterations=1):

        merge(qf)
        qs = recursive_qs_search(qf)[0]
        qv_amount = len(qs.qv_list)

        for k in range(iterations):
            oracle_function(qf)
            if qv_amount != len(qs.qv_list):
                print(qs.qv_list)
                raise Exception("Applied oracle introducing new QuantumVariables without uncomputing/deleting")

    def test_IterEnv(qf, oracle_function, iterations=1):

        merge(qf)
        qs = recursive_qs_search(qf)[0]

        with IterationEnvironment(qs, iterations):
            oracle_function(qf)

    qf = QuantumFloat(1)
    reg = QuantumFloat(1)

    with control(reg[0]):
        test(qf, oracle_function)


    qf = QuantumFloat(1)
    reg = QuantumFloat(1)

    with control(reg[0]):
        test_IterEnv(qf, oracle_function)
        
        
    # Test interaction with custom controls
    

    qbl = QuantumBool()
    qf = QuantumFloat(4)
    qbl.flip()
    merge(qbl, qf)


    with control(qbl):
        with IterationEnvironment(qbl.qs, 4, precompile = False):
            qf += 1

    assert qf.qs.compile().depth() < 75

    