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

from qrisp import QuantumVariable, QuantumArray
from qrisp.qaoa import QUBO_problem, QUBO_obj
from operator import itemgetter
import numpy as np

def test_QUBO():
    from qrisp.default_backend import def_backend

    Q1 = np.array(
        [
            [1,0,0,0],
            [0,0,-0.5,0.5],
            [0,-0.5,0,1],
            [0,0.5,1,-2],
        ]
    )
    qarg = QuantumArray(qtype = QuantumVariable(1), shape = len(Q1))
    QUBO_instance = QUBO_problem(Q1)
    res = QUBO_instance.run(qarg, depth=1, mes_kwargs={"backend" : def_backend, "shots" : 100000}, max_iter = 50)
    best_cost1, res_str = min([(QUBO_obj(bitstring, Q1), bitstring) for bitstring in list(res.keys())], key=itemgetter(0))

    assert best_cost1 == -2.0

    Q2 = np.array(
        [
            [1,-3,-3,-3],
            [-3,1,0,0],
            [-3,0,1,-3],
            [-3,0,-3,1],
        ]
    )
    
    qarg_prep = QuantumArray(qtype = QuantumVariable(1), shape = len(Q2))
    QUBO_instance = QUBO_problem(Q2)
    res = QUBO_instance.run(qarg, depth=1, mes_kwargs={"backend" : def_backend, "shots" : 100000}, max_iter = 50)
    best_cost2, res_str = max([(QUBO_obj(bitstring, Q2), bitstring) for bitstring in list(res.keys())], key=itemgetter(0))

    assert best_cost2 == 2

    Q3 = np.array(
        [
            [1,-6,-6,-6],
            [0,1,0,0],
            [0,0,1,-6],
            [0,0,0,1],
        ]
    )

    qarg = QuantumArray(qtype = QuantumVariable(1), shape = len(Q3))
    QUBO_instance = QUBO_problem(Q3)
    res = QUBO_instance.run(qarg, depth=1, mes_kwargs={"backend" : def_backend, "shots" : 100000}, max_iter = 50)
    best_cost3, res_str = max([(QUBO_obj(bitstring, Q2), bitstring) for bitstring in list(res.keys())], key=itemgetter(0))

    assert best_cost3 == best_cost2
    
