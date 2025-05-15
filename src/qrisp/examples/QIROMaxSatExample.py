"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

# imports
from qrisp import QuantumVariable
import matplotlib.pyplot as plt
import networkx as nx
from qrisp.algorithms.qiro import *

# First we define a graph via the number of nodes and the QuantumVariable arguments

problem = [
    6,
    [[1, 2, -3], [1, 4, -6], [4, 5, 6], [1, 3, -4], [2, 4, 5], [1, 3, 5], [-2, -3, 6]],
]

qarg = QuantumVariable(problem[0])

# set simulator shots
mes_kwargs = {
    # below should be 5k
    "shots": 5000
}

# assign the correct new update functions for qiro from above imports
qiro_instance = QIROProblem(
    problem=problem,
    replacement_routine=create_maxsat_replacement_routine,
    cost_operator=create_maxsat_cost_operator_reduced,
    mixer=qiro_rx_mixer,
    cl_cost_function=create_maxsat_cl_cost_function,
    init_function=qiro_init_function,
)


# We run the qiro instance and get the results!
res_qiro = qiro_instance.run_qiro(
    qarg=qarg,
    depth=3,
    n_recursions=2,
    # mes_kwargs = mes_kwargs
)
# and also the final graph, that has been adjusted
final_Graph = qiro_instance.problem

# get the normal QAOA results for a comparison
# res_qaoa = maxclique_instance.run( qarg = qarg2, depth = 3)


testCostFun = create_maxsat_cl_cost_function(problem=problem)
print("QIRO 5 best results")
maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
for key, val in res_qiro.items():
    if key in maxfive:

        print(key)
        print(testCostFun({key: 1}))

print("final clauses")
print(final_Graph)
