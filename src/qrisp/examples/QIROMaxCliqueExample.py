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
num_nodes = 15
G = nx.erdos_renyi_graph(num_nodes,0.7, seed =  99)
Gtwo = nx.erdos_renyi_graph(num_nodes,0.7, seed =  99)
qarg = QuantumVariable(G.number_of_nodes())
qarg2 = QuantumVariable(Gtwo.number_of_nodes())

# set simulator shots
mes_kwargs = {
    #below should be 5k
    "shots" : 5000
    }

#assign cost_function and maxclique_instance, normal QAOA
testCostFun = create_max_clique_cl_cost_function(Gtwo)
maxclique_instance = QAOAProblem(create_max_indep_set_mixer(G), RX_mixer, create_max_clique_cl_cost_function(G))

# assign the correct new update functions for qiro from above imports
qiro_instance = QIROProblem(problem = Gtwo,  
                            replacement_routine = create_max_clique_replacement_routine, 
                            cost_operator = create_max_clique_cost_operator_reduced,
                            mixer = qiro_rx_mixer,
                            cl_cost_function = create_max_clique_cl_cost_function,
                            init_function = qiro_init_function
                            )


# We run the qiro instance and get the results!
res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, 
                                  #mes_kwargs = mes_kwargs
                                  )
# and also the final graph, that has been adjusted
final_Graph = qiro_instance.problem

# get the normal QAOA results for a comparison
#res_qaoa = maxclique_instance.run( qarg = qarg2, depth = 3)


print("QIRO 5 best results")
maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
for key, val in res_qiro.items():  
    if key in maxfive:
        
        print(key)
        print(testCostFun({key:1}))

# or compare it with the networkx result of the max_clique algorithm...
print("Networkx solution")
print(nx.approximation.max_clique(Gtwo))

# and finally, we draw the final graph and the original graphs to compare them!
plt.figure(1)
nx.draw(final_Graph, with_labels = True)

plt.figure(2)
nx.draw(G, with_labels = True)
plt.show()  
