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

from qrisp.qaoa import QAOAProblem
from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct,maxCliqueCostOp,init_state
from qrisp.qaoa.mixers import RX_mixer
from qrisp import QuantumVariable
import networkx as nx
import matplotlib.pyplot as plt




"""
This is an example implementation of maxClique Problem instance solved with QAOA
The problem is structured as follows:
> Given a Graph G = (V,E) maximize the size of a clique G (i.e. a subset V' of V, which a subgraph in which all pairs of vertices are adjacent).

We will not stick to mathematical assignment of variable names. 

 """

giraf = create_rdm_graph(9,0.7, seed =  133)
#draw graph
#nx.draw(giraf,with_labels = True)
#plt.show() 


qarg = QuantumVariable(giraf.number_of_nodes())



#Instanciate QAOA
#mixer gets Graph as argument
#cost function gets graph as argument 
QAOAinstance = QAOAProblem(cost_operator= maxCliqueCostOp(giraf), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(giraf)) 
QAOAinstance.set_init_function(init_function=init_state)
theNiceQAOA = QAOAinstance.run(qarg=qarg, depth= 5)


import itertools
def aClcostFct(state, G):
    # we assume solution is right
    temp = True
    energy = 0 
    intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
    # get all combinations of vertices in graph that are marked as |1> by the solution 
    combinations = list(itertools.combinations(intlist, 2))
    # if any combination is found in the list of G.edges(), the solution is wrong, and energy == 0
    for combination in combinations:
        if combination not in G.edges():
            temp = False
    # else we just add the number of marked as |1> nodes
    if temp: 
        energy = -len(intlist)
    return(energy)

    

#print the ideal solutions
print("5 most likely Solutions") 
maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
for name, age in theNiceQAOA.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if name in maxfive:

        print((name, age))
        print(aClcostFct(name, giraf))

print("NX solution")
print(nx.max_weight_clique(giraf, weight = None))

#draw graph
nx.draw(giraf,with_labels = True)
plt.show() 



