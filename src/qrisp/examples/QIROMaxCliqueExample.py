# imports 
from qrisp.qiro.qiro_problem import QIROProblem
from qrisp.qaoa.qaoa_problem import QAOAProblem
from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct, maxCliqueCostOp
from qrisp.qiro.qiroproblems.qiroMaxCliqueInfrastr import * 
from qrisp.qaoa.mixers import RX_mixer
from qrisp.qiro.qiro_mixers import qiro_init_function, qiro_RXMixer
from qrisp import QuantumVariable
import matplotlib.pyplot as plt
import networkx as nx



# First we define a graph via the number of nodes and the QuantumVariable arguments
num_nodes = 15
G = create_rdm_graph(num_nodes,0.7, seed =  99)
Gtwo = create_rdm_graph(num_nodes,0.7, seed =  99)
qarg = QuantumVariable(G.number_of_nodes())
qarg2 = QuantumVariable(Gtwo.number_of_nodes())

# set simulator shots
mes_kwargs = {
    #below should be 5k
    "shots" : 5000
    }

#assign cost_function and maxclique_instance, normal QAOA
testCostFun = maxCliqueCostfct(Gtwo)
maxclique_instance = QAOAProblem(maxCliqueCostOp(G), RX_mixer, maxCliqueCostfct(G))

# assign the correct new update functions for qiro from above imports
qiro_instance = QIROProblem(problem = Gtwo,  
                            replacement_routine = create_maxClique_replacement_routine, 
                            cost_operator = create_maxClique_cost_operator_reduced,
                            mixer = qiro_RXMixer,
                            cl_cost_function = maxCliqueCostfct,
                            init_function = qiro_init_function
                            )


# We run the qiro instance and get the results!
res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, 
                                  #mes_kwargs = mes_kwargs
                                  )
# and also the final graph, that has been adjusted
final_Graph = qiro_instance.problem

# get the normal QAOA results for a comparison
res_qaoa = maxclique_instance.run( qarg = qarg2, depth = 3)


# We can then also print the top 5 results for each...
print("QAOA 5 best results")
maxfive = sorted(res_qaoa, key=res_qaoa.get, reverse=True)[:5]
for key, val in res_qaoa.items(): 
    if key in maxfive:
        print(key)
        print(testCostFun({key:1}))


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
