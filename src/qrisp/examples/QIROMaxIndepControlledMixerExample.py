# imports 
from qrisp.algorithms.qiro import * 
from qrisp import QuantumVariable
import networkx as nx


# First we define a graph via the number of nodes and the QuantumVariable arguments
num_nodes = 17
G = nx.erdos_renyi_graph(num_nodes, 0.3, seed =  107)
qarg = QuantumVariable(G.number_of_nodes())


# assign the correct new update functions for qiro from above imports
qiro_instance = QIROProblem(G, 
                            create_max_indep_replacement_routine,
                            qiro_rz_mixer,
                            create_max_indep_controlled_mixer_reduced,
                            create_max_indep_set_cl_cost_function,
                            qiro_max_indep_set_init_function,
                            )

# We run the qiro instance and get the results!
#res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2)


res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2)
# and also the final graph, that has been adjusted
final_Graph = qiro_instance.problem

# Lets see what the 5 best results are
print("QIRO 5 best results")
maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
costFunc = create_max_indep_set_cl_cost_function(G)
for key, val in res_qiro.items():  
    if key in maxfive:
        print(key)
        print(costFunc({key:1}))
        

# and compare them with the networkx result of the max_clique algorithm, where we might just see a better result than the heuristical NX algorithm!
print("Networkx solution")
print(nx.approximation.maximum_independent_set(G))
nx.draw(G, with_labels=True,)
plt.show()

