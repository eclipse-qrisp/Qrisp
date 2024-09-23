
from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetclCostfct, maxIndepSetCostOp
from qrisp.qiro import QIROProblem, trafo_maxPackToMIS, qiro_init_function, qiro_RXMixer, create_maxIndep_replacement_routine, create_maxIndep_cost_operator_reduced
from qrisp import QuantumVariable
import networkx as nx


import random
random.seed(105)
# sets are given as list of lists
#sets = [[0,7,1],[6,5],[2,3],[5,4],[8,7,0], [2,4,7],[1,3,4],[7,9],[1,9],[1,3,8],[4,9],[0,7,9],[0,4,8],[1,4,8]]
# full universe is given as a number of nodes - 1
sol = 12

#create random sets:
sets = []
candidates = list(range(sol))
for index in range(15):
    clause = []
    for index2 in range(3):
        temp = random.choice(candidates)
        candidates.remove(temp)
        clause.append(temp )
    sets.append(clause)
    candidates = list(range(sol))

problem = [sol, sets]
print(sets)

G = trafo_maxPackToMIS(problem=problem)
qarg = QuantumVariable(G.number_of_nodes())

# set simulator shots
mes_kwargs = {
    #below should be 5k
    "shots" : 5000
    }

# assign the correct new update functions for qiro from above imports
qiro_instance = QIROProblem(G, 
                            replacement_routine=create_maxIndep_replacement_routine, 
                            cost_operator= create_maxIndep_cost_operator_reduced,
                            mixer= qiro_RXMixer,
                            cl_cost_function= maxIndepSetclCostfct,
                            init_function= qiro_init_function
                            )

# We run the qiro instance and get the results!
res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, mes_kwargs = mes_kwargs)
# and also the final graph, that has been adjusted
final_Graph = qiro_instance.problem

# Lets see what the 5 best results are
print("QIRO 5 best results")
maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
costFunc = maxIndepSetclCostfct(G)
for key, val in res_qiro.items():  
    if key in maxfive:
        
        print(key)
        print(costFunc({key:1}))
        

# and compare them with the networkx result of the max_clique algorithm, where we might just see a better result than the heuristical NX algorithm!
print("Networkx solution")
print(nx.approximation.maximum_independent_set(G))
