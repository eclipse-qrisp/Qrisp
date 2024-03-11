# imports 
from qrisp.qiro.qiro_problem import QIROProblem
from qrisp.qaoa.problems.maxSatInfrastr import maxSatclCostfct, clausesdecoder
from qrisp.qiro.qiroproblems.qiroMaxSatInfrastr import * 
from qrisp.qiro.qiro_mixers import qiro_init_function, qiro_RXMixer
from qrisp import QuantumVariable
import matplotlib.pyplot as plt
import networkx as nx


problem = [6,[[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]]


decodedClauses = clausesdecoder( problem)
qarg = QuantumVariable(problem[0])

# set simulator shots
mes_kwargs = {
    #below should be 5k
    "shots" : 5000
    }


#assign cost_function and maxclique_instance, normal QAOA
testCostFun = maxSatclCostfct(problem)

# assign the correct new update functions for qiro from above imports
qiro_instance = QIROProblem(problem = problem,  
                            replacement_routine = create_maxSat_replacement_routine, 
                            cost_operator = create_maxSat_cost_operator_reduced,
                            mixer = qiro_RXMixer,
                            cl_cost_function = maxSatclCostfct,
                            init_function = qiro_init_function
                            )


# We run the qiro instance and get the results!
res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, 
                                  #mes_kwargs = mes_kwargs
                                  )

# and also the final graph, that has been adjusted
final_clauses = qiro_instance.problem


print("QIRO 5 best results")
maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
for key, val in res_qiro.items():  
    if key in maxfive:
        
        print(key)
        print(testCostFun({key:1}))

# or compare it with the networkx result of the max_clique algorithm...
# write brute force solution

# and finally, we draw the final graph and the original graphs to compare them!
print(final_clauses)