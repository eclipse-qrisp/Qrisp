# imports 
from qrisp.qiro.qiro_problem import QIROProblem
from qrisp.qaoa.qaoa_problem import QAOAProblem
from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct, maxCliqueCostOp
from qrisp.qiro.qiroproblems.qiroMaxCliqueInfrastr import * 
from qrisp.qaoa.mixers import RX_mixer
from qrisp.qiro.qiro_mixers import qiro_init_function, qiro_RXMixer
from qrisp import QuantumVariable
import matplotlib.pyplot as plt
import networkx as nx

def test_QIRO():

    # First we define a graph via the number of nodes and the QuantumVariable arguments
    num_nodes = 15
    G = nx.erdos_renyi_graph(num_nodes,0.7, seed =  99)
    Gtwo = nx.erdos_renyi_graph(num_nodes,0.7, seed =  99)
    qarg = QuantumVariable(G.number_of_nodes())
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


    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, 
                                    #mes_kwargs = mes_kwargs
                                    )

    #QIRO 5 best results
    maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]

    import itertools
    for key, val in res_qiro.items():  
        if key in maxfive:
            cost = testCostFun({key:1})
            temp = True
            if cost < 0:
                intlist = [s for s in range(len(list(key))) if list(key)[s] == "1"]
                combinations = list(itertools.combinations(intlist, 2))
                # if any combination is found in the list of G.edges(), the solution is wrong, and energy == 0
                for combination in combinations:
                    if combination not in G.edges():
                        temp = False
                        break
            assert temp

