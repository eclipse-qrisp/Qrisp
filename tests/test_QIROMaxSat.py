# imports 
from qrisp.qiro.qiro_problem import QIROProblem
from qrisp.qaoa.problems.maxSatInfrastr import maxSatclCostfct, clausesdecoder
from qrisp.qiro.qiroproblems.qiroMaxSatInfrastr import * 
from qrisp.qiro.qiro_mixers import qiro_init_function, qiro_RXMixer
from qrisp import QuantumVariable
import matplotlib.pyplot as plt
import networkx as nx

def test_QIROmaxSat():
    # define the problem according to maxSat encoding
    problem = [8 , [[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6], [1,7,8], [3,-7,-8], [3,4,8],[4,5,8], [1,2,7]]]
    decodedClauses = clausesdecoder( problem)

    qarg = QuantumVariable(problem[0])

    # set simulator shots
    mes_kwargs = {
        #below should be 5k
        "shots" : 5000
        }


    # assign cost_function and maxclique_instance, normal QAOA
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



    import itertools
    def testCostFun(state):
        obj = 0
        #literally just do a minus 1 op if state is equiv to a given condition
        for index in range(len(decodedClauses)):
            if state in decodedClauses[index]:
                obj -= 1 

        return obj


    #print 5 best sols 
    # assert that sol is in decoded clauses 
    maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
    # assert that sol is in decoded clauses 
    for name in res_qiro.keys():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name in maxfive:
            print(name)
            print(testCostFun(name))
        if testCostFun(name) < 0:
            temp = False
            for index in range(len(decodedClauses)):
                if name in decodedClauses[index]:
                    temp  = True
            assert temp

