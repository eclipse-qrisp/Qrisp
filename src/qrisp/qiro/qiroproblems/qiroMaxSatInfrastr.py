# just write a replacement_routine and throw out not relevant sets

from qrisp import rz, x, cx
import numpy as np
import copy
from qrisp.qiro.qiroproblems.qiro_utils import * 

def create_maxSat_replacement_routine( res, oldproblem, solutions, exclusions):
    
    """
    Creates a replacement routine for the problem structure, i.e. defines the replacement rules. 
    See the original paper for description of the update rules

    Parameters
    ----------
    res : dict
        Result dictionary of initial QAOA optimization procedure.
    oldproblem : List
        The clauses defining the problem instance.
    solutions : List
        Qubits which have been found to be positive correlated, i.e. part of the problem solution.
    exclusions : List
        Qubits which have been found to be negative correlated, i.e. not part of the problem solution, or contradict solution qubits in accordance to the update rules.  

    Returns
    -------
    new_problem: list
        Updated clauses of the problem instance.
    solutions : List
        Updated set of solutions to the problem.
    sign : Int
        The sign of the correlation.
    exclusions : List
        Updated set of exclusions to the problem.

    Examples:
    ---------

    In the example below we show how to employ the structure of this class for a MIS problem and compare it it a normal QAOA implementation

    ::

        # imports 
        from qrisp.qiro.qiro_problem import QIROProblem
        from qrisp.qaoa.problems.maxSatInfrastr import maxSatclCostfct, clausesdecoder
        from qrisp.qiro.qiroproblems.qiroMaxSatInfrastr import * 
        from qrisp.qiro.qiro_mixers import qiro_init_function, qiro_RXMixer
        from qrisp import QuantumVariable
        import matplotlib.pyplot as plt
        import networkx as nx

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


        print("QIRO 5 best results")
        maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
        for key, val in res_qiro.items():  
            if key in maxfive:
                
                print(key)
                print(testCostFun({key:1}))

        # or compare it with the networkx result of the max_clique algorithm...
        # or a write brute force solution, this is up to you

        # and finally, we print the final clauses!
        final_clauses = qiro_instance.problem
        print(final_clauses)


    
    """
    
    numVars = oldproblem[0]
    clauses = copy.deepcopy(oldproblem[1])

    # FOR SINGLE QUBIT CORRELATIONS
    # make -1 here for consistency in the find max
    orig_nodes = [i - 1 for i in list(range(1,numVars + 1)) if not i in exclusions]
    combinations = []

    # add check for solution nodes      
    for index1 in range(len(orig_nodes)-1):
        for index2 in range(index1, len(orig_nodes)):
            combinations.append((orig_nodes[index1],orig_nodes[index2]))
    

    max_item, sign = find_max(orig_nodes, combinations, res, solutions)

    # we just directly remove clauses from clauses
    if isinstance(max_item, int):
        
        max_item += 1

        if sign > 0:
            for item in clauses: 
                # if negation in item --> remove it
                if -1 * max_item in item:
                    clauses.remove(item)
            solutions.append(max_item)
            exclusions.append(max_item)

        elif sign < 0:
            for item in clauses:
                # if number in item --> remove it
                if max_item in item:
                    clauses.remove(item)
            exclusions.append(max_item)

    else:
        max_item[0] += 1 
        max_item[1] += 1 
        if sign > 0:
            for item in clauses:
                # replace with pos. correlated number if its in an item
                if max_item[1] in item:
                    temp = item.index[max_item[1]]
                    item[temp] = max_item[0]
                if -1* max_item[1] in item:
                    temp = item.index[-1* max_item[1]]
                    item[temp] = -1* max_item[0]
                exclusions.append(max_item[1])
                    

        elif sign < 0:
            for item in clauses:
                # replace with neg. correlated number if its in an item
                if max_item[1] in item:
                    temp = item.index[max_item[1]]
                    item[temp] = -1 * max_item[0]
                if -1* max_item[1] in item:
                    temp = item.index[ -1  * max_item[1]]
                    item[temp] = max_item[0]
                exclusions.append(max_item[1])

    # create sign list somewhere?
    return [numVars, clauses], solutions, sign, exclusions


def create_maxSat_cost_operator_reduced(problem, solutions=[]):

    """
    |  Implementation for MaxSat Cost-Operator, in accordance to the original QAOA-paper.
    |  Adjusted for QIRO to also respect found solutions to the problem
    
    Parameters
    ----------
    clauses : List(Lists)
        Clauses to be considered for MaxSat. Should be given as a list of lists and 1-indexed instead 0-indexed, see example
    solutions : List
        Variables that were found to be a part of the solution.
    Returns
    -------
    cost_operator : function
        the Operator applied to the circuit-QuantumVariable

    Examples
    --------
    >>> clauses11 = [[1,2,-3], [1,4,-6], [4,5,6], [1,3,-4], [2,4,5], [1,3,5], [-2,-3,6]]
    
    |  Explanation: 
    |  Clause [1, 2, -4] is fulfilled by the QuantumStates "1100", "1110" 

    * if positive sign: the index has to be "1", if negative sign the index has to be "0".
    * indices not mentioned can be "0" or "1" (the third integer in this case).
    * start with 1 in your clauses! because ``int 0`` has no sign and this problematic. 
    * we want to stay with mathematical formulation, so no handing over strings!

    """
    
    clauses = problem[1]
    satlen = len(clauses[0])

    def maxSatcostEmbed(qv , gamma):
        import itertools
        #clauses = clauses
        
        qc = qv

        for numClause in range(len(clauses)):
            #go through clauses
            clause = clauses[numClause]
        
            for lenofCombination in range(1,satlen+1):
                # get all combinations to assign rz, rzz, rzzz, rzzzzzz....
                combinations = list(itertools.combinations(clause, lenofCombination))
                #print(combinations)
                
                #len == 1: just rz
                if lenofCombination == 1:
                    for index in range(len(combinations)):
                        if  abs( combinations[index][0])-1 not in solutions:
                            # sign for rz mixer
                            signu = - np.sign(combinations[index][0])
                            # always have the "-1" at the end since clauses are not initiated with 0, see above
                            #print(abs( combinations[index][0] - 1 ))
                            rz(signu *  gamma/8, qc[ abs( combinations[index][0]) -1 ] )

                else:
                    #for all combinations of this length 
                    for index in range(len(combinations)):
                        signu = 1

                        # up to final value in combination perform cx gates --> set up the rzz, rzzz, rzzzz ... gates
                        for index2 in range(lenofCombination-1):
                            # (also remember the sign)
                            signu *= - np.sign(combinations[index][index2])
                            cx(qc[abs( combinations[index][index2] ) -1], qc[abs( combinations[index][index2+1] ) -1])

                        signu *= np.sign(combinations[index][lenofCombination-1])
                        # finalize rzz, rzzz, rzzzz ... gates
                        rz(signu*gamma/8, qc[abs( combinations[index][lenofCombination-1] ) -1])
                        
                        # and cnot gates back 
                        for index2 in reversed(range(lenofCombination-1)):
                            cx(  qc[abs(combinations[index][index2] ) -1], qc[abs(combinations[index][index2+1] ) -1 ]) 

        return qc
    
    return maxSatcostEmbed


