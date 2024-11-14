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

# just write a replacement_routine and throw out not relevant sets

from qrisp import rz, x, cx
import numpy as np
import math
import copy
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import * 


def create_maxsat_replacement_routine(res, problem, solutions=[], exclusions=[]):
    """
    Creates a replacement routine for the problem structure, i.e., defines the replacement rules. 
    See the `original paper <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327>`_ for a description of the update rules.

    Parameters
    ----------
    res : dict
        Result dictionary of QAOA optimization procedure.
    problem : tuple(int, list[list[int]])
        The number of variables and a list of clauses for the MaxSat instance.
    solutions : list
        Qubits which were found to be positively correlated, i.e., part of the problem solution.
    exclusions : list
        Qubits which were found to be negatively correlated, i.e., not part of the problem solution, or contradict solution qubits in accordance with the update rules.  

    Returns
    -------
    new_problem: list
        Updated clauses of the problem instance.
    solutions : list
        Updated set of solutions to the problem.
    sign : int
        The sign of the correlation.
    exclusions : list
        Updated set of exclusions to the problem.

    """
    
    numVars = problem[0]
    clauses = copy.deepcopy(problem[1])

    # FOR SINGLE QUBIT CORRELATIONS
    # make -1 here for consistency in the find max
    orig_nodes = [i - 1 for i in list(range(1,numVars + 1)) if not i in exclusions]
    combinations = []

    # add check for solution nodes      
    for index1 in range(len(orig_nodes)-1):
        for index2 in range(index1, len(orig_nodes)):
            combinations.append((orig_nodes[index1],orig_nodes[index2]))
    

    max_item, sign = find_max(orig_nodes, combinations, res, solutions)
    if max_item == None:
        return problem, solutions, 0 ,exclusions

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


def create_maxsat_cost_operator_reduced(problem, solutions=[]):

    """
    Creates the ``cost_operator`` for the problem instance.
    This operator is adjusted to consider qubits that were found to be a part of the problem solution.
    
    Parameters
    ----------
    problem : tuple(int, list[list[int]])
        The number of variables and a list of clauses for the MaxSat instance.
    solutions : list
        Variables that were found to be a part of the solution.

    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$. This function performs the application of the cost operator.

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


