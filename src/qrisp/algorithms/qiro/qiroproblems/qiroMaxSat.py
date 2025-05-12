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
from qrisp.alg_primitives import app_sb_phase_polynomial

import numpy as np
import math
import copy
from qrisp.algorithms.qiro.qiroproblems.qiro_utils import * 

from itertools import combinations


def create_maxsat_replacement_routine(res, problem_updated):
    """
    Creates a replacement routine for the problem structure, i.e., defines the replacement rules. 
    See the `original paper <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327>`_ for a description of the update rules.

    Parameters
    ----------
    res : dict
        Result dictionary of QAOA optimization procedure.
    problem_updated : List
        Updates that happened during the QIRO routine. Consits of the updated problem, a list of Qubits which were found to be positively correlated, i.e. part of the problem solution, 
        and a list Qubits which were found to be negatively correlated, i.e. they contradict solution qubits in accordance with the update rules.  

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

    problem = problem_updated[0]
    solutions = problem_updated[1]
    exclusions =  problem_updated[2]
    
    numVars = problem[0]
    clauses = copy.deepcopy(problem[1])

    # FOR SINGLE QUBIT CORRELATIONS
    # make -1 here for consistency in the find max
    orig_nodes = [i - 1 for i in list(range(1,numVars + 1)) if not i in exclusions]
    combinations_list = list(combinations(orig_nodes, 2))

    max_item, sign = find_max(orig_nodes, combinations_list, res, solutions)
    if max_item == None:
        return problem, solutions, 0 ,exclusions

    # we just directly remove clauses from the problem, or literals from the clause. 
    if isinstance(max_item, int):
        
        max_item += 1

        for sgl_clause in clauses:
            if sign < 0:
                #clause evaluates to TRUE, can empty the clause 
                if max_item in sgl_clause:
                    sgl_clause.clear()
                #clause evaluates to FALSE, can remove the literal
                # if its last literal remaining in the clause, we just remove the clause
                elif -1*max_item in sgl_clause: 
                    if len(sgl_clause) == 1:
                        clauses.remove(sgl_clause)
                    else:
                        val = -1* max_item
                        sgl_clause.remove(val)
                solutions.append(max_item)
                exclusions.append(max_item)

            #same as above
            elif sign > 0:
                if -1* max_item in sgl_clause:
                    sgl_clause.clear()
                elif max_item in sgl_clause: 
                    if len(sgl_clause) == 1:
                        clauses.remove(sgl_clause)
                    else:
                        val =  max_item
                        sgl_clause.remove(val)
                exclusions.append(max_item)


    else:
        max_item = [max_item[0] +1 , max_item[1] + 1 ]

        if sign > 0:
            for sgl_clause in clauses:
                # replace with pos. correlated number if its in an item
                if max_item[1] in sgl_clause:
                    temp = sgl_clause.index(max_item[1])
                    sgl_clause[temp] = max_item[0]
                if -1* max_item[1] in sgl_clause:
                    temp = sgl_clause.index(-1* max_item[1])
                    sgl_clause[temp] = -1* max_item[0]
                exclusions.append(max_item[1])
                    
        elif sign < 0:
            for sgl_clause in clauses:
                # replace with neg. correlated number if its in an item
                if max_item[1] in sgl_clause:
                    temp = sgl_clause.index(max_item[1])
                    sgl_clause[temp] = -1 * max_item[0]
                if -1* max_item[1] in sgl_clause:
                    temp = sgl_clause.index(-1  * max_item[1])
                    sgl_clause[temp] = max_item[0]
                exclusions.append(max_item[1])

    # create sign list somewhere?
    return [numVars, clauses], solutions, sign, exclusions


def create_maxsat_cost_operator_reduced(problem_updated):

    """
    Creates the ``cost_operator`` for the problem instance.
    This operator is adjusted to consider qubits that were found to be a part of the problem solution.
    
    Parameters
    ----------
    problem_updated : List
        Updates that happened during the QIRO routine. Consits of the updated problem, a list of Qubits which were found to be positively correlated, i.e. part of the problem solution, 
        and a list Qubits which were found to be negatively correlated, i.e. they contradict solution qubits in accordance with the update rules.  

    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$. This function performs the application of the cost operator.

    """
    
    from qrisp.algorithms.qaoa import create_maxsat_cost_polynomials

    problem = problem_updated[0]
    
    cost_polynomials, symbols = create_maxsat_cost_polynomials(problem)

    def cost_operator(qv, gamma):
        for P in cost_polynomials:
            if not isinstance(P, int):
                app_sb_phase_polynomial([qv], -P, symbols, t=gamma)

    return cost_operator

    


