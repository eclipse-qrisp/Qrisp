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

from qrisp import h, rz , cx
import numpy as np
from scipy.optimize import minimize
from sympy import Symbol
import itertools

from collections.abc import Iterable 

# rethink this --> can we call clause[3]% 2 in assignment? because otherwise his blows up the circuit size...


def eThrTwocostOp(clauses):

    """
    |  Implementation for eThrTwoLin Cost-Operator.
    |  Straight forward, apply :math:`R_{ZZZ}` gates between all qubits meantioned in each clause.

    
    Parameters
    ----------
    clauses : List(List)
        clauses to be considered for eTwoThrLin
        
    Returns
    -------
    QuantumCircuit: qrisp.QuantumCircuit
        the Operator applied to the circuit-QuantumVariable

    Examples
    --------

    first three are the indices (a_1,j ; a_2,j ; a3,j), last one is b_L
    
    >>> clauses = [(1,3,4,5), (2,6,9,11), (1,5,8,33), (3,4,5,77), (2,4,6,7), (1,3,4,55), (1,3,6,49), (1,3,4,57), (1,3,6,44), (2,4,7,8)]
    >>> cost_operator = eThrTwocostOp(clauses=clauses)

    """
    if not isinstance(clauses, Iterable):
        raise Exception("Wrong structure of problem - clauses have to be iterable!")
    for clause in clauses:
        if not isinstance(clause, Iterable):
            raise Exception("Wrong structure of problem - each clauses has to be an array!")
        for item in clause:
            if not isinstance(item, int):
                raise Exception("Wrong structure of problem - each literal has to an int!")


    def eThrTwocostnoEmbed(qv , gamma):
        #clauses = clauses
        qc = qv
        for numClause in range(len(clauses)):
            #go through clauses and apply quantum gates 
            clause = clauses[numClause]

            cx(qc[clause[0]], qc[clause[1]])
            cx(qc[clause[1]], qc[clause[2]])
            rz(pow(-1, clause[3] % 2)* gamma, qc[clause[2]])
            cx(qc[clause[1]], qc[clause[2]])
            cx(qc[clause[0]], qc[clause[1]])

        return qc
    
    return eThrTwocostnoEmbed



def eTwoThrLinclCostfct(clauses):
    
    """

    Parameters
    ----------
    clauses : List
            clauses to analyze in the problem instance

    Returns
    -------
    Costfunction : function
        the classical function for the problem instance, which takes a dictionary of measurement results as input
    """

    def setupaClCostfct(res_dic):
        energy = 0
        total_counts = 0
        for state in list(res_dic.keys()):
            obj = 0
            #ljust do a minus 1 op if state is equiv to a given condition clause
            for index in range(len(clauses)):
                clause = clauses[index]
                sum = 0
                for index_aj in range(0,2):
                    sum += int(state[clause[index_aj]])
                if sum == clause[3] % 2:
                    obj -= 1 
            energy += obj * res_dic[state]
            total_counts += res_dic[state]
        #print(energy/total_counts)

        return energy/total_counts
    
    return setupaClCostfct


def init_state(qv):
    # all in 0
    h(qv)
    return qv

