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


from qrisp import QuantumVariable, h, barrier, rz, rx , cx, QuantumArray,x 
import numpy as np
from scipy.optimize import minimize
from sympy import Symbol
import itertools
from _collections_abc import Iterable





def maxSatCostOp(problem):

    """
    |  Implementation for MaxSat Cost-Operator, in accordance to the original QAOA-paper.
    |  For each clause :math:`C_i` apply :math:`C_i(x) = 1 - \prod_j x_j` (replace :math:`x_j` with 
    :math:`(1-x_j)` if :math:`x_j` is negated). The problem operator is acquired with the substitution 
    :math:`x_j = (I - Z_j)` , where :math:`Z_j` is the Pauli- :math:`Z` operator applied to the :math:`j` -th qubit.
    
    Parameters
    ----------
    clauses : List(Lists)
        Clauses to be considered for MaxSat. Should be given as a list of lists and 1-indexed instead 0-indexed, see example

        
    Returns
    -------
    QuantumCircuit: qrisp.QuantumCircuit
        the Operator applied to the circuit-QuantumVariable

    Examples
    --------
    >>> clauses11 = (6, [[1,2,-3], [1,4,-6], [4,5,6], [1,3,-4], [2,4,5], [1,3,5], [-2,-3,6]])
    
    |  Explanation: 
    |  First entry of tuple is the number of variables, second is the clauses
    |  Clause [1, 2, -4] is fulfilled by the QuantumStates "1100", "1110" 

    * if the sign is positive : the index has of the QuantumVariable to be "1", if negative sign the index has to be "0".
    * indices not mentioned can be "0" or "1" (the third integer in this case).
    * start with 1 in your clauses! because ``int 0`` has no sign and this is problematic. 
    * we want to keep the mathematical formulation, so no handing over strings!

    Assign the operators

    >>> cost_operator=maxSatCostOp(clauses11),
    >>> mixer=RX_mixer 

    
    """
    

    clauses = problem[1]

    if not isinstance(clauses, Iterable):
        raise Exception("Wrong structure of problem - clauses have to be iterable!")
    satlen = len(clauses[0])
    for index in range(len(clauses)):
        if not satlen ==len(clauses[index]):
            raise Exception("Wrong structure of problem - inconsistent length in clause " + str(index))
        if not isinstance(clauses[index], Iterable):
            raise Exception("Wrong structure of problem - each clauses has to be an array! Problem in clause " + str(index))
        for item in clauses[index]:
            if not isinstance(item, int):
                raise Exception("Wrong structure of problem - each literal has to an int!")



    def maxSatcostEmbed(qv , gamma):
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




def clausesdecoder(problem): 
    """
    Decodes the clause arrays to represent binary bitstrings, that fulfill the clauses
    --> used to calculate objective function, i.e. the classical cost function

    
    Parameters
    ----------
    clauses : List(List)
        clauses to be considered for maxSat (! 1-indexed, instead of 0-indexed, see example above)
    numVars :  int
        the total amount of Variables considered within the clauses

    Returns
    -------
    decodedClauses : Iterable(tuple)
        clauses as bitstrings, to be compared to the QAOA-bitstring results

    Examples
    --------
    As above:

    >>> clauses11 = [[1,2,-3], [1,4,-6], [4,5,6], [1,3,-4], [2,4,5], [1,3,5], [-2,-3,6]]
    >>> decodedClauses = clausesdecoder( clauses = clauses11, numVars = 6)

    Assign ``cost_function``
    
    >>> cl_cost_function=maxSatclCostfct(decodedClauses)

    

    """

    numVars = problem[0]
    clauses = problem[1]

    # create all bitstring possibilites
    binStrings = list(itertools.product([0,1], repeat=numVars))
    decodedClauses = []
    for indexClauses in range(len(clauses)):
        # create temp placeholder 
        tempIter = binStrings 
        clause = clauses[indexClauses]
        for index in clause:
            val = abs(index)-1 
            temp = 1 if np.sign(index) == 1 else 0 
            # placeholder only keeps to bitstrings that fulfill the clause-conditions
            tempIter = [ thing for thing in tempIter if thing[val] == temp ]
        # turn to str, since QAOA return is dict of str
        tempIter2 = ["".join(map(str, item))  for item in tempIter]
        decodedClauses.append(tuple(tempIter2))

    return decodedClauses



def maxSatclCostfct(problem):

    """

    Parameters
    ----------
    decodedClauses : Iterable(tuple)
        clauses as bitstrings, as returned by ``clausesdecoder()`` helper function


    Returns
    -------
    Costfunction : function
        the classical function for the problem instance, which takes a dictionary of measurement results as input

    """


    decodedClauses = clausesdecoder(problem)

    def setupaClCostfct(res_dic):
        energy = 0
        total_counts = 0
        for state in list(res_dic.keys()):
            obj = 0
            #literally just do a minus 1 op if state is equiv to a given condition
            for index in range(len(decodedClauses)):
                if state in decodedClauses[index]:
                    obj -= 1 
            energy += obj * res_dic[state]
            total_counts += res_dic[state]
        #print(energy/total_counts)

        return energy/total_counts
    
    return setupaClCostfct


def init_state(qv):
    h(qv)
    return qv

