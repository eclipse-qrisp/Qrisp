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

from qrisp import *
import numpy as np
from operator import itemgetter

def QUBO_obj(bitstring, Q):
    x = np.array(list(bitstring), dtype=int)
    cost = x.T @ Q @ x
    return cost

def create_QUBO_cl_cost_function(Q):
    """
    Creates the classical cost function for QUBO with the QUBO matrix `Q`` that we are attempting to solve.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.

    Returns
    -------
    cl_cost_function : function
        Classical cost function, which in the end returns the ratio between 
        the energy calculated using the QUBO_obj objective function and the 
        amount of counts used in the experiment.

    """    
    def cl_cost_function(counts):
        
        def QUBO_obj(bitstring, Q):
            x = np.array(list(bitstring), dtype=int)
            cost = x.T @ Q @ x
            return cost
        
        energy = 0
        for meas, meas_count in counts.items():
            obj_for_meas = QUBO_obj(meas,Q)
            energy += obj_for_meas * meas_count
        return energy
    
    return cl_cost_function


# MAPPING QUBO TO GATES                       1 - Z_i                 1
#                               x_i  <-----> ---------:     x^T Q x = - SUM_{i,j}  Q_{ij} (1-Z_i)(1-Z_j) = SUM_{i,j} Q_ij*(Z_i*Z_j - Z_i - Z_j + 1)/4
#                                                2                    4  


def create_QUBO_cost_operator(Q):
    """
    Creates the QUBO operator as a sequence of unitary gates. In the QAOA overview section this is also called the phase separator $U_P$.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.

    Returns
    -------
    cost_operator function.

    """
    #def QUBO_cost_operator(qv, gamma):
        
    #    for i in range(len(Q)):
    #        rz(-0.5*2*gamma*(0.5*Q[i][i]+0.5*sum(Q[i])), qv[i])
    #        for j in range(i+1, len(qv)):
    #            if Q[i][j] !=0:
    #                rzz(0.25*2*gamma*Q[i][j], qv[i], qv[j])
    #return QUBO_cost_operator
    #new try
    def QUBO_cost_operator(qv, gamma):

        gphase(-gamma/4*(np.sum(Q)+np.trace(Q)),qv[0])

        for i in range(len(Q)):
            rz(-gamma/2*(sum(Q[i])+sum(Q[:,i])), qv[i])
            for j in range(len(Q)):
                if i != j and Q[i][j] != 0:
                    rzz(gamma/2*Q[i][j], qv[i], qv[j])
    return QUBO_cost_operator


def QUBO_problem(Q):
    """
    Creates a QAOA problem instance taking the phase separator, appropriate mixer, and
    appropriate classical cost function into account.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.

    Returns
    -------
    QAOAProblem : function
        QAOA problem instance for QUBO with which the QAOA algorithm is ran for.

    """    
    from qrisp.qaoa import QAOAProblem, RX_mixer
    
    return QAOAProblem(create_QUBO_cost_operator(Q), RX_mixer, create_QUBO_cl_cost_function(Q))


def solve_QUBO(Q, depth, backend = None, n_solutions = 1, print_res = True):
    """
    Solves a Quadratic Unconstrained Binary Optimization (QUBO) problem using the Quantum Approximate Optimization Algorithm (QAOA). 
    The function imports the default backend from the 'qrisp.default_backend' module. 
    It defines a quantum argument as a QuantumArray of len(Q) QuantumVariables with size 1. 
    It then runs the QAOA with the given quantum arguments, ``depth``, measurement keyword arguments, and a maximum of 50 iterations for optimization. 
    The functions then considers the first ``n_solutions`` solutions of the QAOA optimization, and as a classical post processing, 
    calculates the cost for each such solution, sorts the solutions by their cost in ascending order, and prints the solutions with their corresponding costs.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.
    depth : int
        The depth (amount of layers) of the QAOA circuit.
    backend : str
        The backend to be used for the quantum/annealing simulation.
    n_solutions : int
        The number of solutions to consider for classical post processing. The defalut is 1.

    Returns
    -------
    None
        The function prints the runtime of the QAOA algorithm and the ``n_solutions`` best solutions with their respective costs.

    """

    # Define quantum argument as a QuantumArray of len(G) QuantumVariables with size 1 or as a QuantumVariable with size len(G)
    qarg = QuantumArray(qtype = QuantumVariable(1), shape = len(Q))

    QUBO_instance = QUBO_problem(Q)

    if backend is None:
        from qrisp.default_backend import def_backend
        backend = def_backend

    else:
        backend = backend

    # Run QAOA with given quantum arguments, depth, measurement keyword arguments and maximum iterations for optimization
    res = QUBO_instance.run(qarg, depth, mes_kwargs={"backend" : backend}, max_iter = 50) # runs the simulation
    res = dict(list(res.items())[:n_solutions])

    # Calculate the cost for each solution
    costs_and_solutions = [(QUBO_obj(bitstring, Q), bitstring) for bitstring in res.keys()]

    # Sort the solutions by their cost in ascending order
    sorted_costs_and_solutions = sorted(costs_and_solutions, key=itemgetter(0))

    if print_res is True:
        # Get the top solutions and print them
        for i in range(n_solutions):
            print(f"Solution {i+1}: {sorted_costs_and_solutions[i][1]} with cost: {sorted_costs_and_solutions[i][0]}")