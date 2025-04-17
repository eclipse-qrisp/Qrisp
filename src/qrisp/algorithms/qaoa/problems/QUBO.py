"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
    Creates the classical cost function for a QUBO instance with matrix ``Q`` that we are attempting to solve.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.

    Returns
    -------
    cl_cost_function : function
        The classical cost function for the problem instance, which takes a dictionary of measurement results as input.

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
    Creates the cost operator for a QUBO instance with matrix ``Q``.
    In the QAOA overview section this is also called the phase separator $U_P$.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.

    Returns
    -------
    cost_operator : function
        A function receiving a :ref:`QuantumVariable` and a real parameter $\gamma$.
        This function performs the application of the cost operator.

    """
    
    def QUBO_cost_operator(qv, gamma):

        # Rescaling for enhancing the performance of the QAOA
        gamma = gamma/np.sqrt(np.linalg.norm(Q))

        gphase(-gamma/4*(np.sum(Q)+np.trace(Q)),qv[0])

        for i in range(len(Q)):
            rz(-gamma/2*(sum(Q[i])+sum(Q[:,i])), qv[i])
            for j in range(len(Q)):
                if i != j and Q[i][j] != 0:
                    rzz(gamma/2*Q[i][j], qv[i], qv[j])

    return QUBO_cost_operator


def QUBO_problem(Q):
    """
    Creates a QAOA problem instance with appropriate phase separator, mixer, and
    classical cost function.

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.

    Returns
    -------
    :ref:`QAOAProblem`
        A QAOA problem instance for a given QUBO matrix ``Q``.

    """    
    from qrisp.qaoa import QAOAProblem, RX_mixer
    
    return QAOAProblem(create_QUBO_cost_operator(Q), RX_mixer, create_QUBO_cl_cost_function(Q))


def solve_QUBO(Q, depth, shots = 5000, max_iter = 50, backend = None):
    """
    Solves a Quadratic Unconstrained Binary Optimization (QUBO) problem using the Quantum Approximate Optimization Algorithm (QAOA). 

    This function creates the QAOA problem for a given QUBO. 
    It defines a quantum argument as a :ref:`QuantumArray` of ``len(Q)`` :ref:`QuantumVariables <QuantumVariable>` with size 1. 
    It then runs the QAOA with the given quantum argument, ``depth`` (number of layers), maximum ``iterations`` of the classical optimizer, and ``shots`` and ``backend`` as measurement keyword arguments. 
    The method performs classical post-processing on the solutions of the QAOA optimization: 
    it calculates the cost for each such solution, sorts the solutions by their costs in ascending order, and returns the sorted list of solutions.

    .. warning::
     
        For small QUBO instance the number of ``shots`` typically exceeds the number of possible solutions.
        In this case, even QAOA with ``depth=0``, i.e., sampling from a uniform superposition, may yield the optimal solution as the classical post-processing amounts to brute force search!
        Performance of :meth:`solve_QUBO <qrisp.qaoa.problems.QUBO.solve_QUBO>` for small instance may not be indicative of performance for large instances. 

    Parameters
    ----------
    Q : np.array
        QUBO matrix to solve.
    depth : int
        The depth (amount of layers) of the QAOA circuit.
    shots : int
        The number of shots. The default is 5000.
    max_iter : int, optional
        The maximal amount of iterations of the ``COBYLA`` optimizer in the QAOA algorithm.
        The default is 50.
    backend : :ref:`BackendClient`, optional
        The backend to be used for the quantum simulation. 
        By default, the Qrisp simulator is used.

    Returns
    -------
    list[tuple]
        A list of tuples representing the solutions: The first element is the cost, the second element is the bitstring, and the thrid element is the probability. 
        Solutions are sorted by their costs in ascending order.

    Examples
    --------

    ::

        from qrisp.qaoa import solve_QUBO
        import numpy as np

        Q = np.array(
            [
                [-17,  10,  10,  10,   0,  20],
                [ 10, -18,  10,  10,  10,  20],
                [ 10,  10, -29,  10,  20,  20],
                [ 10,  10,  10, -19,  10,  10],
                [ 0,   10,  20,  10, -17,  10],
                [ 20,  20,  20,  10,  10, -28],
            ]
        )

        solve_QUBO(Q, depth = 1, shots = 5000)[:5]

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
    res = QUBO_instance.run(qarg, depth, mes_kwargs={"backend" : backend, "shots" : shots}, max_iter = max_iter, init_type='tqa') # runs the simulation

    # Calculate the cost for each solution
    costs_and_solutions = [(QUBO_obj(bitstring, Q), bitstring, res[bitstring]) for bitstring in res.keys()]

    # Sort the solutions by their cost in ascending order
    sorted_costs_and_solutions = sorted(costs_and_solutions, key=itemgetter(0))
    
    return sorted_costs_and_solutions

