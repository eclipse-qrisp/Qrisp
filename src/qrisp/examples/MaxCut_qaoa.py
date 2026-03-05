"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
********************************************************************************
"""

from qrisp.qaoa import (
    create_maxcut_cost_operator,
    RX_mixer,
    QAOAProblem,
    create_maxcut_cl_cost_function,
)
from qrisp import QuantumVariable, QuantumArray, h
import networkx as nx
from operator import itemgetter
import numpy as np

# Create an empty graph
G = nx.Graph()

# Add edges to the graph
G.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])

n = 21

G = nx.random_regular_graph(4, n, seed=None)


def cl_cost_function(counts):
    """
    The cl_cost_function provides the definition of the classical cost function for
    the MaxCut problem instance and calculates the relative energy in respect
    to the amount of counts for each sample.

    Parameters
    ----------
    counts : int
        The amount of counts we are running our simulation with

    Returns
    -------
    energy/total_counts : int
        The classical cost functions returns the ratio between the energy calculated
        using the maxcut_obj objective funcion and the amount of counts used in the
        experiment.
    """

    # Set energy and total_counts to 0
    energy = 0
    total_counts = 0

    # Iterate over all items in counts
    for meas, meas_count in counts.items():

        # Calculate objective function for current measurement
        obj_for_meas = maxcut_obj(meas, G)

        # Update energy and total counts
        energy += obj_for_meas * meas_count
        total_counts += meas_count

    # Return total cost as classical cost function value
    return energy / total_counts


def maxcut_obj(x, G):
    """
    The maxcut_obj is the objective function for the MaxCut problem
    instance. As the name suggests, it calculates the value of the objective function
    using which one can compare results and pinpoint the optimal result with the
    lowest free energy value.

    Our function starts with the value 0 for the cut integer and adds a reward
    factor of -1 in case the edge between two neighboring nodes are cut.

    Parameters
    ----------
    x : QuantumArray
        x is an OutcomeArray with size 1 strings which are compered to mark a cut.

    G : NetworkX graph
        The graph to be optimally colored using QAOA.

    Returns
    -------
    cut : integer
        The MaxCut objective function returns an integer value of the free energy
        objective function which correcsponds to the amount of edges that were cut

    """

    # Set value of cut integer to 0
    cut = 0

    # Iterate over all edges in graph G
    for i, j in G.edges():

        # If the nodes are not the same, the edge is cut
        if x[i] != x[j]:  # the edge is cut
            cut -= 1

    # Return cut as objective function value. One subtracts -1 for each cut because we want to miniimze the objective function
    return cut


# Creates a classical cost function
cl_cost_function = create_maxcut_cl_cost_function(G)

# Depth of the QAOA circuit, usially denoted with p in literature
depth = 5

use_quantum_array = True

# Define quantum argument as a QuantumArray of len(G) QuantumVariables with size 1 or as a QuantumVariable with size len(G)
if use_quantum_array:
    qarg = QuantumArray(qtype=QuantumVariable(1), shape=len(G))
else:
    qarg = QuantumVariable(len(G))


def initial_state_maxcut(qarg):
    """
    The initial_state_maxcut function provides the correct initial state of qubits in
    the system on which we run the optimization. In the case of the MaxCut problem,
    the initial state of the system is a superposition we obtain by applying the Hadamard
    gate for every qubit.

    Parameters
    ----------
    qarg : QuantumArray
        A QuantumArray consisting of QuantumVariables of size one corresponding the number of nodes


    Returns
    -------
    qarg : QuantumArray
        The quantum argument (in our case this is a QuantumArray) adapted to include
        the information of the initial state of the system - superposition.

    """
    h(qarg)
    return qarg


# Define the initial state
# instate = initial_state_maxcut(qarg) # the initial state is not necessary in this case since it's automaticaly set to a superposition

# Creates an unitary MaxCut cost operator
cost_operator = create_maxcut_cost_operator(G)

# from qrisp.interface import QiskitBackend,QiskitRuntimeBackend
from qrisp.default_backend import def_backend

# qasm_sim = QiskitBackend()
# qasm_sim = QiskitRuntimeBackend()
qrisp_sim = def_backend

qaoa_backend = qrisp_sim

import time

# Creates a MaxCut problem instance using the information of the phase separator, mixer, and classical cost function
maxcut_instance = QAOAProblem(cost_operator, RX_mixer, cl_cost_function)

print(maxcut_instance.compile_circuit(qarg, depth=5)[0].depth())
# %%
start_time = time.time()

# Run QAOA with given quantum arguments, depth, measurement keyword arguments and maximum iterations for optimization
res = maxcut_instance.run(
    qarg, depth, mes_kwargs={"backend": qaoa_backend}, max_iter=50
)  # runs the simulation

# Print runtime of QAOA
print(time.time() - start_time)
# if qiskit runtime
# qaoa_backend.close_session()

# Get the best solution and print it
best_cut, best_solution = min(
    [(maxcut_obj(x, G), x) for x in res.keys()], key=itemgetter(0)
)
print(f"Best string: {best_solution} with cut: {-best_cut}")

# Get final solution with optimized gamma and beta angle parameter values and print it
res_str = list(res.keys())[0]
print("QAOA solution: ", res_str)
best_cut, best_solution = (maxcut_obj(res_str, G), res_str)

# Draw graph with node colors specified by final solution
colors = ["r" if best_solution[node] == "0" else "b" for node in G]
nx.draw(G, node_color=colors, pos=nx.bipartite_layout(G, [0, 1, 2]))
