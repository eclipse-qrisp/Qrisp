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

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qrisp import *

def W(qv, n):
    """
    This algorithms prepares tensor product of a (paraqmetrized) W-state of the first n>=1 qubits of the QuantumVariable qv and the |0> state for the remaining qubits.
    We call such a state a partial W-state of size n.

    Parameters
    ----------
    qv : QuantumVariable
        A QuantumVariable of N qubits.
    n : int
        The index n.

    """
    x(qv[0])
    for i in range(1,n):
        phi = 2*np.arcsin(1/np.sqrt(n+1-i)) # Phi for uniform superposition
        pswap(phi, qv[i], qv[0])


def conjugator(a,b):
    h(a)
    cx(a, b)

def quasi_swap(a, b):
    with conjugate(conjugator)(a,b):
        h([a,b])

def pswap(phi, a, b):
    with conjugate(conjugator)(a,b):
        ry(-phi/2, [a,b])    
    

def make_distinct(qv1, qv2, n):
    """
    This algorithm prepares a state of two QuantumVariables qv1, qv2 that is a superposition of all tensor products of basis states of length N with exactly one "1" in the first n qubits, such that 
    the QuantemVariables qv1, qv2 are never in the same state. 
    For example: qv1: |10>+|01>, qv2: |10> --->>> |10>|01> + |01>|10>. Here, the states |01>|01> and |10>|10> are forbidden.

    Parameters
    ----------
    qv1 : QuantumVariable
        N-qubit state that is a tensor product of a (parametrized) W-state of qubits q[0]...q[n] and the |0> state of qubits q[n+1]...q[N-1]
    qv2 : QuantumVariable
        N-qubit state that is a tensor product of a (parametrized) W-state of qubits q[0]...q[n-1] and |0> state of qubits q[n]...q[N-1]
    n : int
        The index n.

    """
    for i in range(n):
        with control(qv1[i]):
            quasi_swap(qv2[n], qv2[i])

def prepare_pbs_state(graph, root, N, q_array):
    """
    This algorithm prepares a superposition state of all feasible solutions for a product breakdown structure (PBS) problem.
    We call this state the PBS superposition state. For this, it utilizes the PBS tree structure: A node (part) and all its (direct) predecessors can never be in the same state (site).

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph representing the PBS.
    root : int
        The root of the graph.
    N : int
        The number sites.

    """
    M = graph.number_of_nodes()
    
    assert M == len(q_array)

    W(q_array[root],N)
    
    add_predecessors(graph, root, N, q_array)
    
    return q_array


def add_predecessors(graph, node, N, q_array):
    """
    A recursive algorithm to add predecessors of a node to the PBS superposition state

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph representing the PBS.
    node : int
        The current node.
    N : int
        The number sites.
    q_array : QuantumArray
        The QuantumArray representing the PBS superposition state.

    """
    predecessors = list(graph.predecessors(node))
    
    m = len(predecessors)
    if(N<m+1):
        raise Exception(
                "Insufficient number of sites N"
        )

    # Prepare states for predecessors (partial W-states of decreasing size n)
    for i in range(m):
        W(q_array[predecessors[i]],N-1-i)

    for i in range(m-2,-1,-1):
        for j in range(i+1,m):
            make_distinct(q_array[predecessors[i]], q_array[predecessors[j]], N-2-i)

    # Create entangled satate of predecessors and node that does satisfy constraints
    for j in range(m):
        make_distinct(q_array[node],q_array[predecessors[j]],N-1)

    # Recursivley add predecessors
    for pred in predecessors:
        add_predecessors(graph, pred, N ,q_array)
