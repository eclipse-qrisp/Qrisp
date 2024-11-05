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


import numba as nb
import numpy as np
import networkx as nx

def multi_hamiltonian_measurement(
        hamiltonians, 
        qarg,
        precision=0.01,
        backend=None,
        shots=1000000,
        compile=True,
        compilation_kwargs={},
        subs_dic={},
        precompiled_qc=None,
        _measurements=None
    ):
    r"""
    This method returns the expected value of a list of Hamiltonians for the state of a quantum argument.

    Parameters
    ----------
    hamiltonians : list[Hamiltonian]
        The Hamiltonians for wich the expeced value is to be evaluated.
    qarg : QuantumVariable, QuantumArray or list[QuantumVariable]
        The quantum argument to evaluate the Hamiltonians on.
    precision: float, optional
        The precision with which the expectation of the Hamiltonians is to be evaluated.
        The default is 0.01.
    backend : BackendClient, optional
        The backend on which to evaluate the quantum circuit. The default can be
        specified in the file default_backend.py.
    shots : integer, optional
        The maximum amount of shots to evaluate the expectation per Hamiltonian. 
        The default is 100000.
    compile : bool, optional
        Boolean indicating if the .compile method of the underlying QuantumSession
        should be called before. The default is True.
    compilation_kwargs  : dict, optional
        Keyword arguments for the compile method. For more details check
        :meth:`QuantumSession.compile <qrisp.QuantumSession.compile>`. The default
        is ``{}``.
    subs_dic : dict, optional
        A dictionary of Sympy symbols and floats to specify parameters in the case
        of a circuit with unspecified, :ref:`abstract parameters<QuantumCircuit>`.
        The default is {}.
    precompiled_qc : QuantumCircuit, optional
            A precompiled quantum circuit.

    Returns
    -------
    list[float]
        The expected value of the Hamiltonians.

    """

    expectations = []
    n = len(hamiltonians)
    for i in range(n):
        expectations.append(hamiltonians[i].get_measurement(qarg,
                                precision=precision,
                                backend=backend,
                                shots=shots,
                                compile=compile,
                                compilation_kwargs=compilation_kwargs,
                                subs_dic=subs_dic,
                                precompiled_qc=precompiled_qc,
                                _measurement=None if _measurements==None else _measurements[i]
                                ))

    return expectations

    

@nb.njit(cache = True)
def rlf_coloring(num_vertices, adjacency_matrix):
    colors = np.full(num_vertices, -1)
    uncolored_vertices = np.arange(num_vertices)
    
    current_color = 0
    while uncolored_vertices.size > 0:
        # Find the vertex with the maximum degree
        degrees = np.sum(adjacency_matrix[uncolored_vertices], axis=1)
        max_degree_index = np.argmax(degrees)
        independent_set = [uncolored_vertices[max_degree_index]]
        
        # Build the maximal independent set
        for vertex in uncolored_vertices:
            if np.all(adjacency_matrix[vertex, np.array(independent_set)] == 0):
                independent_set.append(vertex)
        
        # Assign the current color to all vertices in the independent set
        for vertex in independent_set:
            colors[vertex] = current_color
        
        # Remove colored vertices from the list of uncolored vertices
        uncolored_vertices = np.array([v for v in uncolored_vertices if v not in independent_set])
        
        current_color += 1
    
    return colors

@nb.njit(cache = True)
def dsatur_coloring(num_vertices, adjacency_matrix):
    colors = np.full(num_vertices, -1)
    saturation_degrees = np.zeros(num_vertices, dtype=np.int64)
    uncolored_vertices = np.arange(num_vertices)
    
    # Find the vertex with the maximum degree for the first coloring
    degrees = np.sum(adjacency_matrix, axis=1)
    max_degree_vertex = np.argmax(degrees)
    
    # Color the first vertex
    colors[max_degree_vertex] = 0
    uncolored_vertices = uncolored_vertices[uncolored_vertices != max_degree_vertex]
    
    # Update saturation degrees of neighbors
    for neighbor in range(num_vertices):
        if adjacency_matrix[max_degree_vertex, neighbor] == 1:
            saturation_degrees[neighbor] += 1
    
    while uncolored_vertices.size > 0:
        # Find the vertex with the highest saturation degree
        max_saturation = -1
        max_saturation_vertex = -1
        for vertex in uncolored_vertices:
            if saturation_degrees[vertex] > max_saturation:
                max_saturation = saturation_degrees[vertex]
                max_saturation_vertex = vertex
            elif saturation_degrees[vertex] == max_saturation:
                # If tied, choose the vertex with higher degree
                if degrees[vertex] > degrees[max_saturation_vertex]:
                    max_saturation_vertex = vertex
        
        # Find the lowest available color for this vertex
        used_colors = set()
        for neighbor in range(num_vertices):
            if adjacency_matrix[max_saturation_vertex, neighbor] == 1 and colors[neighbor] != -1:
                used_colors.add(colors[neighbor])
        
        available_color = 0
        while available_color in used_colors:
            available_color += 1
        
        # Assign the color
        colors[max_saturation_vertex] = available_color
        
        # Update saturation degrees of uncolored neighbors
        for neighbor in uncolored_vertices:
            if adjacency_matrix[max_saturation_vertex, neighbor] == 1:
                neighbor_colors = set()
                for v in range(num_vertices):
                    if adjacency_matrix[neighbor, v] == 1 and colors[v] != -1:
                        neighbor_colors.add(colors[v])
                saturation_degrees[neighbor] = len(neighbor_colors)
        
        # Remove the colored vertex from uncolored_vertices
        uncolored_vertices = uncolored_vertices[uncolored_vertices != max_saturation_vertex]
    
    return colors

def find_coloring(G):
    adjacency_matrix = nx.to_numpy_array(G)
    
    coloring_1 = rlf_coloring(len(G), adjacency_matrix)
    coloring_2 = dsatur_coloring(len(G), adjacency_matrix)
    
    if np.max(coloring_1) < np.max(coloring_2):
        return coloring_1
    else:
        return coloring_2


def find_qw_commuting_groups(H):
    G = nx.Graph()
    
    for term_a in H.terms_dict.keys():
        G.add_node(term_a)
        for term_b in H.terms_dict.keys():
            if term_a is term_b:
                continue
            if not term_a.commute_qw(term_b):
                G.add_edge(term_a, term_b)
    
    coloring = find_coloring(G)
    
    groups = []
    for i in range(np.max(coloring)+1): groups.append([])
    
    node_list = list(G.nodes())
    for i in range(len(G)):
        groups[coloring[i]].append(node_list[i])
    
    return groups


