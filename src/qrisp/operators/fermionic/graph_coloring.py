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

@nb.njit
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

def find_coloring(G):
    adjacency_matrix = nx.to_numpy_array(G)
    return rlf_coloring(len(G), adjacency_matrix)

