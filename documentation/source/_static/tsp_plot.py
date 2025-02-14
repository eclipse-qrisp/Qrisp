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

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import itertools

def traveling_salesman_bruteforce(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    if num_nodes <= 1:
        return 0, [0]

    # Generate all possible permutations of cities to visit
    all_permutations = itertools.permutations(range(1, num_nodes))

    min_distance = float('inf')
    best_path = []

    # Iterate through all permutations and calculate the total distance
    for permutation in all_permutations:
        current_distance = 0
        current_path = [0] + list(permutation) + [0]

        for i in range(num_nodes):
            current_distance += adjacency_matrix[current_path[i]][current_path[i + 1]]

        if current_distance < min_distance:
            min_distance = current_distance
            best_path = current_path

    return min_distance, best_path



# Define the adjacency matrix for 4 cities (replace with your data)
adjacency_matrix = np.array([[0, 2, 9, 10, 3],
                             [1, 0, 6, 4, 8],
                             [15, 7, 0, 8, 9],
                             [6, 3, 12, 0, 7],
                             [3, 8, 9, 7, 0]])

# Define the order of cities visited (replace with your solution)
# order_of_cities = [0, 1, 3, 2, 4, 0]  # Start and end at the first city

min_distance, order_of_cities = traveling_salesman_bruteforce(adjacency_matrix)

# Extract the number of cities
num_cities = len(order_of_cities) - 1

# Create a graph using NetworkX
G = nx.Graph()

# Add nodes for each city
for i in range(num_cities):
    G.add_node(i, pos=(np.random.rand(), np.random.rand()))  # Replace with actual city coordinates

edge_labels = {}

# Add edges with distances
for i in range(num_cities):
    for j in range(num_cities):
        if i == j:
            continue
        start_city = i
        end_city = j
        distance = adjacency_matrix[start_city, end_city]
        G.add_edge(start_city, end_city, weight=distance)
        
        if i < j:
            edge_labels[start_city, end_city] = distance

# Create a plot
# plt.figure(figsize=(6, 6))

# Draw the nodes with different colors
pos = nx.spring_layout(G, seed = 182)
# pos = nx.spring_layout(G, seed = 189)
# pos = nx.kamada_kawai_layout(G)
labels = {0 : "a", 1 : "b", 2: "c", 3: "d", 4: "e"}
nx.draw_networkx_nodes(G, pos, node_color='#7d7d7d', node_size=500)

# Draw the edges between cities as dotted lines
nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), style='dotted', edge_color='gray')
nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke")
# Draw the solution path as a solid line
solution_edges = [(order_of_cities[i], order_of_cities[i + 1]) for i in range(num_cities)]

# nx.draw_networkx_edges(G, pos, edgelist=solution_edges, style='solid', edge_color='#263a88', width=7, alpha=0.8,)
nx.draw_networkx_edges(G, pos, edgelist=solution_edges, style='solid', edge_color='#20306f', width=7, alpha=0.8,)
# nx.draw_networkx_edges(G, pos, edgelist=solution_edges, style='solid', edge_color='#015999', width=7, alpha=0.8,)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, label_pos=0.3, font_color = "#444444", bbox = {'boxstyle': 'square',  "ec" : (1.0, 1.0, 1.0), "fc" : (1.0, 1.0, 1.0), "pad" : 0.1})




# Set plot title and labels
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

# Show the plot
plt.axis('off')
plt.grid()
plt.tight_layout() 
# plt.savefig("tsp.svg", format = "svg", dpi = 80, bbox_inches = "tight")
# plt.show()