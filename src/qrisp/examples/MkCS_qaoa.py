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

from qrisp.qaoa import create_coloring_operator, XY_mixer, apply_XY_mixer, QAOAProblem, create_coloring_cl_cost_function, RX_mixer
from qrisp import QuantumArray, QuantumVariable
import networkx as nx
from operator import itemgetter
import numpy as np
import random

class QuantumColor(QuantumVariable):
    """
    The QuantumColor is a custom QuantumVariable implemented with tackling the Max-k-Colorable-Subgraph problem
    and other coloring optimization problems in mind. It provides flexibility in choosing encoding methods and 
    leverages efficient data structures like QuantumArrays to enhance computational performance.

    The QuantumColor class takes as input a list of colors and a flag indicating the preferred encoding 
    method - binary or one-hot encoding. The choice of encoding method has implications for how colors are 
    represented in the quantum computation.

    In binary encoding, each color is represented by a unique binary number. For instance, if there are four 
    colors, Red, Green, Blue, and Yellow, they can be represented, for example, as [0,0], [0,1], [1,0], and [1,1] 
    respectively.

    In contrast, one-hot encoding represents each color as an array where only one element is 1 and the rest are 0. 
    Using the same four-color example, red can be represented as [1,0,0,0], green as [0,1,0,0], blue as [0,0,1,0], 
    and yellow as [0,0,0,1].

    Another key feature of the QuantumColor class is its use of QuantumArrays. QuantumArrays are data structures designed for efficient quantum computation. 
    They allow for compact representation and manipulation of quantum states and operators.

    Parameters.
    ----------
    list_of_colors : list
        The list of colors to be used in the quantum coloring problem.
    one_hot_enc : bool, optional
        The flag to indicate whether to use one-hot encoding. If False, binary encoding is used. We use the one-hot encoding by default.

    Attributes
    ----------
    list_of_colors : list
        The list of colors to be used in the quantum coloring problem.
    one_hot_enc : bool
        The indicator which tells the program whether to use one-hot encoding. If False, binary encoding is used.

    Methods
    -------
    decoder(i)
        Decode the color from the given index for both binary and one-hot encoding.
    """

    def __init__(self, list_of_colors, one_hot_enc = True): 
        """
        Initialize the QuantumColor with a list of colors and a flag indicating whether to use one-hot encoding.

        Parameters
        ----------
        list_of_colors : list
            The list of colors to be used in the coloring problem instance.
        one_hot_enc : bool, optional
            The flag to indicate whether to use one-hot encoding. If False, binary encoding is used. Default is True.

        """
        self.list_of_colors = list_of_colors
        self.one_hot_enc = one_hot_enc

        # If one-hot encoding is used, the size of QuantumVariable is the number of colors
        if one_hot_enc:
            QuantumVariable.__init__(self, size = len(list_of_colors)) 

        # If binary encoding is used, the size of QuantumVariable is the maximal value of log2 for the number of colors
        else:
            QuantumVariable.__init__(self, size = int(np.ceil(np.log2(len(list_of_colors)))))

    def decoder(self, i):
        """
        Decode the color from the given index i.

        Parameters
        ----------
        i : int
            The index to be decoded into a color.

        Returns
        -------
        str
            The decoded color if it exists, otherwise "undefined".
        
        """
        if not self.one_hot_enc:
            # Binary encoding: Each color is represented by a binary number.

            # For example, with four colors Red, Green, Blue and Yellow:
            #Red:   [0,0]
            #Green: [0,1]
            #Green: [1,0]
            #Yellow:[1,1]
            return self.list_of_colors[i]

        else:
            #One hot encoding: Each color is represented by an array where only one element is 1 and rest are 0.

            # For example, with four colors Red, Green, Blue and Yellow:     
            #Red:   [1,0,0,0]
            #Green: [0,1,0,0]
            #Yellow:[0,0,1,0]
            #Blue:  [0,0,0,1]

            is_power_of_two = ((i & (i-1) == 0) and i != 0)

            if is_power_of_two:
                return self.list_of_colors[int(np.log2(i))]

            else:
                return "undefined"

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
#G.add_nodes_from([0, 1, 2, 3])
G.add_nodes_from([0, 1, 2, 3])

# Add edges to the graph
G.add_edges_from([[0,3],[0,1],[0,2],[1,3],[1,2],[2,3]])
#G.add_edges_from([[0,1],[1,2],[1,3],[0,2]])
#G.add_edges_from([[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]])

num_nodes = len(G.nodes)

# The following block of code produces a graph with a predefined number of nodes and randomized edge generation
#-----------------------------------------------------------
# Add nodes to the graph
#G.add_nodes_from(range(num_nodes))

# Add edges to the graph
#for i in range(num_nodes):
#    for j in range(i+1, num_nodes):
#        if random.random() < 0.5: # Probability of edge creation
#            G.add_edge(i, j)
#-----------------------------------------------------------
#nx.draw(G)     # Uncomment this to observe the graph we are trying to color using QAOA

# Define the collection of colors
#color_list = ["red", "blue", "yellow"]
color_list = ["red", "blue", "yellow", "green"]

def cl_cost_function(counts):
    """
    The cl_cost_function provides the definition of the classical cost function for
    the Max-k-Colorable Subgraph problem and calculates the relative energy in respect 
    to the amount of counts for each sample.

    Parameters
    ----------
    counts : int
        The amount of counts we are running our simulation with

    Returns
    -------
    energy/total_counts : int
        The classical cost functions returns the ratio between the energy calculated
        using the mkcs_obj objective funcion and the amount of counts used in the 
        experiment.
    """

    # Set energy and total_counts to 0
    energy = 0
    total_counts = 0

    # Calculate minimum result from mkcs_obj function for all keys in counts  
    min_res = min([mkcs_obj(res, G) for res in counts.keys()])

    # Iterate over all items in counts in reverse order    
    for meas, meas_count in list(counts.items())[::-1]:

        # Calculate objective function for current measurement    
        obj_for_meas = mkcs_obj(meas, G)
            
        # if obj_for_meas == min_res:
        #     print(meas, obj_for_meas, "<=========== Optimal result")
        # else:
        #     print(meas, obj_for_meas)    

        energy += obj_for_meas * meas_count
        total_counts += meas_count

    # Print total cost (ratio of energy to total_counts)    
    #print("Total cost: ", energy/total_counts)

    # Return total cost as classical cost function value
    return energy / total_counts

def mkcs_obj(quantumcolor_array, G):
    """
    The mkcs_obj is the objective function for the Max-k-Colorable Subgraph problem 
    instance. As the name suggests, it calculates the value of the objective function
    using which one can compare results and pinpoint the optimal result with the 
    lowest free energy value.

    Our function starts with the value 1 for the color integer and adds a reward
    factor of 4 in case the neighboring nodes are not of the same color. Through 
    trial and error we observed the benefit of multiplying the reward compared to
    a simple addition, since it increases the frequency of our simulation returning
    the optimal (correct) result

    Parameters
    ----------
    quantumcolor_array : QuantumArray
        A QuantumArray consisting of the color values for each node of graph G

    G : NetworkX graph
        The graph to be optimally colored using QAOA.

    Returns
    -------
    color : integer
        The Max-k-Colorable Subgraph objective function returns an integer value
        of the free energy objective function
          
    """

    # Set value of color integer to 1
    color = 1

    # Iterate over all edges in graph G
    for pair in list(G.edges()):

        # If colors of nodes in current pair are not same, multiply color by reward factor 4
        if quantumcolor_array[pair[0]] != quantumcolor_array[pair[1]]:
            color *= 4

    # Return negative color as objective function value. The negative value is used since we want to minimize the objective function       
    return -color

#---------------------------------------------------------
# cl_cost_function = create_coloring_cl_cost_function(G)
# Create a classical cost function

# Depth of the QAOA circuit, usually denoted with p in literature.
depth = 3 

# Creates an unitary coloring operatur
coloring_operator = create_coloring_operator(G)

use_quantum_array = True

# Define quantum argument as a QuantumArray of QuantumColors
qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = True), shape = num_nodes) 

# Define the initial state, which is a random coloring of all nodes
init_state = [random.choice(color_list) for _ in range(len(G))]

def initial_state_mkcs(qarg):
    """
    The initial_state_mkcs function provides the correct initial state of qubits in
    the system on which we run the optimization. In the case of the Max-k-Colorable 
    Subgraph problem, the initial state of the systemis simply any random coloring 
    of nodes of the graph.

    Parameters
    ----------
    qarg : QuantumArray
        A QuantumArray consisting of the color values for each node of graph G.


    Returns
    -------
    qarg : QuantumArray
        The quantum argument (in our case this is a QuantumArray) adapted to include 
        the information of the initial state of the system.
          
    """

    # Set all elements in qarg to initial state
    qarg[:] = init_state

    # Return updated quantum argument
    return qarg

from qrisp.default_backend import def_backend
from qrisp.interface import VirtualQiskitBackend
# Set default backend for QAOA
qrisp_sim = def_backend

# Set backend for QAOA
qaoa_backend = qrisp_sim
import time

# Creates a graph coloring problem instance using the information of the phase separator, mixer, and classical cost function
coloring_instance = QAOAProblem(coloring_operator, apply_XY_mixer, cl_cost_function) 
# coloring_instance = QAOAProblem(coloring_operator, RX_mixer, cl_cost_function) 

# Sets the initial state to the one defined above. If no initial state is defined, it is automatically set to a superposition.
coloring_instance.set_init_function(initial_state_mkcs) 

start_time = time.time()

# Run QAOA with given quantum arguments, depth, measurement keyword arguments and maximum iterations for optimization
res = coloring_instance.run(qarg, depth, mes_kwargs={"backend" : qaoa_backend}, max_iter = 25)

print(qarg.qs)

# Print runtime of QAOA 
print(time.time()-start_time)
#coloring_operator(quantumcolor_array, gamma)


# Get the best solution and print it
best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in res.keys()], key=itemgetter(0))
print(f"Best string: {best_solution} with coloring: {-best_coloring}")

# Get final solution with optimized gamma and beta angle parameter values and print it
best_coloring, res_str = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in list(res.keys())[:5]], key=itemgetter(0))
print("QAOA solution: ", res_str)
best_coloring, best_solution = (mkcs_obj(res_str,G),res_str)

# Draw graph with node colors specified by final solution
nx.draw(G, node_color=res_str, with_labels=True)

import matplotlib.pyplot as plt
# Show plot
plt.show()