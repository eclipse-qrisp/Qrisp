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

from qrisp import QuantumArray
from qrisp.qaoa import QAOAProblem, QuantumColor, XY_mixer, apply_XY_mixer, RX_mixer, create_coloring_operator, create_coloring_cl_cost_function
import random
import networkx as nx
from operator import itemgetter


from qrisp.default_backend import def_backend

qrisp_sim = def_backend
qaoa_backend = qrisp_sim

depth = 3
def mkcs_obj(quantumcolor_array, G):
        # Set value of color integer to 1
    color = 1

        # Iterate over all edges in graph G
    for pair in list(G.edges()):

            # If colors of nodes in current pair are not same, multiply color by reward factor 4
        if quantumcolor_array[pair[0]] != quantumcolor_array[pair[1]]:
            color *= 4

        # Return negative color as objective function value. The negative value is used since we want to minimize the objective function       
    return -color

def test_mkcs_G1e2c():
    ###### Trivial case with 1 edge and 2 colors,
    G1e2c = nx.Graph()
    G1e2c.add_edge(0, 1)
    num_nodes = len(G1e2c.nodes)

    color_list = ["one", "two"]

    def G1e2c_onehot():
        # ONE-HOT 
        qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = True), shape = num_nodes) 

        mkcs_1e2c = QAOAProblem(create_coloring_operator(G1e2c), apply_XY_mixer, create_coloring_cl_cost_function(G1e2c))
        init_state = [random.choice(color_list) for _ in range(len(G1e2c))]

        mkcs_1e2c.set_init_function(lambda x : x.encode(init_state))
        res1e2c = mkcs_1e2c.run(qarg, depth, mes_kwargs={"backend" : qaoa_backend}, max_iter = 25)

        best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G1e2c),quantumcolor_array) for quantumcolor_array in res1e2c.keys()], key=itemgetter(0))

            # Get final solution with optimized gamma and beta angle parameter values and print it
        best_coloring, res_str = min([(mkcs_obj(quantumcolor_array,G1e2c),quantumcolor_array) for quantumcolor_array in list(res1e2c.keys())[:5]], key=itemgetter(0))
        best_coloring, best_solution = (mkcs_obj(res_str,G1e2c),res_str)

        if res_str[0] != res_str[1]:
            return True
    
    for _ in range(5):
        if G1e2c_onehot() == True:
            break
            
    else: 
        assert False


#    onehot_simple = G1e2c_onehot()

    def G1e2c_bin():
        qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = False), shape = num_nodes) 

        mkcs_1e2c = QAOAProblem(create_coloring_operator(G1e2c), RX_mixer, create_coloring_cl_cost_function(G1e2c))
        init_state = [random.choice(color_list) for _ in range(len(G1e2c))]

        mkcs_1e2c.set_init_function(lambda x : x.encode(init_state))
        res1e2c = mkcs_1e2c.run(qarg, depth, mes_kwargs={"backend" : qaoa_backend}, max_iter = 25)

        best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G1e2c),quantumcolor_array) for quantumcolor_array in res1e2c.keys()], key=itemgetter(0))

            # Get final solution with optimized gamma and beta angle parameter values and print it
        best_coloring, res_str = min([(mkcs_obj(quantumcolor_array,G1e2c),quantumcolor_array) for quantumcolor_array in list(res1e2c.keys())[:5]], key=itemgetter(0))
        best_coloring, best_solution = (mkcs_obj(res_str,G1e2c),res_str)

        if res_str[0] != res_str[1]:
            return True
    
#    bin_simple = G1e2c_bin
    for _ in range(5):
        if G1e2c_bin() == True:
            break
    else: 
        assert False

def test_mkcs_5nodes():
    G = nx.Graph()
    G.add_edges_from([[0,1],[0,4],[1,2],[1,3],[1,4],[2,3],[3,4]])

    num_nodes = len(G.nodes)

    color_list = ["one", "two", "three", "four"]

    def G_onehot():
        qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = True), shape = num_nodes) 

        mkcs_onehot = QAOAProblem(create_coloring_operator(G), apply_XY_mixer, create_coloring_cl_cost_function(G))
        init_state = [random.choice(color_list) for _ in range(len(G))]

        mkcs_onehot.set_init_function(lambda x : x.encode(init_state))

        res_onehot = mkcs_onehot.run(qarg, depth, mes_kwargs={"backend" : qaoa_backend}, max_iter = 25)

        best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in res_onehot.keys()], key=itemgetter(0))

            # Get final solution with optimized gamma and beta angle parameter values and print it
        best_coloring_onehot, res_str_onehot = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in list(res_onehot.keys())[:5]], key=itemgetter(0))
        best_coloring_onehot, best_solution_onehot = (mkcs_obj(res_str_onehot,G),res_str_onehot)
        
        if all(res_str_onehot[pair[0]] != res_str_onehot[pair[1]] for pair in list(G.edges())):
            return True


    for _ in range(5):
        if G_onehot() == True:
            break
    else: 
        assert False
#    onehot = G_onehot

    def G_bin():
        qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = False), shape = num_nodes) 

        mkcs_bin = QAOAProblem(create_coloring_operator(G), RX_mixer, create_coloring_cl_cost_function(G))
        init_state = [random.choice(color_list) for _ in range(len(G))]

        mkcs_bin.set_init_function(lambda x : x.encode(init_state))

        res_bin = mkcs_bin.run(qarg, depth, mes_kwargs={"backend" : qaoa_backend}, max_iter = 25)

        best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in res_bin.keys()], key=itemgetter(0))

        best_coloring_bin, res_str_bin = min([(mkcs_obj(quantumcolor_array,G),quantumcolor_array) for quantumcolor_array in list(res_bin.keys())[:5]], key=itemgetter(0))
        best_coloring_bin, best_solution_bin = (mkcs_obj(res_str_bin,G),res_str_bin)

        if all(res_str_bin[pair[0]] != res_str_bin[pair[1]] for pair in list(G.edges())):
            return True
    
    for _ in range(20):
        print(G_bin())
        if not G_bin() == True:
            print("Ayayay")
        


test_mkcs_5nodes()
# ONE HOT




#  BINARY

