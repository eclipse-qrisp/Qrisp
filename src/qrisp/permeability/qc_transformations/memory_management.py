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

import networkx as nx
import numpy as np
from numba import njit, prange

def optimize_allocations(qc):
    from qrisp.permeability import PermeabilityGraph, TerminatorNode

    # G = dag_from_qc(qc, remove_init_nodes=True)
    G = PermeabilityGraph(qc, remove_artificials = True)
    qc_new = qc.clearcopy()

    dealloc_identifier = lambda x: x.op.name == "qb_dealloc"
    alloc_identifer = lambda x: x.op.name == "qb_alloc"

    # mcx_identifier = lambda x : isinstance(x.op, PTControlledOperation) and x.op.base_operation.name == "x"
    # nmcx_identifier = lambda x : not mcx_identifier(x)
    # sub_sort = lambda G : topological_sort(G, prefer = mcx_identifier, delay = nmcx_identifier)
    # for n in topological_sort(G, prefer = dealloc_identifier, delay = alloc_identifer, sub_sort = sub_sort):

    # The sub_sort function takes the indices of the gates in the source qc as
    # a sorting index. This induces a valid topological ordering because of [proof]
    def sub_sort(dag):
        nodes = list(dag.nodes())
        def sort_key(x):
            if isinstance(x, TerminatorNode):
                return 0
            else:
                return x.qc_index
        nodes.sort(key = sort_key)
        return nodes

    for n in topological_sort(G, prefer=dealloc_identifier, delay=alloc_identifer, sub_sort = sub_sort):
        if n.instr:
            qc_new.append(n.instr)

    #The above algorithm does not move allocation gates to their latest possible
    #position (only compared to other deallocation gates)    
    #We therefore manually move the allocation gates to the position right
    #before the first actual instruction on that qubit.
    new_data = []
    delayed_qubit_alloc_dic = {}
    
    for i in range(len(qc_new.data)):
        instr = qc_new.data[i]
        
        if instr.op.name == "qb_alloc":
            delayed_qubit_alloc_dic[instr.qubits[0]] = instr
        else:
            # We sort the qubits in order to prevent non-deterministic compilation behavior
            alloc_qubits = list(set(delayed_qubit_alloc_dic.keys()).intersection(instr.qubits))
            alloc_qubits.sort(key = lambda x : hash(x))
            for qb in alloc_qubits:
                new_data.append(delayed_qubit_alloc_dic[qb])
                del delayed_qubit_alloc_dic[qb]
            
            new_data.append(instr)
    
    for instr in delayed_qubit_alloc_dic.values():
        new_data.append(instr)
    
    qc_new.data = new_data

    return qc_new


# This function performs a topological sort of the graph G where we try to execute any
# deallocation gates as early as possible while still adhering to the topological order.
# We try to perform a depth-first search as described here:
# https://en.wikipedia.org/wiki/Topological_sorting
# According to the Wikipedia page, we are allowed to pick any node as a "starting point"
# of the DF-search, which allows us to modify the algorithm such that it optimizes the
# (de)allocation order. The general idea is to pick the deallocation nodes as
# starting points, where we order them, such that those deallocation nodes that
# "require" the least allocation nodes are executed first. "Require" here means that
# there is a causal relationship between the allocation and deallocation nodes,
# i.e. there is a path from the allocation node to the deallocation node.


# We can thus determine the amount of allocation nodes required for a deallocation node
# n by counting, the amount of allocation nodes in the "ancestors" subgraph of n.
def topological_sort(G, prefer=None, delay=None, sub_sort=nx.topological_sort):
    """
    Function to perform a topological sort on an Unqomp DAG which allows preferring/
    delaying specific types of nodes

    Parameters
    ----------
    G : nx.DiGraph
        The Unqomp DAG.
    prefer : function, optional
        Function which returns True, when presented with an Instruction, that should be
        preferred. The default is the function that returns False on all Operations
    delay : function, optional
        Function which returns True, when presented with an Instruction, that should be
        delayed. The default is the function that returns False on all Operations
    sub_sort : function, optional
        A function which performs a topological sort, which can sorting preferences of
        secondary importance. The default is nx.topological_sort.

    Returns
    -------
    lin : list[UnqompNode]
        The linearized list of UnqompNodes. The init nodes are not included.

    """

    if prefer is None:
        prefer = lambda x: False

    if delay is None:
        delay = lambda x: False

    G = G.copy()
    # Collect the prefered nodes
    prefered_nodes = []

    for n in G.nodes():
        n.processed = False
        if n.instr is None:
            continue
        
        if prefer(n.instr):
            prefered_nodes.append(n)

        

    # For large scales, finding the ancestors is a bottleneck. We therefore use a
    # jitted version
    if len(G) * len(prefered_nodes) > 10000:
        anc_lists = ancestors(G, prefered_nodes)
    else:
        anc_lists = []
        for i in range(len(prefered_nodes)):
            anc_lists.append(list(nx.ancestors(G, prefered_nodes[i])))

    node_ancs = {
        prefered_nodes[i]: anc_lists[i] for i in range(len(prefered_nodes))
    }
    
    # We sort the nodes in order to prevent non-deterministic compilation behavior
    prefered_nodes.sort(key=lambda x: len(node_ancs[x]) + 1/hash(x.instr))
    
    # Determine the required delay nodes for each prefered nodes
    required_delay_nodes = {n: [] for n in prefered_nodes}

    for n in prefered_nodes:
        for k in node_ancs[n]:
            if k.instr:
                if delay(k.instr):
                    required_delay_nodes[n].append(k)

    required_delay_nodes = {n: set(required_delay_nodes[n]) for n in prefered_nodes}

    
    # Generate linearization
    lin = []

    while prefered_nodes:
        # Sort nodes accordingly
        prefered_nodes.sort(key=lambda x: len(required_delay_nodes[x]))

        node = prefered_nodes.pop(0)
        ancs = []

        for n in node_ancs[node] + [node]:
            if n.processed:
                continue
            else:
                n.processed = True
                ancs.append(n)

        sub_graph = G.subgraph(ancs)

        lin += list(sub_sort(sub_graph))

        continue

        for n in prefered_nodes:
            required_delay_nodes[n] = (
                required_delay_nodes[n] - required_delay_nodes[node]
            )

    # Linearize the remainder
    remainder = []
    for n in G.nodes():
        if n.processed:
            continue
        else:
            n.processed = True
            remainder.append(n)

    # lin += list(sub_sort(G))
    lin += list(sub_sort(G.subgraph(remainder)))

    return lin


@njit(cache=True)
def ancestors_jitted(start_index, indptr, indices, node_amount):
    to_do_array = np.zeros(node_amount, dtype=np.byte)
    to_do_array[start_index] = 1
    done_array = np.zeros(node_amount, dtype=np.byte)

    stack = 1
    while stack:
        node = np.argmax(to_do_array)
        to_do_array[node] = 0

        for i in range(indptr[node], indptr[node + 1]):
            new_node = indices[i]
            if done_array[new_node] == 0:
                to_do_array[new_node] = 1
                stack += 1

        done_array[node] = 1
        stack -= 1

    return np.nonzero(done_array)[0]


@njit(parallel=True, cache=True)
def ancestors_jitted_wrapper(start_indices, indptr, indices, node_amount):
    res = [np.zeros(1, dtype=np.int64)] * len(start_indices)
    for i in prange(len(start_indices)):
        start_index = start_indices[i]
        res[i] = ancestors_jitted(start_index, indptr, indices, node_amount)

    return res


def ancestors(dag, start_nodes):
    node_list = list(dag.nodes())

    sprs_mat = nx.to_scipy_sparse_array(dag, format="csc")

    start_indices = []
    for i in range(len(dag)):
        if node_list[i] in start_nodes:
            start_indices.append(i)

    res_list_indices = ancestors_jitted_wrapper(
        np.array(start_indices).astype(np.int32),
        sprs_mat.indptr,
        sprs_mat.indices.astype(np.int32),
        len(dag),
    )

    node_list = [
        [node_list[j] for j in anc_indices] for anc_indices in res_list_indices
    ]

    return node_list
