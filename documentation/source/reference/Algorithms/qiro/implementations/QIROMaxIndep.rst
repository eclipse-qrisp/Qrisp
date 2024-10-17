.. _maxIndependentSetQIRO:

QIRO MaxIndepSet
================

.. currentmodule:: qrisp.qiro.qiroproblems.qiroMaxIndepSet


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` maximize the size of an independent set, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are mutually non-adjacent.


Replacement routine
-------------------

Based on the **maximum absolute entry** of the correlation matrix M and its sign, one of the following replacements is employed:

* If :math:`\text{M}_{ii} \geq 0` is the maximum absolute value, then the :math:`i`-th vertex is set to be in the independent set (IS). In turn, all vertices that share an edge with this vertex can be removed from the graph, since including them in the solution would violate the problem constraints.

* If :math:`\text{M}_{ii} < 0` is the maximum absolute value, we remove :math:`i`-th vertex from the graph.

* If :math:`\text{M}_{ij} > 0, (i, j) ∈ E` was selected, we remove both vertices from the graph with the argument, that, since both of them would be in the same state in the final solution, including both as part of the solution would violate the constraint, as they share an edge. In turn, they can be removed from the graph. 

* If :math:`\text{M}_{ij} < 0, (i, j) ∈ E` was selected, we remove all vertices that share an edge with both vertices :math:`i` and :math:`j`. Since one of the vertices :math:`i` and :math:`j` will be part of the final solution (but not both), any vertex that is connected to both :math:`i` and :math:`j` is guaranteed to violate the problem constraints, and can be removed from the graph. In this case, it may be possible that no vertex is found to be a canditate for being removed. We will then simply choose the second biggest absolute value of **M** for the replacement routine.


.. autofunction:: create_maxIndep_replacement_routine


QIRO Cost operator 
------------------

.. autofunction:: create_maxIndep_cost_operator_reduced


Example implementation
----------------------
::

    from qrisp import QuantumVariable
    from qrisp.qiro import QIROProblem, create_maxIndep_replacement_routine, create_maxIndep_cost_operator_reduced, qiro_RXMixer, qiro_init_function
    from qrisp.qaoa import create_max_indep_set_cl_cost_function
    import matplotlib.pyplot as plt
    import networkx as nx

    # Define a random graph via the number of nodes and the QuantumVariable arguments
    num_nodes = 13
    G = nx.erdos_renyi_graph(num_nodes, 0.4, seed = 107)
    qarg = QuantumVariable(G.number_of_nodes())

    qiro_instance = QIROProblem(G,
                                replacement_routine=create_maxIndep_replacement_routine,
                                cost_operator=create_maxIndep_cost_operator_reduced,
                                mixer=qiro_RXMixer,
                                cl_cost_function=create_max_indep_set_cl_cost_function,
                                init_function=qiro_init_function
                                )
    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth=3, n_recursions=2)

That’s it! In the following, we print the 5 most likely solutions together with their cost values, and compare to the NetworkX solution.

::
    
    cl_cost = create_max_indep_set_cl_cost_function(G)

    print("5 most likely QIRO solutions")
    max_five_qiro = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
    for res in max_five_qiro: 
        print([index for index, value in enumerate(res) if value == '1'])
        print(cl_cost({res : 1}))

    print("Networkx solution")
    print(nx.approximation.maximum_independent_set(G))

Finally, we visualize the final QIRO graph and the most likely solution.

::

    # The final graph that has been adjusted
    final_graph = qiro_instance.problem
    
    # Draw the final graph and the original graph for comparison
    plt.figure(1)
    nx.draw(final_graph, with_labels=True, node_color='#ADD8E6', edge_color='#D3D3D3')
    plt.title('Final QIRO graph')

    plt.figure(2)
    most_likely = [index for index, value in enumerate(max_five_qiro[0]) if value == '1']
    nx.draw(G, with_labels=True, 
            node_color=['#FFCCCB' if node in most_likely else '#ADD8E6' for node in G.nodes()],
            edge_color='#D3D3D3')
    plt.title('Original graph with most likely QIRO solution')
    plt.show()  