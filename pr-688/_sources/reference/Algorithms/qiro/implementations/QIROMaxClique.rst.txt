.. _maxcliqueQIRO:

QIRO MaxClique 
==============

.. currentmodule:: qrisp.qiro.qiroproblems.qiroMaxClique


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` maximize the size of a clique, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are adjacent.


Replacement routine
-------------------

In this instance the replacements can be rather significant. In-depth investigation may show relaxations to be perform better on larger instances.
Based on the **maximum absolute entry** of the correlation matrix M and its sign, one of the following replacements is employed:

* If :math:`\text{M}_{ii} \geq 0` is the maximum absolute value, then the :math:`i`-th vertex is set to be in the clique set. In turn, all vertices that **do not share an edge with this vertex can be removed from the graph**, since including them in the solution would violate the problem constraints.

* If :math:`\text{M}_{ii} < 0` is the maximum absolute value, we remove :math:`i`-th vertex from the graph.

* If :math:`\text{M}_{ij} > 0, (i, j) ∈ E` was selected, we keep both vertices as part of the solution and remove all non-neighboring vertices. The likelihood of this rule failing is low, but remains to be investigated. 

* If :math:`\text{M}_{ij} < 0, (i, j) ∈ E` was selected, we remove all vertices that **do not** share an edge with either vertex :math:`i` or :math:`j`. Since one of the vertices :math:`i` and :math:`j` will be part of the final solution (but not both), any vertex that is **not** connected to either :math:`i` or :math:`j` (or both) is guaranteed to violate the problem constraints, and can be removed from the graph. 


.. autofunction:: create_max_clique_replacement_routine


QIRO Cost operator 
------------------

.. autofunction:: create_max_clique_cost_operator_reduced


Example implementation
----------------------

::   

    from qrisp import QuantumVariable
    from qrisp.qiro import QIROProblem, create_max_clique_replacement_routine, create_max_clique_cost_operator_reduced, qiro_rx_mixer, qiro_init_function
    from qrisp.qaoa import max_clique_problem, create_max_clique_cl_cost_function
    import matplotlib.pyplot as plt
    import networkx as nx

    # Define a random graph via the number of nodes and the QuantumVariable arguments
    num_nodes = 15
    G = nx.erdos_renyi_graph(num_nodes, 0.7, seed=99)
    qarg = QuantumVariable(G.number_of_nodes())

    # QIRO
    qiro_instance = QIROProblem(problem=G,
                                replacement_routine= reate_max_clique_replacement_routine,
                                cost_operator=create_max_clique_cost_operator_reduced,
                                mixer=qiro_rx_mixer,
                                cl_cost_function=create_max_clique_cl_cost_function,
                                init_function=qiro_init_function
                                )
    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth=3, n_recursions=2)

    # The final graph that has been adjusted
    final_graph = qiro_instance.problem

That’s it! In the following, we print the 5 most likely solutions together with their cost values, and compare to the NetworkX solution.

::

    cl_cost = create_max_clique_cl_cost_function(G)
    
    print("5 most likely QIRO solutions")
    max_five_qiro = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
    for res in max_five_qiro:
        print([index for index, value in enumerate(res) if value == '1'])
        print(cl_cost({res : 1}))

    print("Networkx solution")
    print(nx.approximation.max_clique(G))

Finally, we visualize the final QIRO graph and the most likely solution.

::

    # Draw the final graph and the original graph for comparison
    plt.figure(1)
    nx.draw(final_graph, with_labels=True, node_color='#ADD8E6', edge_color='#D3D3D3')
    plt.title('Final QIRO graph')

    plt.figure(2)
    most_likely = [index for index, value in enumerate(max_five_qiro[0]) if value == '1']
    nx.draw(G, with_labels=True,
            node_color=['#FFCCCB' if node in most_likely else '#ADD8E6' for node in G.nodes()],
            edge_color=['#FFCCCB' if edge[0] in most_likely and edge[1] in most_likely else '#D3D3D3' for edge in G.edges()])
    plt.title('Original graph with most likely QIRO solution')
    plt.show()