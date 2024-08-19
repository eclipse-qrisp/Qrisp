.. _maxcliqueQIRO:

QIRO MaxClique 
==============

.. currentmodule:: qrisp.qiro.qiroproblems.qiroMaxCliqueInfrastr


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` maximize the size of a clique, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are adjacent.


Replacement routine
-------------------

In this instance the replacements can be rather drastic. In-depth investigation may show relaxations to be better performant on larger instances.
Based on the **maximum absolute entry** of the correlation matrix M and its sign, one of the following replacements is employed:

* If :math:`\text{M}_{ii} \geq 0` is the maximum absolute value, then the :math:`i`-th vertex is set to be in the clique set. In turn, we can remove all vertices that **do not share an edge with this vertex can be removed from the graph**, since including them in the solution would violate the problem constraints.

* If :math:`\text{M}_{ii} < 0` is the maximum absolute value we remove :math:`i`-th vertex from the graph

* If :math:`\text{M}_{ij} > 0,  (i, j) ∈ E` was selected, we keep both nodes as part of the solution and remove all non-bordering nodes. The likelyhood of this behaviour failing is low, but remains to be investigated. 

* If :math:`\text{M}_{ij} < 0,  (i, j) ∈ E` was selected, we remove all nodes that **do not** share an edge with either vertice :math:`i` or :math:`j`. Since one of the vertices :math:`i` and :math:`j` will be part of the final solution (but not both), any vertex that is **not** connected to either :math:`i` or :math:`j` (or both) is guaranteed to violate the problem constraints, and can be removed from the graph. 



.. autofunction:: create_maxClique_replacement_routine


QIRO Cost operator 
------------------

.. autofunction:: create_maxClique_cost_operator_reduced


Full Example implementation:
----------------------------
::

    # imports 
    from qrisp import QuantumVariable
    import matplotlib.pyplot as plt
    import networkx as nx
    from qrisp.qiro import QIROProblem, qiro_init_function, qiro_RXMixer, create_maxClique_cost_operator_reduced, create_maxClique_replacement_routine
    from qrisp.qaoa import QAOAProblem,  RX_mixer, maxCliqueCostfct, maxCliqueCostOp



    # First we define a graph via the number of nodes and the QuantumVariable arguments
    num_nodes = 15
    G = nx.erdos_renyi_graph(num_nodes,0.7, seed =  99)
    Gtwo = nx.erdos_renyi_graph(num_nodes,0.7, seed =  99)
    qarg = QuantumVariable(G.number_of_nodes())
    qarg2 = QuantumVariable(Gtwo.number_of_nodes())

    #assign cost_function and maxclique_instance, normal QAOA
    testCostFun = maxCliqueCostfct(Gtwo)
    maxclique_instance = QAOAProblem(maxCliqueCostOp(G), RX_mixer, maxCliqueCostfct(G))

    # assign the correct new update functions for qiro from above imports
    qiro_instance = QIROProblem(problem = Gtwo,  
                                replacement_routine = create_maxClique_replacement_routine, 
                                cost_operator = create_maxClique_cost_operator_reduced,
                                mixer = qiro_RXMixer,
                                cl_cost_function = maxCliqueCostfct,
                                init_function = qiro_init_function
                                )


    # We run the qiro instance and get the results!
    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, 
                                    #mes_kwargs = mes_kwargs
                                    )
    # and also the final graph, that has been adjusted
    final_Graph = qiro_instance.problem

    # get the normal QAOA results for a comparison
    #res_qaoa = maxclique_instance.run( qarg = qarg2, depth = 3)

    # or compare it with the networkx result of the max_clique algorithm...
    print("Networkx solution")
    print(nx.approximation.max_clique(Gtwo))

    # and finally, we draw the final graph and the original graphs to compare them!
    plt.figure(1)
    nx.draw(final_Graph, with_labels = True)

    plt.figure(2)
    nx.draw(G, with_labels = True)
    plt.show()  
