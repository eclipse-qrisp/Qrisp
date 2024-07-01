.. _maxIndependentSetQIRO:

QIRO MaxIndependentSet
======================

.. currentmodule:: qiro.qiroproblems.qiroMaxIndepSetInfrastr



Problem description
-------------------

Given a Graph  :math:`G = (V,E)` maximize the size of a clique, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are mutually non-adjacent.




Replacement routine
-------------------

Based on the **maximum absolute entry** of the correlation matrix M and its sign, one of the following replacements is employed:

* If :math:`\text{M}_{ii} \geq 0` is the maximum absolute value, then the :math:`i`-th vertex is set to be in the independent set (IS). In turn, we can remove all vertices that share an edge with this vertex can be removed from the graph, since including them in the solution would violate the problem constraints

* If :math:`\text{M}_{ii} < 0` is the maximum absolute value we remove :math:`i`-th vertex from the graph

* If :math:`\text{M}_{ij} > 0,  (i, j) ∈ E` was selected, we remove both nodes from the graph with the argument, that, since both of them would be in the same state in the final solution, including both as part of the solution would violate the constraint, as they share an edge. In turn, they can be removed from the graph. 

* If :math:`\text{M}_{ij} < 0,  (i, j) ∈ E` was selected, we remove all nodes that share an edge with both vertices :math:`i` and :math:`j`. Since one of the vertices :math:`i` and :math:`j` will be part of the final solution (but not both), any vertex that is connected to both :math:`i` and :math:`j` is guaranteed to violate the problem constraints, and can be removed from the graph. In this case it may be possible, that no vertex is found to be as a canditate for removing. We will then simple chose second biggest absolute value of **M** for the replacement routine.


.. autofunction:: create_maxIndep_replacement_routine


QIRO Cost operator 
------------------

.. autofunction:: create_maxIndep_cost_operator_reduced


Full Example implementation:
----------------------------
::

    # imports 
    from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetclCostfct, maxIndepSetCostOp
    from qrisp.qiro import QIROProblem, qiro_init_function, qiro_RXMixer, create_maxIndep_replacement_routine, create_maxIndep_cost_operator_reduced
    from qrisp import QuantumVariable
    import networkx as nx



    # First we define a graph via the number of nodes and the QuantumVariable arguments
    num_nodes = 13
    G = nx.erdos_renyi_graph(num_nodes, 0.4, seed =  107)
    qarg = QuantumVariable(G.number_of_nodes())


    # assign the correct new update functions for qiro from above imports
    qiro_instance = QIROProblem(G, 
                                replacement_routine=create_maxIndep_replacement_routine, 
                                cost_operator= create_maxIndep_cost_operator_reduced,
                                mixer= qiro_RXMixer,
                                cl_cost_function= maxIndepSetclCostfct,
                                init_function= qiro_init_function
                                )

    # We run the qiro instance and get the results!
    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2, mes_kwargs = mes_kwargs)
    # and also the final graph, that has been adjusted
    final_Graph = qiro_instance.problem

    # Lets see what the 5 best results are
    print("QIRO 5 best results")
    maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
    costFunc = maxIndepSetclCostfct(G)
    for key, val in res_qiro.items():  
        if key in maxfive:
            
            print(key)
            print(costFunc({key:1}))
            

    # and compare them with the networkx result of the max_clique algorithm, where we might just see a better result than the heuristical NX algorithm!
    print("Networkx solution")
    print(nx.approximation.maximum_independent_set(G))
