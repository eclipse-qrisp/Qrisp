.. _maxKColorableSubgraphQAOA:

QAOA M$\\kappa$CS
=================

The Max-$\kappa$-Colorable Subgraph problem and its QAOA implementation is discussed in plenty of detail in the :ref:`Max-$\\kappa$-Colorable Subgraph tutorial <MkCSQAOA>`.
All the necessary ingredients and required steps to run QAOA are elaborated on in an easy to grasp manner.

Here, we provide a condensed implementation of QAOA for M$\kappa$CS using all of the predefined functions.


.. currentmodule:: qrisp.qaoa.problems.maxKColorableSubgraph


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` find a maximum cut, i.e., a bipartition $S$, $V\setminus S$ of the set of vertices $V$ such that the number of edges between $S$ and $V\setminus S$ is maximal.


Cost operator
-------------

.. autofunction:: create_coloring_operator


Classical cost function
-----------------------

.. autofunction:: create_coloring_cl_cost_function


Coloring problem
----------------

.. autofunction:: graph_coloring_problem


Example implementation
----------------------

::

    from qrisp import QuantumArray
    from qrisp.qaoa import QAOAProblem, RX_mixer, apply_XY_mixer, QuantumColor, create_coloring_cl_cost_function, create_coloring_operator
    import networkx as nx
    import random
    from operator import itemgetter

    G = nx.Graph()
    G.add_edges_from([[0,1],[0,4],[1,2],[1,3],[1,4],[2,3],[3,4]])
    color_list = ["red", "blue", "yellow", "green"]

    qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc=True), shape = G.number_of_nodes())
    #qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc=False), shape = G.number_of_nodes()) # use one_hot_enc=False for binary encoding

    qaoa_coloring = QAOAProblem(cost_operator=create_coloring_operator(G),
                            mixer=apply_XY_mixer, # use RX_mixer for binary encoding
                            cl_cost_function=create_coloring_cl_cost_function(G))

    init_state = [random.choice(color_list) for _ in range(len(G))]
    qaoa_coloring.set_init_function(lambda x : x.encode(init_state))

    result = qaoa_coloring.run(qarg=qarg, depth=3, max_iter=50)

That's it! In the following, we print the 5 most likely solutions together with their cost values.

::
   
    cl_cost =create_coloring_cl_cost_function(G)

    print("5 most likely solutions")
    max_five = sorted(result.items(), key=lambda item: item[1], reverse=True)[:5]
    for res, prob in max_five:
        print(res, prob, cl_cost({res : 1}))

Finally, we visualize the most likely solution.

::

    best_of_five = min([(cl_cost({item[0]:1}),item[0]) for item in max_five], key=itemgetter(0))[1]
    nx.draw(G, with_labels=True,
            node_color=[best_of_five[node] for node in G.nodes()])

.. image:: ./mKCS.png
  :scale: 100%
  :align: center

