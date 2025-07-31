.. _maxCutQAOA:

QAOA MaxCut
===========

The MaxCut problem and its QAOA implementation is discussed in plenty of detail in the :ref:`QAOA tutorial <QAOA101>`.
All the necessary ingredients and required steps to run QAOA are elaborated on in an easy to grasp manner.

Here, we provide a condensed implementation of QAOA for MaxCut using all of the predefined functions.


.. currentmodule:: qrisp.qaoa.problems.maxCut


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` find a maximum cut, i.e., a bipartition $S$, $V\setminus S$ of the set of vertices $V$ such that the number of edges between $S$ and $V\setminus S$ is maximal.


Cost operator
-------------

.. autofunction:: create_maxcut_cost_operator


Classical cost function
-----------------------

.. autofunction:: create_maxcut_cl_cost_function


MaxCut problem
--------------

.. autofunction:: maxcut_problem


Example implementation
----------------------

::

    from qrisp import QuantumVariable
    from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxcut_cl_cost_function, create_maxcut_cost_operator
    import networkx as nx

    G = nx.erdos_renyi_graph(6, 0.7, seed = 133)
    qarg = QuantumVariable(G.number_of_nodes())

    qaoa_maxcut = QAOAProblem(cost_operator=create_maxcut_cost_operator(G),
                            mixer=RX_mixer, 
                            cl_cost_function=create_maxcut_cl_cost_function(G))
    results = qaoa_maxcut.run(qarg, depth=5, max_iter=50)

That's it! Feel free to experiment with the ``init_type='tqa'`` option in the :meth:`.run <qrisp.qaoa.QAOAProblem.run>` method for improved performance.
In the following, we print the 5 most likely solutions together with their cost values.

::
   
    cl_cost = create_maxcut_cl_cost_function(G)

    print("5 most likely solutions")
    max_five = sorted(results.items(), key=lambda item: item[1], reverse=True)[:5]
    for res, prob in max_five:
        print(res, prob, cl_cost({res : 1}))

Finally, we visualize the most likely solution.

::

    most_likely = max_five[0][0]
    nx.draw(G, with_labels = True, font_color='white', node_size=1000, font_size=22,
            node_color=['#6929C4' if most_likely[node]=='0' else '#20306f' for node in G.nodes()],
            edge_color='#D3D3D3',
            pos = nx.bipartite_layout(G, [node for node in G.nodes() if most_likely[node]=='0']))

.. image:: ./maxCut.png
  :scale: 60%
  :align: center

