.. _maxcliqueQAOA:

QAOA MaxClique
==============

.. currentmodule:: qrisp.qaoa.problems.maxCliqueInfrastr


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` find a maximal clique, i.e., a subset of vertices :math:`V' \subset V` such that all pairs of vertices are connected by an edge in $G$.


Cost operator
-------------

.. autofunction:: maxCliqueCostOp


Classical cost function
-----------------------

.. autofunction:: maxCliqueCostfct


Full Example implementation:
----------------------------
::

    # imports
    from qrisp.qaoa import QAOAProblem
    from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct, maxCliqueCostOp 
    from qrisp.qaoa.mixers import RX_mixer
    from qrisp import QuantumVariable
    import networkx as nx
    import matplotlib.pyplot as plt

    # create graph
    G = nx.erdos_renyi_graph(9,0.7, seed =  133)
    qarg = QuantumVariable(G.number_of_nodes())

    # run the instance
    QAOAinstance = QAOAProblem(cost_operator= maxCliqueCostOp(G), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(G))
    theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

    clCostfct = maxCliqueCostfct(G)

    # print the most likely solutions
    print("5 most likely solutions")
    maxfive = sorted(theNiceQAOA.items(), key=lambda item: item[1], reverse=True)[:5]
    for res, val in maxfive:
    print([index for index, value in enumerate(res) if value == '1'], val)
    print(clCostfct({res : 1}))

    print("NX solution")
    print(nx.max_weight_clique(G, weight = None))

    # draw graph
    nx.draw(G,with_labels = True)
    plt.show()