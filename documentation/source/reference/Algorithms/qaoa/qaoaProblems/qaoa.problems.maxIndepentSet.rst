.. _maxindependentsetQAOA:

QAOA MaxIndependentSet
======================



.. currentmodule:: qrisp.qaoa.problems.maxIndepSetInfrastr


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` find a maximal independent set, i.e., a subset of vertices :math:`V' \subset V` such that all pairs of vertices are mutually non-adjacent in the graph $G$.

Cost operator
-------------

.. autofunction:: maxIndepSetCostOp


Classical cost function
-----------------------

.. autofunction:: maxIndepSetCostfct


Example implementation:
-----------------------

::

    # imports
    from qrisp.qaoa import QAOAProblem
    from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetCostOp, maxIndepSetCostfct
    from qrisp.qaoa.mixers import RX_mixer
    from qrisp import QuantumVariable
    import networkx as nx
    import matplotlib.pyplot as plt

    # create graph
    G = nx.erdos_renyi_graph(9, 0.5, seed =  133)
    qarg = QuantumVariable(G.number_of_nodes())

    # run the instance
    QAOAinstance = QAOAProblem(cost_operator= maxIndepSetCostOp(G), mixer= RX_mixer, cl_cost_function=maxIndepSetCostfct(G))
    theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

    clCostfct = maxIndepSetCostfct(G)

    # print the most likely solutions
    print("5 most likely solutions")
    maxfive = sorted(theNiceQAOA.items(), key=lambda item: item[1], reverse=True)[:5]
    for res, val in maxfive:
        print([index for index, value in enumerate(res) if value == '1'], val)
        print(clCostfct({res : 1}))

    print("NX solution")
    print(nx.maximal_independent_set(G))

    # draw graph
    nx.draw(G,with_labels = True)
    plt.show()