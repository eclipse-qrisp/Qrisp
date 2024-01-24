.. _maxcliqueQAOA:

QAOA MaxClique problem implementation
=====================================

.. currentmodule:: qaoa.problems.maxCliqueInfrastr


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` maximize the size of a clique, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are adjacent.






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
    from qrisp.qaoa.problems.create_rdm_graph import create_rdm_graph
    from qrisp.qaoa.problems.maxCliqueInfrastr import maxCliqueCostfct,maxCliqueCostOp,init_state
    from qrisp.qaoa.mixers import RX_mixer
    from qrisp import QuantumVariable
    import networkx as nx
    import matplotlib.pyplot as plt

    #create graph
    G = create_rdm_graph(9,0.7, seed =  133)
    qarg = QuantumVariable(G.number_of_nodes())

    #run the instance
    QAOAinstance = QAOAProblem(cost_operator= maxCliqueCostOp(G), mixer= RX_mixer, cl_cost_function=maxCliqueCostfct(G)) 
    QAOAinstance.set_init_function(init_function=init_state)
    theNiceQAOA = QAOAinstance.run(qarg=qarg, depth= 5)

    clCostfct = maxCliqueCostfct(G)

    #print the ideal solutions
    print("5 most likely Solutions") 
    maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
    for res, val in theNiceQAOA.items():  
        if res in maxfive:
            print((res, val))
            print(clCostfct({res : 1}))

    print("NX solution")
    print(nx.max_weight_clique(G, weight = None))
    #draw graph
    nx.draw(G,with_labels = True)
    plt.show() 
