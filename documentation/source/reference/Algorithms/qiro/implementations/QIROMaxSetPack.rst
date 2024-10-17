.. _maxSetPackingQIRO:

QIRO MaxSetPacking
==================

.. currentmodule:: qrisp.qiro.qiroproblems.qiroMaxSetPack


Problem description
-------------------

Given a universe :math:`[n]` and :math:`m` subsets :math:`\mathcal S = (S_j)^m_{j=1}` , :math:`S_j \subset [n]`, find the maximum
cardinality subcollection :math:`\mathcal S' \subset \mathcal S` of pairwise disjoint subsets.
Following the work of `Hadfield et al. <https://arxiv.org/abs/1709.03489>`_, the MaxSetPacking problem is solved by solving the :ref:`MaxIndepSet <maxIndependentSetQIRO>` problem on the constraint graph $G=(V,E)$
where vertices correspond to subsets in $\mathcal S$ and edges correspond to pairs of intersecting subsets.


Transformation to MIS
---------------------

.. autofunction:: transform_max_set_pack_to_mis


Example implementation
----------------------

::

    from qrisp import QuantumVariable
    from qrisp.qiro import QIROProblem, create_maxIndep_replacement_routine, create_maxIndep_cost_operator_reduced, qiro_RXMixer, qiro_init_function
    from qrisp.qaoa import create_max_indep_set_cl_cost_function
    import matplotlib.pyplot as plt
    import networkx as nx

    # sets are given as list of sets
    sets = [{0,7,1},{6,5},{2,3},{5,4},{8,7,0},{2,4,7},{1,3,4},{7,9},{1,9},{1,3,8},{4,9},{0,7,9},{0,4,8},{1,4,8}]
 
    G = transform_max_set_pack_to_mis(sets)
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
        print([sets[index] for index, value in enumerate(res) if value == '1'])
        print(cl_cost({res : 1}))

    print("Networkx solution")
    print([sets[index] for index in nx.approximation.maximum_independent_set(G)])
