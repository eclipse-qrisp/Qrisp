.. _maxSetPackQAOA:

QAOA MaxSetPacking
==================


Problem description
-------------------

Given a universe :math:`[n]` and :math:`m` subsets :math:`\mathcal S = (S_j)^m_{j=1}` , :math:`S_j \subset [n]`, find the maximum
cardinality subcollection :math:`\mathcal S' \subset \mathcal S` of pairwise disjoint subsets.
Following the work of `Hadfield et al. <https://arxiv.org/abs/1709.03489>`_, the MaxSetPacking problem is solved by solving the MaxIndepSet problem on the constraint graph $G=(V,E)$
where vertices correspond to subsets in $\mathcal S$ and edges correspond to pairs of intersecting subsets.


Example implementation
----------------------

::
    
    from qrisp import QuantumVariable
    from qrisp.qaoa import QAOAProblem, RZ_mixer
    from qrisp.qaoa.problems.maxIndepSet import create_max_indep_set_cl_cost_function, create_max_indep_set_mixer, max_indep_set_init_function
    import networkx as nx
    from itertools import combinations

    sets = [{0,7,1},{6,5},{2,3},{5,4},{8,7,0},{1}]

    def non_empty_intersection(sets):
        return [(i, j) for (i, s1), (j, s2) in combinations(enumerate(sets), 2) if s1.intersection(s2)]

    # create constraint graph
    G = nx.Graph()
    G.add_nodes_from(range(len(sets)))
    G.add_edges_from(non_empty_intersection(sets))
    qarg = QuantumVariable(G.number_of_nodes())

    qaoa_max_indep_set = QAOAProblem(cost_operator=RZ_mixer,
                                    mixer=create_max_indep_set_mixer(G),
                                    cl_cost_function=create_max_indep_set_cl_cost_function(G),
                                    init_function=max_indep_set_init_function)
    results = qaoa_max_indep_set.run(qarg=qarg, depth=5)

That's it! In the following, we print the 5 most likely solutions together with their cost values.

::

    cl_cost = create_max_indep_set_cl_cost_function(G)

    print("5 most likely solutions")
    max_five = sorted(results.items(), key=lambda item: item[1], reverse=True)[:5]
    for res, prob in max_five:
        print([sets[index] for index, value in enumerate(res) if value == '1'], prob, cl_cost({res : 1}))
