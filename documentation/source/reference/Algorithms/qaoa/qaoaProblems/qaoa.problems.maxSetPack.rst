.. _maxsetpackQAOA:

QAOA MaxSetPacking
==================



.. currentmodule:: qrisp.qaoa.problems.maxSetPackInfrastr


Problem description
-------------------

Given a universe :math:`[n]` and :math:`m` subsets :math:`S = (S_j)^m_{j=1}` , :math:`S_j \subset [n]` find the maximum
cardinality subcollection :math:`S' \subset S` of pairwise disjoint subsets.


Cost operator
-------------

.. autofunction:: maxSetPackCostOp


Classical cost function
-----------------------

.. autofunction:: maxSetPackclCostfct


Helper function
---------------

.. autofunction:: get_neighbourhood_relations


Full example implementation:
----------------------------

::
    
    from qrisp.qaoa import QAOAProblem
    from qrisp.qaoa.mixers import RZ_mixer
    from qrisp.qaoa.problems.maxSetPackInfrastr import maxSetPackclCostfct,maxSetPackCostOp, get_neighbourhood_relations, init_state
    from qrisp import QuantumVariable

    # sets are given as list of lists
    sets = [[0,7,1],[6,5],[2,3],[5,4],[8,7,0],[1]]
    # full universe is given as a tuple
    sol = (0,1,2,3,4,5,6,7,8)

    # the realtions between the sets, i.e. with vertice is in which other sets
    print(get_neighbourhood_relations(sets, len_universe=len(sol)))

    # assign the operators
    cost_fun = maxSetPackclCostfct(sets=sets,universe=sol)
    mixerOp = RZ_mixer()
    costOp = maxSetPackCostOp(sets=sets, universe=sol)

    #initialize the qarg
    qarg = QuantumVariable(len(sets))

    # run the qaoa
    QAOAinstance = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
    QAOAinstance.set_init_function(init_function=init_state)
    InitTest = QAOAinstance.run(qarg=qarg, depth=5)

    # assign the cost_func
    def testCostFun(state, universe):
        list_universe = [True]*len(universe)
        temp = True
        obj = 0
        intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
        sol_sets = [sets[index] for index in intlist]
        for seto in sol_sets:
            for val in seto:
                if list_universe[val]:
                    list_universe[val] = False
                else: 
                    temp = False 
                    break
        if temp:
            obj -= len(intlist)
        return obj

    # print the most likely solutions
    print("5 most likely Solutions") 
    maxfive = sorted(InitTest, key=InitTest.get, reverse=True)[:5]
    for res, val in InitTest.items():  
        if res in maxfive:
            print((res, val))
            print(testCostFun(res, universe=sol))  