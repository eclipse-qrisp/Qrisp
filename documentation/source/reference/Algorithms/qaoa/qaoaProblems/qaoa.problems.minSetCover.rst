.. _minsetcoverQAOA:

QAOA MinSetCover
================

.. currentmodule:: qaoa.problems.minSetCoverInfrastr

Problem description
-------------------

Given a universe :math:`[n]` and :math:`m` subsets :math:`S = (S_j)^m_{j=1}` , :math:`S_j \subset [n]` find the maximum
cardinality subcollection :math:`S' \subset S` of the :math:`S_j` such that their union recovers :math:`[n]` .


Cost operator
-------------

.. autofunction:: minSetCoverCostOp


Classical cost function
-----------------------

.. autofunction:: minSetCoverclCostfct


Helper function
---------------

.. autofunction:: get_neighbourhood_relations


Full example implementation:
----------------------------

::
   
   from qrisp.qaoa import QAOAProblem
   from qrisp.qaoa.mixers import RZ_mixer
   from qrisp.qaoa.problems.minSetCoverInfrastr import minSetCoverclCostfct,minSetCoverCostOp, init_state

   from qrisp import QuantumVariable

   # sets are given as list of lists
   sets = [[0,1,2,3],[1,5,6,4],[0,2,6,3,4,5],[3,4,0,1],[1,2,3,0],[1]]
   # full universe is given as a tuple
   sol = (0,1,2,3,4,5,6)

   # assign operators
   cost_fun = minSetCoverclCostfct(sets=sets,universe = sol)
   mixerOp = RZ_mixer()
   costOp = minSetCoverCostOp(sets=sets, universe=sol)

   #initialize variable
   qarg = QuantumVariable(len(sets))

   #+run qaoa
   QAOAinstance = QAOAProblem(cost_operator=costOp ,mixer= mixerOp, cl_cost_function=cost_fun)
   QAOAinstance.set_init_function(init_function=init_state)
   InitTest = QAOAinstance.run(qarg=qarg, depth=5)

   # create example cost_func
   def testCostFun(state,universe):
      obj = 0
      intlist = [s for s in range(len(list(state))) if list(state)[s] == "1"]
      sol_sets = [sets[index] for index in intlist]
      res = ()
      for seto in sol_sets:
         res = tuple(set(res+ seto)) 
      if res == universe:
         obj -= len(intlist)

      return obj


   # print the most likely solutions
   print("5 most likely Solutions") 
   maxfive = sorted(InitTest, key=InitTest.get, reverse=True)[:5]
   for res, val in InitTest.items():  
      if res in maxfive:
         print((res, val))
         print(testCostFun(res, universe=sol))  
    

.. |br| raw:: html

   <br />