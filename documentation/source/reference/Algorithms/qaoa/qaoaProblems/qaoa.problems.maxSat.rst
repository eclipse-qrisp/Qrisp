.. _maxsatQAOA:

QAOA MaxSat
===========


.. currentmodule:: qrisp.qaoa.problems.maxSatInfrastr

Problem description
-------------------

Given :math:`m` disjunctive clauses over :math:`n` Boolean variables :math:`x` , where each clause
contains at most :math:`l \geq 2` literals, find a variable assignment that maximizes the number of
satisfied clauses. 

Cost operator
-------------

.. autofunction:: maxSatCostOp


Classical cost function
-----------------------

.. autofunction:: maxSatclCostfct


Helper function
---------------

.. autofunction:: clausesdecoder


Full example implementation:
----------------------------

::

   from qrisp.qaoa import QAOAProblem
   from qrisp.qaoa.problems.maxSatInfrastr import maxSatclCostfct, maxSatCostOp, clausesdecoder, init_state
   from qrisp.qaoa.mixers import RX_mixer
   from qrisp import QuantumVariable


   clauses11 = [[1,2,-3],[1,4,-6], [4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]

   #Clauses are decoded, s.t. the Cost-Optimizer can read them
   #numVars is the amount of considered variables, i.e. highest number (= Number of Qubits in Circuit aswell)
   decodedClauses = clausesdecoder( clauses = clauses11, numVars = 6)
   #print(decodedClauses)

   qarg = QuantumVariable(len(clauses11))

   #CostOperator-Generator has to be called with the clauses
   #CostFct-Generator has to be called with decodedClauses
   QAOAinstance = QAOAProblem(cost_operator=maxSatCostOp(clauses11), mixer=RX_mixer, cl_cost_function=maxSatclCostfct(decodedClauses))
   QAOAinstance.set_init_function(init_function=init_state)
   theNiceQAOA = QAOAinstance.run(qarg=qarg, depth=5)

    #print the ideal solutions
    print("5 most likely Solutions") 
    maxfive = sorted(theNiceQAOA, key=theNiceQAOA.get, reverse=True)[:5]
    for res, val in theNiceQAOA.items(): 
        if res in maxfive:
            print((res, val))
            print(clCostfct({res : 1}))
            

   print("Final energy value and associated solution values")
   costfct = maxSatclCostfct(decodedClauses)
   print(costfct(theNiceQAOA))
   #Final Result-dictionary
   resDict = dict()
   for index in decodedClauses:
      for index2 in index:
         if index2 in resDict.keys():
               resDict[index2] +=1
         else:
               resDict[index2] =1
   print(resDict)



.. |br| raw:: html

   <br />