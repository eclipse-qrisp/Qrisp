.. _minSetCoverQAOA:

QAOA MinSetCover
================

.. currentmodule:: qrisp.qaoa.problems.minSetCover


Problem description
-------------------

Given a universe :math:`[n]` and :math:`m` subsets :math:`\mathcal S = (S_j)^m_{j=1}` , :math:`S_j \subset [n]` find the minimum
cardinality subcollection :math:`\mathcal S' \subset \mathcal S` of the :math:`S_j` such that their union recovers :math:`[n]`.
The QAOA implementation is based on the work of `Hadfield et al. <https://arxiv.org/abs/1709.03489>`_


Mixer
-----

.. autofunction:: create_min_set_cover_mixer


Classical cost function
-----------------------

.. autofunction:: create_min_set_cover_cl_cost_function


Initial state function
----------------------

.. autofunction:: min_set_cover_init_function


MinSetCover problem
-------------------

.. autofunction:: min_set_cover_problem


Example implementation
----------------------

::
   
   from qrisp import QuantumVariable
   from qrisp.qaoa import QAOAProblem, RZ_mixer, create_min_set_cover_mixer, create_min_set_cover_cl_cost_function, min_set_cover_init_function

   sets = [{0,1,2,3},{1,5,6,4},{0,2,6,3,4,5},{3,4,0,1},{1,2,3,0},{1}]
   universe = set.union(*sets)
   qarg = QuantumVariable(len(sets))

   qaoa_min_set_cover = QAOAProblem(cost_operator=RZ_mixer, 
                                    mixer= create_min_set_cover_mixer(sets, universe), 
                                    cl_cost_function=create_min_set_cover_cl_cost_function(sets, universe),
                                    init_function=min_set_cover_init_function)
   results = qaoa_min_set_cover.run(qarg, depth=5)

That’s it! In the following, we print the 5 most likely solutions together with their cost values.

::

   cl_cost = create_min_set_cover_cl_cost_function(sets, universe)

   print("5 most likely solutions")
   max_five = sorted(results.items(), key=lambda item: item[1], reverse=True)[:5]
   for res, prob in max_five:
      print([sets[index] for index, value in enumerate(res) if value == '1'], prob, cl_cost({res : 1}))
