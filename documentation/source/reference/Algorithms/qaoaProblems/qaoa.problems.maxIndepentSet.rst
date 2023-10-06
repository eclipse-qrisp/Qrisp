.. _QAOAMaxIndependentSet:

QAOA MaxIndependentSet problem implementation
=============================================



.. currentmodule:: qaoa.problems.maxIndepSetInfrastr


Problem description
-------------------

Given a Graph  :math:`G = (V,E)` maximize the size of a clique, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are mutually non-adjacent.

Cost operator
-------------

.. autofunction:: maxIndepSetCostOp


Classical cost function
-----------------------

.. autofunction:: maxIndepSetclCostfct

