.. _maxsetpackQAOA:

QAOA MaxSetPacking problem implementation
=========================================



.. currentmodule:: qaoa.problems.maxSetPackInfrastr


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

