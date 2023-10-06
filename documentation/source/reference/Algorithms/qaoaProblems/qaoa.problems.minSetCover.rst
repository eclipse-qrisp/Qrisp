.. _minsetcoverQAOA:

QAOA MinSetCover problem implementation
=======================================

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


    

.. |br| raw:: html

   <br />