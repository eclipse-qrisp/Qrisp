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

