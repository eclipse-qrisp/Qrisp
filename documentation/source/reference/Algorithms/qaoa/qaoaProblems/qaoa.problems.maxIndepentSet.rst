.. _maxindependentsetQAOA:

QAOA MaxIndependentSet
======================



.. currentmodule:: qaoa.problems.maxIndepSetInfrastr


Problem description
-------------------

This implementation is descibed in the tutorial section. Go have a look for indepth explanations!

Given a Graph  :math:`G = (V,E)` maximize the size of a clique, i.e. a subset :math:`V' \subset V` in which all pairs of vertices are mutually non-adjacent.

Cost operator
-------------

.. autofunction:: maxIndepSetCostOp


Classical cost function
-----------------------

.. autofunction:: maxIndepSetclCostfct

