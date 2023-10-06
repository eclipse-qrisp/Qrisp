.. _maxsatQAOA:

QAOA MaxSat problem implementation
==================================


.. currentmodule:: qaoa.problems.maxSatInfrastr

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


.. |br| raw:: html

   <br />