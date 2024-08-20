.. _eThrTwoLinQAOA:


QAOA eThrTwoLin
===============

This is a QAOA implementation of the eThrLinTwo problem.


Problem description
-------------------


Given a set of :math:`m` three-variable equations :math:`A = {A_j}` ,  over n binary variables :math:`x \in \{ 0, 1 \}^n` , where each equation :math:`A_j` is of the form  
:math:`x_{a_{1,j}} + x_{a_{1,j}} + x_{a_{1,j}} = b_j \text{ mod }  2`,
where :math:`a_{1,j} \, ; \,  a_{1,j}; \,  a_{1,j} \in [ n ]` and :math:`b_j \in \{ 0 , 1 \}` find an assignment  :math:`x \in \{ 0, 1 \}^n` that maximizes the number of satisfied equations.

.. currentmodule:: qrisp.qaoa.problems.eThrTwoLinInfrastr



Cost operator
-------------

.. autofunction:: eThrTwocostOp


Classical cost function
-----------------------

.. autofunction:: eTwoThrLinclCostfct



.. |br| raw:: html

   <br />