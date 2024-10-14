.. _eThrLinTwoQAOA:


QAOA E3Lin2
===========


.. currentmodule:: qrisp.qaoa.problems.eThrLinTwo


Problem description
-------------------


Given a set of :math:`m` three-variable equations :math:`A = \{A_j\}`, over $n$ binary variables :math:`x\in\{0,1\}^n`, where each equation :math:`A_j` is of the form  
:math:`x_{a_{1,j}} + x_{a_{2,j}} + x_{a_{3,j}} = b_j \text{ mod }  2`,
where :math:`a_{1,j},\;a_{2,j},\;a_{3,j}\in [n]` and :math:`b_j\in\{0,1\}`, find an assignment :math:`x\in\{0,1\}^n` that maximizes the number of satisfied equations.

Here, the equations are represented by a list of clauses, for example ``[0,1,3,1]`` corresponds to $[a_{1,j},a_{2,j},a_{3,j},b_j]$, that is, the equation

$$x_0+x_1+x_3 = 1 \\mod 2$$


Cost operator
-------------

.. autofunction:: create_e3lin2_cost_operator


Classical cost function
-----------------------

.. autofunction:: create_e3lin2_cl_cost_function


Inital state function
---------------------

.. autofunction:: e3lin2_init_function


E3Lin2 problem
--------------

.. autofunction:: e3lin2_problem


Example implementation
----------------------

::

   from qrisp import QuantumVariable
   from qrisp.algorithms.qaoa import QAOAProblem, RX_mixer, create_e3lin2_cl_cost_function, create_e3lin2_cost_operator, e3lin2_init_function

   clauses = [[0,1,2,1],[1,2,3,0],[0,1,4,0],[0,2,4,1],[2,4,5,1],[1,3,5,1],[2,3,4,0]]
   num_variables = 6
   qarg = QuantumVariable(num_variables)

   qaoa_e3lin2 = QAOAProblem(cost_operator=create_e3lin2_cost_operator(clauses),
                              mixer=RX_mixer,
                              cl_cost_function=create_e3lin2_cl_cost_function(clauses),
                              init_function=e3lin2_init_function)
   results = qaoa_e3lin2.run(qarg=qarg, depth=5)

That's it! In the following, we print the 5 most likely solutions together with their cost values.

::

   cl_cost = create_e3lin2_cl_cost_function(clauses)

   print("5 most likely solutions")
   max_five = sorted(results.items(), key=lambda item: item[1], reverse=True)[:5]
   for res, prob in max_five:
      print(res, prob, cl_cost({res : 1}))

