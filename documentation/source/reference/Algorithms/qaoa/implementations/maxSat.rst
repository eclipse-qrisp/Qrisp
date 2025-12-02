.. _maxSatQAOA:

QAOA MaxSat
===========


.. currentmodule:: qrisp.qaoa.problems.maxSat


Problem description
-------------------

Given :math:`m` disjunctive clauses over :math:`n` Boolean variables :math:`x_1,\dotsc,x_n`, where each clause
contains at most :math:`l \geq 2` literals, find a variable assignment that maximizes the number of
satisfied clauses. 

For example, for $l=3$ and $n=5$ Boolean variables $x_1,\dotsc,x_5$ consider the clauses

$$(x_1\\lor x_2\\lor x_3), (\\bar x_1\\lor x_2\\lor x_4), (\\bar x_2\\lor x_3\\lor \\bar x_5), (\\bar x_3\\lor x_4\\lor x_5)$$

where $\bar x_i$ denotes the negation of $x_i$. What is sought now is an assignment of the variables with 0 (False) or 1 (True)
that maximizes the number of satisfied clauses.
In this case, the assignment $x=x_1x_2x_3x_4x_5=01111$ satisfies all clauses.

Given $n$ variables $x_1,\dotsc,x_n$ we represent each clause as a list of integers such that the absolute values correspond to the indices of the variables and a minus sign corresponds to negation of the variable.
For example, the above clauses correspond to 

$$[1,2,3],\\quad [-1,2,4],\\quad [-2,3,-5],\\quad [-3,4,5]$$

The cost function for the maximum satisfyability problem is given by

$$C(x)=\\sum_{\\alpha=1}^mC_{\\alpha}(x)$$

where for each clause $C_{\alpha}(x)=1$ if the corresponding clause is satisfied and $C_{\alpha}(x)=0$ otherwise. Here, 

$$C_{\\alpha}(x)=1-\\prod_{j>0}(1-x_j)\\prod_{j<0}x_{-j}$$


Cost operator
-------------

.. autofunction:: create_maxsat_cost_operator


Classical cost function
-----------------------

.. autofunction:: create_maxsat_cl_cost_function


Helper functions
----------------

.. autofunction:: create_maxsat_cost_polynomials


MaxSat problem
--------------

.. autofunction:: maxsat_problem


Example implementation
----------------------

::

   from qrisp import QuantumVariable
   from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxsat_cl_cost_function, create_maxsat_cost_operator

   clauses = [[1,2,-3],[1,4,-6],[4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]
   num_vars = 6
   problem = (num_vars, clauses)
   qarg = QuantumVariable(num_vars)

   qaoa_max_indep_set = QAOAProblem(cost_operator=create_maxsat_cost_operator(problem),
                                    mixer=RX_mixer,
                                    cl_cost_function=create_maxsat_cl_cost_function(problem))
   results = qaoa_max_indep_set.run(qarg, depth=5)

That's it! Feel free to experiment with the ``init_type='tqa'`` option in the :meth:`.run <qrisp.qaoa.QAOAProblem.run>` method for improved performance.
In the following, we print the 5 most likely solutions together with their cost values.

::
   
   cl_cost = create_maxsat_cl_cost_function(problem)

   print("5 most likely solutions")
   max_five = sorted(results.items(), key=lambda item: item[1], reverse=True)[:5]
   for res, prob in max_five:
      print(res, prob, cl_cost({res : 1}))

