.. _maxSatQIRO:

QIRO MaxSat
===========

.. currentmodule:: qrisp.qiro.qiroproblems.qiroMaxSat


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


Replacement routine
-------------------

The replacements work based on the problem encoding. We discard clauses, based on the truth value assignments as those are what is described by the qubits.
Based on the **maximum absolute entry** of the correlation matrix M and its sign, one of the following replacements is employed:

* If :math:`\text{M}_{ii} \geq 0` is the maximum absolute value, then the :math:`i`-th value is set to be true. In turn, we can remove all clauses that negate that value. 

* If :math:`\text{M}_{ii} < 0` is the maximum absolute value we remove :math:`i`-th value is set to be false and we will exclude clause that include that value. 

* If :math:`\text{M}_{ij} > 0,  (i, j) ∈ E` was selected, we transfer this correlation to the clauses, by replacing the first item with the second one. 

* If :math:`\text{M}_{ij} < 0,  (i, j) ∈ E` was selected, we transfer this correlation to the clauses, by replacing the first item with the **negation** of the second one. 



.. autofunction:: create_maxsat_replacement_routine


QIRO Cost operator 
------------------

.. autofunction:: create_maxsat_cost_operator_reduced


Example implementation
----------------------

::

    from qrisp import QuantumVariable
    from qrisp.qaoa import create_maxsat_cl_cost_function
    from qrisp.qiro import QIROProblem, qiro_init_function, qiro_RXMixer, create_maxsat_replacement_routine, create_maxsat_cost_operator_reduced

    clauses = [[1,2,-3],[1,4,-6],[4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6],[1,7,8],[3,-7,-8],[3,4,8],[4,5,8],[1,2,7]]
    num_vars = 8
    problem = (num_vars, clauses)
    qarg = QuantumVariable(num_vars)

    qiro_instance = QIROProblem(problem = problem,
                                replacement_routine = create_maxsat_replacement_routine,
                                cost_operator = create_maxsat_cost_operator_reduced,
                                mixer = qiro_RXMixer,
                                cl_cost_function = create_maxsat_cl_cost_function,
                                init_function = qiro_init_function
                                )
    res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = 3, n_recursions = 2)   

That’s it! In the following, we print the 5 most likely solutions together with their cost values, and the final clauses.

::
    
    cl_cost = create_maxsat_cl_cost_function(problem)

    print("5 most likely QIRO solutions")
    max_five_qiro = sorted(res_qiro, key=res_qiro.get, reverse=True)[:5]
    for res in max_five_qiro: 
        print(res)
        print(cl_cost({res : 1}))

    final_clauses = qiro_instance.problem
    print("final clauses")
    print(final_clauses)