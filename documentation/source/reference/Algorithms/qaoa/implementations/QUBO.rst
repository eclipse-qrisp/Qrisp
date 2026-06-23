.. _QUBOQAOA:

QAOA QUBO
=========

The QUBO problem and its QAOA implementation is discussed in plenty of detail in the :ref:`QAOA tutorial <QAOA101>`.
All the necessary ingredients and required steps to run QAOA are elaborated on in an easy to grasp manner.

Here, we provide a condensed implementation of QAOA for QUBO using all of the predefined functions.


.. currentmodule:: qrisp.qaoa.problems.QUBO


Problem description
-------------------

Given an $n\times n$ QUBO matrix $Q$ and a vector $x=(x_1,\dotsc,x_n)^T$ of binary variables $x_i\in\{0,1\}$, find an assignment of the variables $x_i$ that minimizes the cost function $C(x)=x^TQx$.


Solve QUBO
----------

.. autofunction:: solve_QUBO


Cost operator
-------------

.. autofunction:: create_QUBO_cost_operator


Classical cost function
-----------------------

.. autofunction:: create_QUBO_cl_cost_function


QUBO problem
------------

.. autofunction:: QUBO_problem


Example implementation
----------------------

::

    from qrisp import QuantumVariable, QuantumArray
    from qrisp.qaoa import QUBO_problem, QUBO_obj
    from operator import itemgetter
    import numpy as np

    Q = np.array(
        [
            [-17,  20,  20,  20,   0,  40],
            [  0, -18,  20,  20,  20,  40],
            [  0,   0, -29,  20,  40,  40],
            [  0,   0,   0, -19,  20,  20],
            [  0,   0,   0,   0, -17,  20],
            [  0,   0,   0,   0,   0, -28],
        ]
    )

    qarg = QuantumArray(qtype=QuantumVariable(1), shape=len(Q))

    QUBO_instance = QUBO_problem(Q)
    res = QUBO_instance.run(qarg, depth=1, max_iter=50)

That's it! In the following, we print the 5 best solutions together with their cost values.

::
   
    costs_and_solutions = [(QUBO_obj(bitstring, Q), bitstring) for bitstring in res.keys()]
    sorted_costs_and_solutions = sorted(costs_and_solutions, key=itemgetter(0))

    for i in range(5):
        print(f"Solution {i+1}: {sorted_costs_and_solutions[i][1]} with cost: {sorted_costs_and_solutions[i][0]} and probability: {res[sorted_costs_and_solutions[i][1]]}")


