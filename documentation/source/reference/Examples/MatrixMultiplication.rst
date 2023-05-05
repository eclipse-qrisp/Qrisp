.. _MatrixMultiplication:

Matrix Multiplication
=====================

In this example we will showcase how Qrisps matrix multiplication interface can be utilized.

>>> import numpy as np
>>> from qrisp import QuantumFloat, QuantumArray, x, h, dot

Define QuantumFloat to create two QuantumArrays. Initialize the arrays and perform the multiplication with :meth:`dot <qrisp.dot>`:

>>> qf = QuantumFloat(3)
>>> q_array_0 = QuantumArray(qtype = qf)
>>> q_array_1 = QuantumArray(qtype = qf)
>>> q_array_0[:] = [2,3]
>>> q_array_1[:] = [[0,2],[1,0]]
>>> res = dot(q_array_0, q_array_1)
>>> print(res)
{OutcomeArray([[3, 4]]): 1.0}
