.. _MatrixMultiplication:

Matrix Multiplication
=====================

In this example we will showcase how Qrisps matrix multiplication interface can be utilized.

>>> import numpy as np
>>> from qrisp import QuantumFloat, QuantumArray, QuantumBool, x, h, dot, tensordot, multi_measurement

Dot Product
-----------

Define QuantumFloat to create two QuantumArrays. Initialize the arrays and perform the multiplication with :meth:`dot <qrisp.dot>`:

>>> qf = QuantumFloat(3)
>>> q_array_0 = QuantumArray(qtype = qf, shape = 2)
>>> q_array_1 = QuantumArray(qtype = qf, shape = (2,2))
>>> q_array_0[:] = [2,3]
>>> q_array_1[:] = [[0,2],[1,0]]
>>> res = dot(q_array_0, q_array_1)
>>> print(res)
{OutcomeArray([[3, 4]]): 1.0}

Tensor Dot
----------

The :meth:`tensordot <qrisp.tensordot>` function contracts specified axes of two quantum arrays. Here we reshape a 4x4 identity matrix and contract it with a 2x2 matrix:

>>> qf = QuantumFloat(3, 0, signed=False)
>>> test_qm_0 = QuantumArray(qf, shape=(4, 4))
>>> test_qm_0[:] = np.eye(4)
>>> test_qm_0 = test_qm_0.reshape(2, 2, 2, 2)
>>> test_qm_1 = QuantumArray(qf, shape=(2, 2))
>>> test_qm_1[:] = np.eye(2)
>>> res = tensordot(test_qm_0, test_qm_1, (-1, 0))
>>> print(res)
{OutcomeArray([[1, 0], [0, 1]]): 1.0}

Semi-Classical Multiplication
-----------------------------

The ``@`` operator supports multiplication between a classical numpy array and a QuantumArray:

>>> qtype = QuantumFloat(3, signed=False)
>>> cl_array = np.array([[1, 2], [3, 4], [5, 6]])
>>> q_array = QuantumArray(qtype = qtype, shape = (2, 2))
>>> q_array[:] = [[1, 0], [0, 1]]
>>> print(cl_array @ q_array)
{OutcomeArray([[ 1  2]
 [ 3  4]
 [ 5  6]]): 1.0}

Quantum Indexing
---------------

QuantumArrays can be indexed with quantum variables to create superpositions of array entries:

>>> q_array = QuantumArray(QuantumBool(), (4, 4))
>>> index_0 = QuantumFloat(2, signed = False)
>>> index_1 = QuantumFloat(2, signed = False)
>>> index_0[:] = 2
>>> index_1[:] = 1
>>> h(index_0[0])
>>> with q_array[index_0, index_1] as entry:
...     entry.flip()
>>> print(multi_measurement([index_0, index_1, q_array]))
