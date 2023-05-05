.. _InplaceMatrixMultiplication:

In-Place Matrix Multiplication
=================================

In this example we will demonstrate how the :meth:`inplace_matrix_app<qrisp.inplace_matrix_app>` function can be utilized to perform an inplace multiplication of a QuantumArray with a classical matrix.


>>> from qrisp import QuantumArray, QuantumFloat, inplace_matrix_app
>>> import numpy as np

Due to reversibility, this operation can only be realized with an invertible matrix. Overflow is handled by a `modular <https://en.wikipedia.org/wiki/Modular_arithmetic>`_ behavior, such that the matrix needs to be invertible over $\mathbb{Z}/{2^n}\mathbb{Z} $. This is equivalent to the determinant being odd. We generate a random matrix having this property.

::

    import random
    
    def generate_random_inv_matrix(n, bit):
        
        
        found = False

        while found == False:
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = random.randint(0, 2 ** bit - 1)

            det = np.round(np.linalg.det(matrix) % 2 ** bit)

            found = bool(det%2)

        return matrix


The in-place matrix multiplication applies a classical matrix (in the form of a numpy array) to a QuantumArray of suitable size.

::

   bit = 5
   n = 3

   qtype = QuantumFloat(bit)
   vector = QuantumArray(qtype = qtype, shape = n)

   x_values = np.array([0, 2, 1])
   vector[:] = x_values

   matrix = generate_random_inv_matrix(n, bit)
   
   inplace_matrix_app(vector, matrix)

Evaluate the result:

>>> print(vector)
{OutcomeArray([ 8, 25, 17]): 1.0}

Compare with the classical expectation:

>>> print(((np.dot(matrix, x_values)) % (2 ** bit)).astype(int).transpose())
[ 8 25 17]
