.. _Loops:

Loops
=====

In this example we will showcase the qRange iterator. Using this object, we can perform loops where the termination condition is based on a QuantumFloat. ::


   from qrisp import QuantumFloat, qRange, h

   n = QuantumFloat(3, signed = True, name = "n")
   qf = QuantumFloat(5, name = "qf")


   n[:] = 4

   h(n[0])

   n_results = n.get_measurement()


   for i in qRange(n):
       qf += i



According to Gauß's formula, we have

.. math::
   
   \sum_{k = 1}^n k = \frac{n (n+1)}{2}


Therefore our expectation is: 

>>> print([n*(n+1)/2 for n in n_results.keys()])
[10.0, 15.0]

Querying the simulator delivers the result:

>>> print(qf)
{10: 0.5, 15: 0.5}
