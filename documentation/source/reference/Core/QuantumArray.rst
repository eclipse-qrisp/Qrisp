.. _QuantumArray:

QuantumArray
============

.. currentmodule:: qrisp
.. autoclass:: QuantumArray

Methods
=======

.. autosummary::
   :toctree: generated/
   
   QuantumArray.delete
   QuantumArray.duplicate
   QuantumArray.encode
   QuantumArray.get_measurement
   QuantumArray.most_likely

Array Restructuring
-------------------

.. note::
    
    These methods never allocate additional qubits and instead return a 
    `"view" <https://numpy.org/doc/2.2/user/basics.copies.html>`_.

.. autosummary::
   :toctree: generated/

   QuantumArray.concatenate
   QuantumArray.flatten
   QuantumArray.ravel
   QuantumArray.reshape
   QuantumArray.swapaxes
   QuantumArray.transpose

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   QuantumArray.__add__
   QuantumArray.__sub__
   QuantumArray.__mul__
   QuantumArray.__eq__
   QuantumArray.__ne__
   QuantumArray.__gt__
   QuantumArray.__ge__
   QuantumArray.__lt__
   QuantumArray.__le__

   
   
   
   