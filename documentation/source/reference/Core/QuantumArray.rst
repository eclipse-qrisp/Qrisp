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
   QuantumArray.get_measurement
   QuantumArray.most_likely
   QuantumArray.duplicate

Array Restructuring
-------------------

.. note::
    
    These methods never allocate additional qubits and instead return a 
    `"view" <https://numpy.org/doc/2.2/user/basics.copies.html>`_.

.. autosummary::
   :toctree: generated/

   QuantumArray.reshape
   QuantumArray.transpose
   QuantumArray.flatten
   QuantumArray.ravel
   QuantumArray.swapaxes
   QuantumArray.concatenate