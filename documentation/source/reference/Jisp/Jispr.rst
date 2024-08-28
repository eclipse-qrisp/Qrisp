.. _jispr:

Jispr
=====

.. currentmodule:: qrisp.jisp
.. autoclass:: Jispr

Methods
=======

.. autosummary::
   :toctree: generated/
   
  
   Jispr.inverse
   Jispr.control
   Jispr.to_qc
   Jispr.to_qir
   Jispr.to_mlir
   Jispr.to_catalyst_jaxpr
   

Advanced details
================

Jispr defines 3 new Jax data types:
        
* QuantumCircuit, which represents an object that tracks what kind of manipulations are applied to the quantum state.
* QubitArray, which represents an array of qubits, that can have a dynamic amount of qubits
* Qubit, which represents individual qubits.
    