.. _resource_estimation:

Resource Estimation
===================

Qrisp provides utilities to estimate resources of a quantum circuit
represented by a ``jaspr``. These functions analyze the structure of the
circuit.

.. currentmodule:: qrisp.jasp

Gate count
----------

.. autofunction:: count_ops


Circuit depth
-------------

.. autofunction:: depth


Number of qubits
----------------

.. autofunction:: num_qubits