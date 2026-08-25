.. _QCounting:

Quantum Counting
================

.. currentmodule:: qrisp

.. autofunction:: quantum_counting

Example
-------

Use :meth:`quantum_counting <qrisp.quantum_counting>` to estimate the number of marked states in a quantum search space:

>>> from qrisp import QuantumFloat, quantum_counting
>>> from qrisp.grover import tag_state
>>> def oracle(qv):
...     for val in [0, 2, 3, 4, 5]:
...         tag_state({qv: val})
>>> qv = QuantumFloat(3, signed = False)
>>> quantum_counting(qv, oracle, 5) # doctest: +SKIP
4.780361288064514