.. _prepare:

Quantum State Preparation
=========================

.. currentmodule:: qrisp

.. autofunction:: prepare

Example
-------

Prepare an arbitrary state with complex amplitudes by assigning a dictionary to a :ref:`QuantumFloat`:

>>> from qrisp import QuantumFloat
>>> qf = QuantumFloat(4, -2, signed = True)
>>> state_dic = {2.75: 1/4**0.5, -1.5: -1/4**0.5, 2: 1/4**0.5, 3: 1j/4**0.5}
>>> qf[:] = state_dic

Inspect individual amplitudes with the statevector debugger:

>>> debugger = qf.qs.statevector("function")
>>> print("Amplitude of state 2.75: ", debugger({qf: 2.75})) # doctest: +SKIP
Amplitude of state 2.75:  (0.50000006+3.0615774e-07j)
>>> print("Amplitude of state 3: ", debugger({qf: 3})) # doctest: +SKIP
Amplitude of state 3:  (-2.7677137e-07+0.5000001j)
