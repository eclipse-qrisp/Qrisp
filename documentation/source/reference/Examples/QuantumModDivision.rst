.. _QuantumModDivision:

Quantum Mod Division
====================

We demonstrate the :meth:`q_divmod` function which is the quantum equivalent of modulo division with remainder.

Defines the numbers to divide, encode them and perform the division.

>>> from qrisp import QuantumFloat, q_divmod, multi_measurement
>>> numerator = QuantumFloat(5)
>>> numerator[:] = 13
>>> divisor = QuantumFloat(5)
>>> divisor[:] = 4
>>> quotient, remainder = q_divmod(numerator, divisor)
>>> print(multi_measurement([quotient, remainder]))
{(3, 1.0): 1.0}

