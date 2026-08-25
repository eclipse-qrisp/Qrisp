.. _QuantumModDivision:

Quantum Division
================

We demonstrate the quantum division functions :meth:`q_divmod`, :meth:`q_div` and :meth:`qf_inversion`.

Modulo Division with Remainder
------------------------------

The :meth:`q_divmod` function performs modulo division with remainder.

Define the numbers to divide, encode them and perform the division:

>>> from qrisp import QuantumFloat, q_divmod, multi_measurement
>>> numerator = QuantumFloat(5)
>>> numerator[:] = 13
>>> divisor = QuantumFloat(5)
>>> divisor[:] = 4
>>> quotient, remainder = q_divmod(numerator, divisor)
>>> print(multi_measurement([quotient, remainder]))
{(3, 1.0): 1.0}

The ``prec`` parameter controls the precision of the quotient and the ``adder`` keyword selects the adder implementation:

>>> numerator = QuantumFloat(4, -2, signed=True)
>>> divisor = QuantumFloat(4, -2, signed=True)
>>> numerator.encode(0.5)
>>> divisor.encode(-1.25)
>>> quotient, remainder = q_divmod(numerator, divisor, prec=4, adder="thapliyal")
>>> print(multi_measurement([quotient, remainder]))
{(-1, 0.25): 1.0}

Division without Remainder
--------------------------

The :meth:`q_div` function performs integer division without computing the remainder:

>>> from qrisp import q_div
>>> qf_0 = QuantumFloat(5)
>>> qf_1 = QuantumFloat(5)
>>> qf_0[:] = 8
>>> qf_1[:] = 5
>>> qf_2 = q_div(qf_0, qf_1, prec=0)
>>> print(qf_2)
{1: 1.0}

Float Inversion
---------------

The :meth:`qf_inversion` function computes the multiplicative inverse of a quantum float:

>>> from qrisp import qf_inversion
>>> qf = QuantumFloat(5, -2)
>>> qf.encode(0.75)
>>> inverted_float = qf_inversion(qf, prec=6)
>>> print(inverted_float)
{1.33331298828125: 1.0}

