.. _HelloWorld:

Hello World
===========

The well known "hello world" program can be coded within a few lines:

>>> from qrisp import QuantumString
>>> q_str = QuantumString(size = len("hello world"))
>>> q_str[:] = "hello world"
>>> print(q_str)
{'hello world': 1.0}

QuantumString Operations
------------------------

QuantumStrings support concatenation with the ``+=`` and ``+`` operators as well as the :meth:`duplicate <qrisp.QuantumString.duplicate>` method:

>>> q_str = QuantumString(size = len("hello"))
>>> q_str_2 = QuantumString(size = len("world"))
>>> q_str[:] = "hello"
>>> q_str_2[:] = "world"
>>> q_str += " "
>>> q_str += q_str_2
>>> q_str += "! "
>>> q_str_3 = q_str.duplicate(init = True)
>>> print(q_str + q_str_3)
{'hello world! hello world!': 1.0}