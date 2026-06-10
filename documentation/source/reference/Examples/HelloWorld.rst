.. _HelloWorld:

Hello World
===========

The well known "hello word" program can be coded within a few lines:

>>> from qrisp import QuantumString
>>> q_str = QuantumString(size = len("hello world"))
>>> q_str[:] = "hello world"
>>> print(q_str)
{'hello world': 1.0}