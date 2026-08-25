.. _QuantumDictionary:

QuantumDictionary
=================

.. currentmodule:: qrisp
.. autoclass:: QuantumDictionary

Example
-------

QuantumDictionary works like a Python dictionary but maps quantum variables to quantum values. Create a dictionary with flexible (string) return types:

>>> from qrisp import QuantumDictionary, QuantumFloat, QuantumVariable, h
>>> test_qd = QuantumDictionary()
>>> test_qd["hello"] = "hallo"
>>> test_qd["world"] = "welt"
>>> key_qv = QuantumVariable.custom(["hello", "world"])
>>> h(key_qv)
>>> value_qv = test_qd[key_qv]
>>> print(value_qv) # doctest: +SKIP
{'hallo': 0.5, 'welt': 0.5}

A typed QuantumDictionary returns :ref:`QuantumFloat` values that support arithmetic:

>>> float_qd = QuantumDictionary(return_type = QuantumFloat(4, -1))
>>> float_qd.update({"hallo": 1.5, "welt": 0.5})
>>> value_qf = float_qd[key_qv]
>>> value_qf += 2.5
>>> print(type(value_qf))
<class 'qrisp.qtypes.quantum_float.QuantumFloat'>

Methods
=======

.. autosummary::
   :toctree: generated/
   
   QuantumDictionary.load