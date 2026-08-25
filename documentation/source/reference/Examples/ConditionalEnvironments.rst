.. _ConditionalEnvironments:

Conditional Environments
========================

Qrisp supports conditional execution through quantum conditions. The :meth:`adaptive_condition <qrisp.adaptive_condition>` decorator enables building equality-checking functions that work with both classical values and quantum variables.

Example
-------

Define a quantum equality function using :meth:`adaptive_condition <qrisp.adaptive_condition>`:

>>> from qrisp import QuantumBool, QuantumVariable, adaptive_condition, cx, h, mcx
>>> def quantum_eq(input_0, input_1, test_kwargs = "test"):
...     return quantum_eq_inner(input_0, input_1, test_kwargs = test_kwargs)
>>> @adaptive_condition
... def quantum_eq_inner(input_0, input_1, test_kwargs = "test"):
...     res = QuantumBool(name = "cond. bool*")
...     if isinstance(input_1, QuantumVariable):
...         cx(input_0, input_1)
...         mcx(input_1, res, ctrl_state = 0)
...         cx(input_0, input_1)
...         return res
...     else:
...         label_int = input_0.encoder(input_1)
...         mcx(input_0, res, ctrl_state = label_int, method = "gray_pt")
...         return res

Use the condition in a ``with`` block to execute gates conditionally:

>>> cond_1_qv = QuantumVariable(2)
>>> cond_1_qv[:] = "11"
>>> a = QuantumBool()
>>> b = QuantumBool()
>>> h(a)
>>> cx(a, b)
>>> h(cond_1_qv[0])
>>> with quantum_eq(cond_1_qv, "11") as cond_qb_1:
...     h(b)
...     with quantum_eq(cond_1_qv, "11"):
...         cx(a, b)
...         h(a)
>>> print(a) # doctest: +SKIP
{False: 0.5, True: 0.5}
>>> print(b) # doctest: +SKIP
{False: 0.5, True: 0.5}
