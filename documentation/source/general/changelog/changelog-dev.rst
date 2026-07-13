Development Changelog
--------------------

.. _changelog_dev:

New Features
~~~~~~~~~~~~

- **sample() and expectation_value() now accept arbitrary return values**
  Sampling kernels (the functions passed to :func:`~qrisp.jasp.sample` and
  :func:`~qrisp.jasp.expectation_value`) may now return classical values
  from mid-circuit measurements, ``QuantumVariable``\ s, or a mixture of
  both.  Previously only ``QuantumVariable`` returns were supported.
  ``QuantumVariable``\ s in the return are automatically measured and
  decoded; classical values are interleaved in-place.

  Terminal sampling (decorator and Japify with ``terminal_sampling=True``) 
  rejects kernels that return classical values with a descriptive 
  error — use ``terminal_sampling=False`` (the default) for those cases.

Improvements
~~~~~~~~~~~~

- Updated docstrings for ``sample()``, ``expectation_value()``, and
  ``terminal_sampling()`` to use "sampling kernel" terminology and document
  the new arbitrary-return-value capability.
