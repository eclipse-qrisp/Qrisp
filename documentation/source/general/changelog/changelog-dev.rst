.. _v0.10:

Qrisp 0.10
==========

Qrisp 0.10 continues to push the boundaries of high-level quantum programming.
This release brings a streamlined contribution experience with automated
changelog enforcement.

New Features
------------

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

- **Backend-based sampling via** ``@backend_sampler``
  The new :func:`~qrisp.jasp.backend_sampler` decorator routes
  :func:`~qrisp.jasp.sample` and :func:`~qrisp.jasp.expectation_value`
  calls through a real quantum backend instead of the Jaspify simulator.
  The quantum circuit is extracted once, executed on the backend for all
  shots, and the classical post-processing (decoding, accumulator updates,
  expectation-value computation) is replayed via the Jaspr's own while-loop
  — compiled through :func:`jax.jit`.

  Key capabilities:

  * Supports any backend implementing the :ref:`Backend Interface <BackendInterface>`.
  * Handles multiple ``sample()`` / ``expectation_value()`` calls in the
    same decorated function, each independently routed.
  * Propagates the backend interception through JAX control-flow
    primitives (``fori_loop``, ``while_loop``, ``cond``, ``scan``,
    nested ``jit`` / ``pjit``).
  * Raises ``RuntimeError`` for kernels containing real-time feedback
    (mid-circuit measurements whose outcomes control subsequent gates).
  * Raises ``RuntimeError`` when quantum operations are used without a
    surrounding ``sample()`` / ``expectation_value()`` call.

Improvements
------------

- Updated docstrings for ``sample()``, ``expectation_value()``, and
  ``terminal_sampling()`` to use "sampling kernel" terminology and document
  the new arbitrary-return-value capability.

Other New Features
------------------

.. Add other new features above this line

Bug Fixes
---------

* Fixed a bug where :func:`dot <qrisp.dot>` failed with a
  ``TypeError: 'QuantumArrayIterator' object is not iterable``
  (`PR #642 <https://github.com/eclipse-qrisp/Qrisp/pull/642>`_).

* Updated Qiskit example in documentation to use ``AerSimulator`` instead of the
  deprecated ``Aer.get_backend()`` API
  (`PR #690 <https://github.com/eclipse-qrisp/Qrisp/pull/690>`_).

* Fixed Cirq ``FutureWarning`` by explicitly setting ``use_repetition_ids=True``
  in ``CircuitOperation`` calls
  (`PR #709 <https://github.com/eclipse-qrisp/Qrisp/pull/709>`_).

Compatibility
-------------

.. Add compatibility notes above this line

New Tutorials/ Updated Documentation
-------------------------------------

.. Add new tutorials above this line

API Changes
-----------

.. Add API changes above this line

Development
-----------

* Added Dependabot configuration for automated dependency updates
  (grouped by type, with labels applied automatically).

* Added a changelog reminder workflow, the ``make release-notes`` script,
  a developer changelog (``changelog-dev.rst``), a release guide,
  fixed the workflow permissions to allow posting PR comments,
  and skipped the check for Dependabot PRs
  (`PR #658 <https://github.com/eclipse-qrisp/Qrisp/pull/658>`_,
  `PR #715 <https://github.com/eclipse-qrisp/Qrisp/pull/715>`_,
  `PR #727 <https://github.com/eclipse-qrisp/Qrisp/pull/727>`_).

* Added pip dependency caching to the CI test workflow
  (`PR #685 <https://github.com/eclipse-qrisp/Qrisp/pull/685>`_).

* Added pytest coverage reporting to the CI test workflow
  (`PR #712 <https://github.com/eclipse-qrisp/Qrisp/pull/712>`_).

First Time Contributors 🎉
--------------------------

* `alighazi288 <https://github.com/alighazi288>`_
* `NedislavKolev <https://github.com/NedislavKolev>`_
* `Shanwis <https://github.com/Shanwis>`_