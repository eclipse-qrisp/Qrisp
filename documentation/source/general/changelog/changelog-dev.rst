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

* Fixed jasp-mode crashes when tracing a ``while``/``scan`` loop with a
  single carried value, uncovered while removing duplicated interpreter
  code across four execution backends
  (`PR #770 <https://github.com/eclipse-qrisp/Qrisp/pull/770>`_).
  
* Fixed :class:`~qrisp.interface.QiskitBackend` failing with
  ``OverflowError: int too big to convert`` — or, for small classical
  registers, silently returning wrong counts — on providers that report
  measurement results as binary rather than hexadecimal strings, such as
  ``qiskit-iqm``.  Circuits are now submitted through the wrapped backend's
  own ``run()`` method and counts are read via ``Result.get_counts()``,
  instead of going through Qiskit's ``BackendSamplerV2`` primitive, which
  rebuilds counts from the hex-encoded ``memory`` field.
  :class:`~qrisp.interface.QiskitRuntimeBackend` continues to use
  ``SamplerV2``, which IBM Runtime requires.  Passing a real IBM Quantum
  backend to :class:`~qrisp.interface.QiskitBackend` now raises a ``TypeError``
  pointing at :class:`~qrisp.interface.QiskitRuntimeBackend`; IBM *fake*
  backends are unaffected
  (`PR #788 <https://github.com/eclipse-qrisp/Qrisp/pull/788>`_).

* Fixed a bug where :func:`prepare <qrisp.prepare>` with ``method="qswitch"``
  raised a ``ValueError`` when used inside an :func:`invert <qrisp.invert>` or
  :func:`control <qrisp.control>` environment in Jasp mode
  (`PR #769 <https://github.com/eclipse-qrisp/Qrisp/pull/769>`_).

* Removed reduant imports in the top-level ``qrisp`` package.
  (`PR #796 <https://github.com/eclipse-qrisp/Qrisp/pull/796>`_).

* Fixed a bug where :class:`~qrisp.QuantumModulus` constructed with a traced
  (Jasp-dynamic) modulus leaked a stale JAX tracer into subsequent, independent
  ``jaspify``/``make_jaspr`` calls, raising
  ``jax.errors.UnexpectedTracerError`` on the second and later calls. The
  modulus is now threaded through the ``QuantumVariable`` pytree as a proper
  traced attribute instead of being passed into static auxiliary data
  (`PR #802 <https://github.com/eclipse-qrisp/Qrisp/pull/802>`_).

* Updated broken link in TSP tutorial to point to the
  correct archived Qiskit textbook.
  (`PR #804 <https://github.com/eclipse-qrisp/Qrisp/pull/804>`_).

Compatibility
-------------

.. Add compatibility notes above this line

New Tutorials/ Updated Documentation
-------------------------------------

.. Add new tutorials above this line

API Changes
-----------

* :class:`~qrisp.interface.IQMBackend` is now a delegation shim that
  re-exports ``IQMBackend`` from ``iqm.qrisp_iqm`` (IQM client).
  The backend implementation and its tests live in the IQM client
  repository.  The Qrisp-side module provides a placeholder with a
  helpful ``ImportError`` when the ``iqm-client[qrisp]`` package is
  not installed.
  (`PR #757 <https://github.com/eclipse-qrisp/Qrisp/pull/757>`_).

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

* Added pytest coverage reporting to the CI test workflow and selective
  coverage reporting for the ``qrisp`` package.
  (`PR #712 <https://github.com/eclipse-qrisp/Qrisp/pull/712>`_,
  `PR #774 <https://github.com/eclipse-qrisp/Qrisp/pull/774>`_).

* Performed a large-scale refactoring of the jasp (JAX-tracing) interpreter
  subsystem, consolidating control-flow, equation-copying, and caching logic
  that had been independently duplicated across the Catalyst,
  classical-simulation, profiling, and post-processing backends into shared
  helper functions
  (`PR #770 <https://github.com/eclipse-qrisp/Qrisp/pull/770>`_).

* Added a ``reviewdog``-based CI workflow that runs ``ruff`` on pull requests
  and surfaces lint findings as inline, non-blocking review comments on
  newly added lines
  (`PR #639 <https://github.com/eclipse-qrisp/Qrisp/pull/639>`_).

Dependency Upgrades
-------------------

* Bumped myst-parser from 5.0.0 to 5.1.0
  (`PR #729 <https://github.com/eclipse-qrisp/Qrisp/pull/729>`_).

* Bumped ipykernel from 7.2.0 to 7.3.0
  (`PR #734 <https://github.com/eclipse-qrisp/Qrisp/pull/734>`_).

* Bumped pytest from 9.1.0 to 9.1.1
  (`PR #723 <https://github.com/eclipse-qrisp/Qrisp/pull/723>`_).

* Replaced pinned IQM dependencies with ``iqm-client[qrisp]`` in the
  ``iqm`` optional dependency group.
  (`PR #757 <https://github.com/eclipse-qrisp/Qrisp/pull/757>`_).
  
* Bumped ``actions/setup-python`` from 6 to 7
  (`PR #760 <https://github.com/eclipse-qrisp/Qrisp/pull/760>`_).

.. Add dependency upgrades above this line

First Time Contributors 🎉
--------------------------

* `alighazi288 <https://github.com/alighazi288>`_
* `NedislavKolev <https://github.com/NedislavKolev>`_
* `Shanwis <https://github.com/Shanwis>`_
* `micpap25 <https://github.com/micpap25>`_
