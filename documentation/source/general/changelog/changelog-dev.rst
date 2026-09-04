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
  * Raises ``ValueError`` for a non-positive shot count — including a
    dynamic one, which only becomes concrete once the backend runs.

Improvements
------------

- :class:`~qrisp.interface.QiskitJob` and :class:`~qrisp.interface.AQTJob`
  now skip the live provider query and return the cached status once a job
  is done, cancelled, or errored. The :class:`~qrisp.interface.Job` base
  class docstring for ``status()``/``refresh()`` now consistently describes
  this as an optional optimization implementations may take advantage of,
  rather than a guarantee
  (`PR #806 <https://github.com/eclipse-qrisp/Qrisp/pull/806>`_).

- Updated docstrings for ``sample()``, ``expectation_value()``, and
  ``terminal_sampling()`` to use "sampling kernel" terminology and document
  the new arbitrary-return-value capability.

- Improved the simulator's circuit preprocessing: circuit reordering is
  faster, and gate grouping for circuits with 63+ qubits now stays on the
  fast Numba-jitted path (via chunked qubit bitmasks) instead of falling
  back to a slower, non-jitted implementation. All functions and classes
  used for simulator preprocessing are strictly internal and marked with a
  leading underscore.
  (`PR #704 <https://github.com/eclipse-qrisp/Qrisp/pull/704>`_)

Other New Features
------------------

- Added an ``RYYGate`` for Ising-YY couplings to accompany the existing ``RXXGate``
  and ``RZZGate``
  (`PR #797 <https://github.com/eclipse-qrisp/Qrisp/pull/797>`_).
  
- **Dicke state preparation via divide-and-conquer**
  :func:`~qrisp.dicke_state` now accepts a ``method`` keyword argument.  In
  addition to the existing ``"deterministic"`` method
  (`arXiv:1904.07358 <https://arxiv.org/abs/1904.07358>`_), the new
  ``"divide-and-conquer"`` method
  (`arXiv:2112.12435 <https://arxiv.org/abs/2112.12435>`_) prepares the two
  halves of the variable on disjoint qubits, roughly halving the circuit
  depth.  It requires the input to have Hamming weight exactly ``k``, whereas
  ``"deterministic"`` implements the full Dicke state unitary and therefore
  also accepts any input weight ``l <= k``, preparing ``D(n, l)``.  The
  default is ``"deterministic"``, so existing code is unaffected
  (`PR #767 <https://github.com/eclipse-qrisp/Qrisp/pull/767>`_).

- **Added an AI policy note to the issue templates**
  All issue templates now state that the project does not accept
  AI-generated pull requests and that automated agents should not submit PRs
  or post comments. Human contributors may use LLMs as an aid, provided they
  fully understand and take responsibility for the changes implemented
  (`PR #816 <https://github.com/eclipse-qrisp/Qrisp/pull/816>`_).

.. Add other new features above this line

Bug Fixes
---------

* Fixed the precision of :meth:`get_unitary <qrisp.QuantumCircuit.get_unitary>`.
  Unitary matrices are now computed in ``complex128`` precision, removing the
  spurious ~1e-7 off-diagonal entries that previously appeared where a
  unitary should vanish exactly (e.g. phase-tolerant controlled gates).
  (`PR #787 <https://github.com/eclipse-qrisp/Qrisp/pull/787>`_).

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
  
* Fixed :func:`dicke_state <qrisp.dicke_state>` for ``k = 0``, which emitted a
  ladder of identity-acting *Split & Cyclic Shift* blocks and traced a
  negative-length loop range under Jasp
  (`PR #767 <https://github.com/eclipse-qrisp/Qrisp/pull/767>`_).

* Fix a code typo in the Jasp tutorial which printed the wrong variable
  when checking which variables are dynamic.
  (`PR #828 <https://github.com/eclipse-qrisp/Qrisp/pull/828>`_).

* Removed a stray double blank line in ``QubitOperator.simulate``, left behind
  by an import-hoisting cleanup, which broke ``ruff format --check`` on
  ``main`` right after merge.

Compatibility
-------------

* :func:`~qrisp.dicke_state` now raises a ``ValueError`` for an unrecognized
  ``method``, and for a ``k`` outside ``0 <= k <= len(qv)`` when both are
  plain Python integers (i.e. outside of Jasp tracing).  The latter previously
  produced an incorrect state silently
  (`PR #767 <https://github.com/eclipse-qrisp/Qrisp/pull/767>`_).

.. Add compatibility notes above this line

New Tutorials/ Updated Documentation
-------------------------------------

- Fixed outdated or inaccurate docstrings and examples across the Jasp
  module (control flow, sampling, simulators, optimization tools,
  ``BigInteger``, and ``Jaspr`` MLIR/QIR export)
  (`PR #805 <https://github.com/eclipse-qrisp/Qrisp/pull/805>`_).

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

* Renamed the *Split & Cyclic Shift* helper used by
  :func:`~qrisp.dicke_state` from ``split_cycle_shift`` to
  ``_split_cycle_shift``, marking it private.  Its parameters were renamed
  from ``highIndex``/``lowIndex`` to ``n``/``k`` to match the notation of
  `arXiv:1904.07358 <https://arxiv.org/abs/1904.07358>`_.  The unitary
  implemented is unchanged
  (`PR #814 <https://github.com/eclipse-qrisp/Qrisp/pull/814>`_).

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

* Added an ``all`` optional dependency group that installs the base package
  plus every other extra except ``aqt`` (``qiskit``, ``iqm``, ``catalyst``,
  ``xdsl``, ``docs``, and ``dev``) and updated the Development Guide's
  installation instructions to reference it
  (`PR #807 <https://github.com/eclipse-qrisp/Qrisp/pull/807>`_).
* Added a ``reviewdog``-based CI workflow that runs ``ruff`` on pull requests
  and surfaces lint findings as annotations on the GitHub Checks tab of
  newly added lines instead of as inline review comments on the PR
  (`PR #639 <https://github.com/eclipse-qrisp/Qrisp/pull/639>`_,
  `PR #812 <https://github.com/eclipse-qrisp/Qrisp/pull/812>`_).

* Converted every source file's Eclipse Public License header from a
  module-level docstring into a ``#``-prefixed comment block, and gave
  every file that lacked one a real one-line module docstring. The header
  had been written as a docstring, which pydocstyle interpreted as the
  module's documentation and flagged for style violations (``D205``,
  ``D212``, ...) on essentially every file; it was never meant to be read
  as documentation. Existing rich module-level documentation was preserved
  verbatim, only reformatted to satisfy ``D205``
  (`PR #820 <https://github.com/eclipse-qrisp/Qrisp/pull/820>`_).

* Extended the ``ruff`` ignore list in ``pyproject.toml`` with the docstring
  style rules ``D209``, ``D212``, ``D401``, ``D402``, ``D404``, and ``D416``
  (relaxing pydocstyle conventions), plus ``PLC0415`` (function-level imports
  used to avoid circular imports) and ``E402`` (module-level imports placed
  after a module docstring)
  (`PR #811 <https://github.com/eclipse-qrisp/Qrisp/pull/811>`_).

* Added type hints across ``BlockEncoding`` and the ``QubitOperator``/
  ``Hamiltonian`` operator algebra. This exposed two latent bugs:
  ``BlockEncoding``'s constructor methods (``from_lcu``, ``from_operator``,
  etc.) had their ``cls`` parameter typed as an instance rather than
  ``type[BlockEncoding]``, and ``Hamiltonian``'s abstract methods were
  typed as returning ``None``, breaking every subclass override
  (`PR #817 <https://github.com/eclipse-qrisp/Qrisp/pull/817>`_).

* Added a ``pull_request`` trigger to the ``ruff format --check`` workflow,
  which previously ran only on pushes to ``main``. This meant formatting
  regressions were never caught during PR review and only surfaced once
  merged into ``main``.

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

* Pinned ``ruff`` to ``0.15.18`` in the ``dev-code-style`` dependency group
  and updated the ``reviewdog`` CI workflow to install the version specified
  in ``pyproject.toml``
  (`PR #819 <https://github.com/eclipse-qrisp/Qrisp/pull/819>`_).

* Bumped ``ruff`` from ``0.15.18`` to ``0.16.4``. On ``0.15.x``,
  ``ruff format --check --output-format=rdjson`` silently required
  ``--preview`` mode just to emit structured output at all — an unrelated
  concern from preview *formatting rules*, which the project does not use —
  causing the ``reviewdog`` formatter check to crash. ``0.16.0`` stabilized
  structured output formats for the formatter, fixing this without changing
  which formatting rules apply
  (`PR #820 <https://github.com/eclipse-qrisp/Qrisp/pull/820>`_).

.. Add dependency upgrades above this line

First Time Contributors 🎉
--------------------------

* `alighazi288 <https://github.com/alighazi288>`_
* `NedislavKolev <https://github.com/NedislavKolev>`_
* `Shanwis <https://github.com/Shanwis>`_
* `micpap25 <https://github.com/micpap25>`_
* `JiriGuthJarkovsky <https://github.com/JiriGuthJarkovsky>`_
