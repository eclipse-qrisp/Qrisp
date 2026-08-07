.. _v0.10:

Qrisp 0.10
==========

Qrisp 0.10 continues to push the boundaries of high-level quantum programming.
This release brings a streamlined contribution experience with automated
changelog enforcement.

New Features
------------

- **Pytree-aware return signatures for** ``sample()`` **and**
  ``expectation_value()``
  The return structure now mirrors the structure of the sampling kernel's
  return value.  For example ``return a, b, c`` from a kernel produces
  ``(array_a, array_b, array_c)`` — a tuple of three 1D arrays — instead of
  a single flat ``(shots, 3)`` array.  Nested tuples, lists, and dicts are
  preserved (e.g. ``[ (a,b), c ]`` → ``[ (array_a, array_b), array_c ]``).
  Each leaf retains its native dtype (``bool`` stays ``bool``, ``int`` stays
  ``int``) and array-valued leaves naturally stack along the leading
  dimension (``(3,)`` → ``(shots, 3)``).  User-defined JAX pytree types
  raise a descriptive ``TypeError``.

  **Breaking change:** Multi-value returns are now tuples of arrays instead
  of a single 2D array.  Code that indexed ``res[:, i]`` must use
  ``res[i]``.  The same applies to ``expectation_value()`` which now returns
  a tuple of scalars for multi-value kernels.

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

- **layerize — ASAP circuit scheduling pass**
  The new :func:`~qrisp.layerize` pass reorders circuit instructions
  into as-soon-as-possible layers: gates acting on disjoint qubits are
  pulled into the same time layer, compacting the Stim timeline diagram
  without changing circuit semantics.  Only instructions that represent a
  physical time step advance the layer clock, so each error channel stays in
  the time step of the gate it annotates, and each layer is emitted so that
  Stim actually draws it in parallel.

  ``layerize(insert_barriers=True)`` additionally writes the schedule
  back into the circuit as barriers, giving one ``TICK`` per time step in the
  Stim output.

- **promote_barriers — widen barriers to the full circuit**
  The new :func:`~qrisp.promote_barriers` pass rewrites every barrier in a
  circuit to span all of its qubits, which is how a local fence is declared to
  be a global time boundary.  Promotion adds scheduling constraints, so it 
  is an explicit opt-in: it widens the schedule and inflates the idle-noise
  budget of the qubits the original barriers did not name.

- **insert_stim_noise — circuit-level Stim noise model**
  The new :func:`~qrisp.insert_stim_noise` pass annotates a
  :class:`~qrisp.QuantumCircuit` with the circuit-level depolarizing noise
  model used for quantum error-correction benchmarks.  Every qubit receives
  exactly one noise channel per physical time step: ``DEPOLARIZE2`` after a
  two-qubit gate, ``DEPOLARIZE1`` after a one-qubit gate or for each layer a
  qubit idles through, ``X_ERROR`` before a measurement and after a reset.
  The channels are :class:`~qrisp.misc.stim_tools.StimNoiseGate` operations,
  which act as identities for the Qrisp simulator and become genuine noisy
  channels under :meth:`~qrisp.QuantumCircuit.to_stim`.

  The pass inserts only — the instruction order of the input circuit, and
  with it the Stim measurement record, is preserved.  Instructions that carry
  no physical time step (``qb_alloc`` / ``qb_dealloc``, the global phase
  ``gphase``, and the ``parity`` annotation that becomes a Stim ``DETECTOR``)
  neither receive noise nor create a noise layer.  With the default
  ``only_necessary=True`` a channel the circuit already carries is respected
  rather than duplicated, where "already carries" means an exactly matching
  channel type on exactly the matching qubits, directly next to the instruction
  it annotates on the thread of each of those qubits; its error probability may
  differ, and an instruction on an unrelated qubit in between does not break the
  match.  Each channel annotates at most one instruction, so
  ``reset``/``X_ERROR``/``measure`` gives the hand-placed channel to the ``reset``
  it shares a time step with and amends the measurement.  Gates on more than two
  qubits are rejected with a ``ValueError`` before any output is produced.

  The pass shares its scheduler with :func:`~qrisp.layerize`, so the layers it
  inserts noise for are the layers ``layerize`` later draws.  Running

  .. code-block::

      insert_stim_noise -> layerize(insert_barriers=True) -> to_stim

  therefore renders one column and one ``TICK`` per time step with each qubit's
  single channel in it, which is how an inserted model is checked.  Barriers are
  honoured at their declared width: a full-width barrier is a global time
  boundary and flushes the outstanding idle noise of the whole register, while a
  partial barrier only fences the qubits it names — use
  :func:`~qrisp.promote_barriers` first if global boundaries are wanted, at the
  price of a wider schedule and a larger idle-noise budget.

Improvements
------------

- Updated docstrings for ``sample()``, ``expectation_value()``, and
  ``terminal_sampling()`` to use "sampling kernel" terminology and document
  the new arbitrary-return-value capability.

- **A Stim** ``TICK`` **is now emitted only for full-width barriers.**  A
  barrier constrains the qubits it names, while a ``TICK`` is a global
  time-step boundary; the two coincide exactly for a full-width barrier.  A
  barrier over part of the register is a local fence and no longer produces a
  ``TICK`` — use :func:`~qrisp.promote_barriers` to widen one.  These
  ``TICK``\ s are purely presentational: :func:`~qrisp.find_detectors` strips
  every incoming ``TICK`` and regenerates the moment structure it needs.

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