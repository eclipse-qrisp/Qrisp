.. _v0.10:

Qrisp 0.10
==========

Qrisp 0.10 continues to push the boundaries of high-level quantum programming.
The headline feature is a new **CUDA-Q interface**, letting Qrisp kernels compile
and run directly on NVIDIA's `CUDA-Q <https://nvidia.github.io/cuda-quantum/>`_
platform. This release also brings a streamlined contribution experience with
automated changelog enforcement.

CUDA-Q Interface
-----------------

Qrisp code can now be compiled and executed on `CUDA-Q <https://nvidia.github.io/cuda-quantum/>`_,
NVIDIA's platform for hybrid quantum-classical computing
(`PR #549 <https://github.com/eclipse-qrisp/Qrisp/pull/549>`_). A new lowering
pipeline translates Jaspr programs into CUDA-Q's Quake MLIR dialect via
`xDSL <https://xdsl.dev/>`_, so that Jasp-traceable Qrisp functions can be
run as a native CUDA-Q kernel.

* The :func:`cudaq_kernel <qrisp.jasp.cudaq_interface.cudaq_kernel>` decorator
  turns a Qrisp function into a CUDA-Q kernel, executable with ``cudaq.run``
  or, via ``execution_mode="sample"``, with ``cudaq.sample``.
* Hybrid control flow lowers natively to Quake, including mid-circuit
  measurement with feed-forward, ``q_while_loop``, ``q_cond``, and
  ``q_switch``, allowing measurement outcomes to influence subsequent
  quantum operations at runtime.
* Kernels support scalar (``int``, ``float``, ``bool``) and array
  parameters via :class:`FixedShapeNDArray <qrisp.jasp.cudaq_interface.FixedShapeNDArray>`,
  as well as multiple return values, enabling runtime-configurable circuits
  without recompilation.
* Existing Qrisp functionality - including :ref:`QuantumFloat <QuantumFloat>`
  arithmetic, algorithm primitives, and block encodings - carries over to the CUDA-Q backend.

See the new :doc:`CUDA-Q tutorial </general/tutorial/CUDAQ>` for a hands-on
introduction, from a first Bell-state kernel to hybrid variational workflows.

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

* :doc:`CUDA-Q tutorial </general/tutorial/CUDAQ>` - Compiling and running
  Qrisp kernels on NVIDIA's CUDA-Q platform, from a Bell-state example to
  hybrid quantum-classical workflows
  (`PR #549 <https://github.com/eclipse-qrisp/Qrisp/pull/549>`_).

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
