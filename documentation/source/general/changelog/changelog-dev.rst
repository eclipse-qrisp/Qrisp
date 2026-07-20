.. _v0.10:

Qrisp 0.10
==========

Qrisp 0.10 continues to push the boundaries of high-level quantum programming.
This release brings a streamlined contribution experience with automated
changelog enforcement.

Other New Features
------------------

* Added :func:`thapliyal_adder <qrisp.thapliyal_adder>`, exposing the Thapliyal
  in-place adder from `arXiv:1712.02630 <https://arxiv.org/abs/1712.02630>`_ with
  a modern, Jasp-compatible signature mirroring
  :func:`cuccaro_adder <qrisp.cuccaro_adder>`. It works in both static and dynamic
  (JASP) modes, accepts a classical or quantum first operand, handles unequal
  register sizes (only the first operand is resized, giving modulo addition when
  it is truncated), and supports ``c_in``, ``c_out``, and a
  :func:`custom_control <qrisp.custom_control>`-based controlled version
  (`issue #395 <https://github.com/eclipse-qrisp/Qrisp/issues/395>`_).

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
