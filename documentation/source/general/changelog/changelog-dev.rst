.. _v0.10:

Qrisp 0.10
==========

Qrisp 0.10 continues to push the boundaries of high-level quantum programming.
This release brings a streamlined contribution experience with automated
changelog enforcement.

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

* Fixed a bug where :func:`prepare <qrisp.prepare>` with ``method="qswitch"``
  raised a ``ValueError`` when used inside an :func:`invert <qrisp.invert>` or
  :func:`control <qrisp.control>` environment in Jasp mode
  (`Issue #421 <https://github.com/eclipse-qrisp/Qrisp/issues/421>`_).

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

Dependency Upgrades
-------------------

* Bumped myst-parser from 5.0.0 to 5.1.0
  (`PR #729 <https://github.com/eclipse-qrisp/Qrisp/pull/729>`_).

* Bumped ipykernel from 7.2.0 to 7.3.0
  (`PR #734 <https://github.com/eclipse-qrisp/Qrisp/pull/734>`_).

* Bumped pytest from 9.1.0 to 9.1.1
  (`PR #723 <https://github.com/eclipse-qrisp/Qrisp/pull/723>`_).

* Bumped ``actions/setup-python`` from 6 to 7
  (`PR #760 <https://github.com/eclipse-qrisp/Qrisp/pull/760>`_).

.. Add dependency upgrades above this line

First Time Contributors 🎉
--------------------------

* `alighazi288 <https://github.com/alighazi288>`_
* `NedislavKolev <https://github.com/NedislavKolev>`_
* `Shanwis <https://github.com/Shanwis>`_
