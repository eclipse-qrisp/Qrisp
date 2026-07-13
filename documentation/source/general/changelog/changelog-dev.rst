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
