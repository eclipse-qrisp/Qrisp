Development Guide
=================

Welcome, and thank you for your interest in contributing to Qrisp!

This guide collects the engineering standards and conventions that help keep
the codebase maintainable and welcoming to new contributors. It is intended
to complement (not replace) the existing project documentation and GitHub
issue tracker.

The goal is to enable incremental improvements to code quality, test coverage,
and documentation alongside ongoing feature development. You do not need to
work through every section in order as each page stands on its own.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Topic
     - Description
   * - :ref:`DevGuideGettingStarted`
     - Set up a local development environment and verify the baseline test suite.
   * - :ref:`DevGuideCodeQuality`
     - Static analysis, code style, and type annotations.
   * - :ref:`WritingTests`
     - How to write and structure tests using pytest.
   * - :ref:`DevGuideDocumentation`
     - Building the docs locally and writing correct docstrings.
   * - :ref:`DevGuideIssuesPullRequests`
     - Opening issues and pull requests, asking for help, and common mistakes to avoid.
   * - :ref:`DevGuideWriteaTutorial`
     - Writing a tutorial or example for your implementation.
   * - :ref:`DevGuideFirstIssues`
     - First issues to tackle and feature requests.
   * - :ref:`DevGuideReleasing`
     - How to cut a Qrisp release — version bump, tagging, building, and publishing.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   code_quality
   writing_tests
   documentation
   issues_pull_requests
   write_a_tutorial
   issues_to_tackle
   releasing
