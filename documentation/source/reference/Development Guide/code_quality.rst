.. _DevGuideCodeQuality:

Code Quality
============

This page covers the engineering standards that contributions to Qrisp are
expected to meet: static analysis, code style, and type annotations.

.. note::

    The checks described on this page are not yet enforced automatically by the
    CI pipeline or pre-commit hooks. They are recommended practices that
    contributors are encouraged to follow for the changes they introduce. Automated enforcement will be
    introduced in the future.

Preserving existing behaviour
------------------------------

Unless a breaking change has been explicitly agreed on with the maintainers,
every contribution must preserve all existing user-facing behaviour.

A practical rule of thumb: if a user's existing code would stop working after
your change, it is a breaking change. If you believe a breaking change is
necessary, clearly document it in the pull request description so maintainers
can make an informed decision.

Static analysis
---------------

Use **pylint** or **ruff** to catch common issues before submitting a pull
request.

- Pylint documentation: https://pylint.readthedocs.io/
- Ruff documentation: https://docs.astral.sh/ruff/

Issues worth prioritising include the ones related to code logic, efficiency, and maintainability:

- Mutable default arguments
- Unnecessary ``elif`` or ``return`` statements
- Repeated blocks of code
- Deeply nested functions
- Unused imports or arguments
- Missing docstrings for public functions and classes
- etc.

Some warnings are cosmetic (e.g., ``line-too-long``) and can be deferred.
Others point to real quality issues and should be addressed.

Relevant style guides:

- PEP 8 — Python Style Guide: https://peps.python.org/pep-0008/
- PEP 257 — Docstring Conventions: https://peps.python.org/pep-0257/

Code logic and readability
---------------------------

Small improvements to readability and structure are highly encouraged, even
when they are not strictly bug fixes. A well-placed rename or a function split
into two smaller ones often makes the biggest difference for future
contributors.

When working on core components, avoid changes that significantly degrade
performance: unnecessary allocations, non-vectorised operations where
vectorisation is natural, or tightly looped code that could be simplified.

If a change might affect performance, mention it in the pull request
description.

AI-assisted development tools (such as GitHub Copilot) can help identify
simplifications, but manual review is essential: these tools sometimes change
code logic subtly. Always verify generated output carefully and that all associated unit tests pass.

Type annotations
----------------

Adding type annotations wherever possible improves readability, IDE support,
and static analysis coverage.

Annotating at least the function signature (parameters and return type) is
already a meaningful improvement. You do not need to annotate every local
variable.

Static type checkers will often surface inconsistencies that point to subtle
bugs:

- **Pylance** (VS Code) — inline, zero-configuration
- **mypy** (command line): https://mypy.readthedocs.io/

Useful references:

- Python typing module: https://docs.python.org/3/library/typing.html
- Real Python guide to type checking: https://realpython.com/python-type-checking/
