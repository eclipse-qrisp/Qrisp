.. _WritingTests:

Writing Tests
=============

Qrisp uses `pytest <https://docs.pytest.org>`_ for its test suite. Every piece
of public, user-facing functionality should be covered by at least one test.

Test layout
-----------

Test files mirror the structure of the source tree:

.. code-block:: text

    tests/
    ├── algorithms_tests/
    ├── block_encodings_tests/
    ├── circuit_tests/
    ├── core_tests/
    ├── interface_tests/
    ├── jax_tests/
    ├── operators_tests/
    └── primitives_tests/

Place new test files in the subsystem directory that corresponds to the source
module you are testing. For example:

.. code-block:: text

    src/qrisp/circuit/quantum_circuit.py
      → tests/circuit_tests/test_quantum_circuit.py

If a test file for that module does not exist, create one following this
convention.

Writing a test
--------------

Each test file must be named ``test_<feature>.py`` and each test function must
start with ``test_``. Tests should be:

- *Small and focused*: avoid large tests that validate many behaviours at
  once.
- *Deterministic*: avoid randomness unless the feature itself is
  probabilistic.
- *Explicit*: the test name and body should make it clear exactly which
  behaviour is being validated.

We refer to the official pytest documentation for more detailed guidance 
on writing tests (see the ``Useful references`` section below).

Running the tests
-----------------

.. code-block:: bash

    # Run the full suite
    pytest tests/

    # Run only a specific subsystem
    pytest tests/circuit_tests/

    # Run a single file
    pytest tests/circuit_tests/test_quantum_circuit.py

    # Run with verbose output
    pytest -v tests/circuit_tests/test_quantum_circuit.py

Checking test coverage
----------------------

To verify that the code you added is actually exercised by your tests, you can
use `coverage <https://coverage.readthedocs.io>`_:

.. code-block:: bash

    pip install coverage
    coverage run -m pytest tests/circuit_tests/test_quantum_circuit.py
    coverage html

This is a useful sanity check before opening a pull request.

A good starting point when looking for what to test is the docstring examples
of the function or class you are working on. If those examples exist, they
should all pass as tests. If they do not exist, adding them is a contribution
in itself (see :ref:`DevGuideDocumentation`).

Useful references
-----------------

- pytest documentation: https://docs.pytest.org/
- Parametrized tests: https://docs.pytest.org/en/stable/how-to/parametrize.html
- Fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
