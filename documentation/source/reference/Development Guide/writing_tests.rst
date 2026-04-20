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

The following example is taken from
``tests/circuit_tests/test_quantum_circuit.py``. It groups related tests in a
class, documents each test with a one-line docstring, and covers both the
happy path and an error case:

.. code-block:: python

    import pytest
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    class TestQuantumCircuitInitialization:
        """Tests for QuantumCircuit initialization."""

        def test_initialization(self):
            """Test basic initialization with qubit and clbit counts."""
            qc = QuantumCircuit(num_qubits=3, num_clbits=2)
            assert len(qc.qubits) == 3
            assert len(qc.clbits) == 2

        def test_error_wrong_type_initialization(self):
            """Test that wrong argument types raise TypeError."""
            with pytest.raises(TypeError, match="type str for num_qubits"):
                QuantumCircuit(num_qubits="3")

            with pytest.raises(TypeError, match="type float for num_clbits"):
                QuantumCircuit(num_clbits=2.5)

Parametrised tests
------------------

Use ``@pytest.mark.parametrize`` to test multiple inputs without duplicating
code. The following example is also from
``tests/circuit_tests/test_quantum_circuit.py`` and checks that the unitary of
an empty circuit is the identity matrix for several qubit counts:

.. code-block:: python

    import pytest
    import numpy as np
    from qrisp.circuit.quantum_circuit import QuantumCircuit

    class TestQuantumCircuitMethods:
        """Tests for QuantumCircuit methods."""

        @pytest.mark.parametrize("num_qubits", [1, 2, 3])
        def test_get_unitary_empty_circuit_is_identity(self, num_qubits):
            """Empty circuit of n qubits has the 2ⁿ × 2ⁿ identity as its unitary."""
            dim = 2**num_qubits
            assert np.allclose(QuantumCircuit(num_qubits).get_unitary(), np.eye(dim))

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
