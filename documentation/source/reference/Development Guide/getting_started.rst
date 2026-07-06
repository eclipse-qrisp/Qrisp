.. _DevGuideGettingStarted:

Getting Started
===============

Welcome, and thank you for your interest in contributing to Qrisp!

This page walks you through setting up a local development environment and
verifying that everything works before you make any changes.

Environment setup
-----------------

Qrisp currently requires **Python 3.11 or later**. We recommend working inside a virtual
environment (e.g. ``venv``, ``conda``, or any other tool you prefer).

We recommend installing Qrisp in *editable* mode so that every change you make to the source is
reflected immediately without reinstalling:

.. code-block:: bash

    pip install -e .

For development, you will also want the test and documentation dependencies. Choose the installation method for your operating system below:

**Linux and macOS**

.. code-block:: bash

    pip install -e ".[test,docs]"

**Windows**

.. note::
    The ``test`` dependency group includes ``pyscf``, which is not supported on Windows. Windows users must install the compatible dependencies individually:

.. code-block:: bash

    # Install optional dependency groups (excluding test)
    pip install -e ".[docs,iqm,xdsl,aqt,qiskit]"

    # Install Windows-compatible testing framework tools
    pip install pytest qiskit-aer cirq pytest-httpx stim

    # Install tqecd from source
    python -m pip install git+https://github.com/tqec/tqecd.git

.. list-table:: Optional dependency groups
   :header-rows: 1
   :widths: 15 40

   * - Group
     - What it installs
   * - ``test``
     - Test runner (pytest), simulators (qiskit-aer, cirq), and chemistry (pyscf *(Not supported on Windows)*)
   * - ``docs``
     - Sphinx and related extensions for building the documentation
   * - ``aqt``
     - Client for AQT quantum hardware
   * - ``qiskit``
     - Qiskit Aer simulator and IBM Quantum Runtime hardware
   * - ``iqm``
     - Client for IQM quantum hardware
   * - ``catalyst``
     - PennyLane Catalyst JIT compiler *(Not supported on Windows)*
   * - ``xdsl``
     - xDSL compiler infrastructure

Include the groups you need in brackets, either at install time or later:

.. code-block:: bash

    # All at once (Linux/macOS)
    pip install -e ".[test,docs,iqm,catalyst,xdsl,aqt,qiskit]"

    # Or individual groups after the base install
    pip install -e ".[test]"
    pip install -e ".[iqm]"

.. note::

    The quotes around the bracket expression are required in most shells (``zsh``, ``bash``, etc.)
    to prevent the shell from interpreting the square brackets.

Verify the installation
-----------------------

Confirm that Qrisp can be imported and reports the expected version:

.. code-block:: bash

    python -c "import qrisp; print(qrisp.__version__)"

Running the test suite
----------------------

Before making any changes, confirm that the existing test suite passes on your
machine. This gives you a clean baseline to compare against.

**Linux and macOS**

.. code-block:: bash

    # Run the full suite (note: this currently takes up to ~1 hour)
    pytest tests/

**Windows**

.. note::
    Because certain unsupported dependencies are missing on Windows, you must ignore specific interface and JAX tests when running the suite:

.. code-block:: bash

    # Run the full suite while ignoring unsupported dependency tests
    pytest tests/ --ignore=tests/interface_tests/test_converter_qrisp_to_qml.py --ignore=tests/jax_tests/test_catalyst_interface.py --ignore=tests/jax_tests/test_qubit_array_fusion.py

**Subsystem testing (All Platforms)**

.. code-block:: bash

    # Run only a specific subsystem — much faster during development
    pytest tests/circuit_tests/

    # Run a single file
    pytest tests/circuit_tests/test_quantum_circuit.py

If any tests fail *before* you have made changes, please open a GitHub issue
rather than working around the failure. Keep in mind that some tests are
stochastic (depending on random seeds, a test may fail in one run and pass in
another). If you see an isolated failure that disappears on a second run, try
running the test a few more times before concluding there is a bug.

.. note::

    The test suite is organised by subsystem and mirrors the layout of the
    source tree. See :ref:`WritingTests` for the full directory structure and
    guidance on adding new tests.

Next steps
----------

Once your environment is set up and the baseline tests pass, continue with the
rest of this guide:

- :ref:`DevGuideCodeQuality` — static analysis, style, and type annotations
- :ref:`WritingTests` — how to write and structure tests
- :ref:`DevGuideDocumentation` — building and checking the documentation
- :ref:`DevGuideIssuesPullRequests` — opening issues and pull requests, asking for help, and common mistakes to avoid

Further reading 
---------------

If you are completly new to open source development of scientific software, please give the resources below a read. 

* `pythonpackaging.info <https://pythonpackaging.info/>`_

    A guide covering current best practices for developing, publishing, and maintaining a scientific Python package.

* `Research Software Engineering with Python <https://third-bit.com/py-rse/>`_

    Full workflow of building reliable research software covering topics like Git, testing, automation, and project organization.

* `GitHub Docs – Creating a Pull Request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_

    Proposing changes to a repository by opening a pull request, covering how to create one from a branch, add a description, and submit it for review.
