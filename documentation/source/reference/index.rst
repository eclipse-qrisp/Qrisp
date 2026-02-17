Reference
---------

Introduction
------------

Quantum computing promises strong speedups for problems across optimization, chemistry, and machine learning - but actually *programming* a quantum computer remains far harder than it should be. Most quantum frameworks still operate at the level of individual qubits and gates: the quantum equivalent of writing assembly. The result is code that is tedious to write, difficult to maintain, and nearly impossible to scale.

Qrisp is a high-level quantum programming framework, written in pure Python, that closes this gap. It lets developers express quantum algorithms using typed variables, infix arithmetic, automatic memory management, and classical control flow - concepts familiar from everyday software engineering - while compiling everything down to executable quantum circuits. With its Jax-based compilation backend (Jasp), Qrisp scales to problem sizes that are entirely out of reach for interpreter-level circuit construction, and natively supports real-time classical computation within quantum programs.

Qrisp targets three goals simultaneously:

* **Expressiveness**: A high-level programming interface with typed quantum variables, automatic uncomputation, and environment-based control flow that maps naturally to the way algorithms are described on paper.
* **Performance**: Compilation to efficient circuits through qubit recycling, gate optimization, and - via Jasp - lowering to LLVM-backed infrastructure that can handle thousands of qubits.
* **Portability**: Output as standard circuit objects that run on physical backends from IBM, IQM, AQT, Rigetti, and others, as well as a wide range of simulators.

Since Qrisp programs are Python, developers have direct access to the entire scientific Python ecosystem - NumPy, SciPy, Jax, and beyond - without any language boundary.


Framework Overview
------------------

QuantumVariables
================

The :doc:`Core/QuantumVariable` is the central building block. It represents a quantum register - a named collection of qubits - together with encoding and decoding logic that maps between bitstrings and human-readable values. Gate operations are applied directly to variables or individual qubits within them:

::

    from qrisp import QuantumVariable, h, cx

    qv = QuantumVariable(3)
    h(qv[0])
    cx(qv[0], qv[1])
    cx(qv[1], qv[2])
    print(qv)
    # Outcome: {'000': 0.5, '111': 0.5}

Under the hood, each QuantumVariable is registered in a :doc:`Core/QuantumSession` that manages its lifecycle - qubit allocation, entanglement tracking, compilation, and deallocation. A system of automatic :doc:`Core/Session Merging` ensures that variables from different sessions are transparently unified when they interact, so in most cases the user never has to think about sessions at all.


Quantum Types
=============

Raw QuantumVariable instances are general-purpose but lack domain-specific semantics. Qrisp provides a :doc:`type system <Quantum Types/index>` built on Python's class inheritance: any subclass of QuantumVariable that defines a custom ``decoder`` becomes a new quantum data type. Several types are built in:

* :doc:`Quantum Types/QuantumFloat` - Represents (fractional) numbers with configurable mantissa size, exponent, and sign. Supports full infix arithmetic (``+``, ``-``, ``*``, ``/``, comparisons).
* :doc:`Quantum Types/QuantumBool` - A single-qubit boolean, often produced as the result of comparisons between quantum values.
* :doc:`Quantum Types/QuantumModulus` - Integers mod :math:`N`, used in modular arithmetic for algorithms like Shor's.
* :doc:`Quantum Types/QuantumChar` / :doc:`Quantum Types/QuantumString` - Character and string types for symbolic computation.

Creating a custom quantum type is straightforward - inherit from QuantumVariable and override the ``decoder`` method:

::

    import datetime
    from qrisp import QuantumVariable

    class QuantumDate(QuantumVariable):
        def __init__(self, size, starting_date):
            self.starting_date = starting_date
            QuantumVariable.__init__(self, size)

        def decoder(self, i):
            return self.starting_date + datetime.timedelta(i)

Quantum typing integrates with Python's type hints, enabling self-documenting function signatures and type-based dispatch in the Jasp compilation pipeline.


QuantumArrays
=============

Multiple QuantumVariable instances of the same type can be organized into a :doc:`Core/QuantumArray` - an N-dimensional array that inherits from NumPy's ``ndarray``. This brings familiar operations like slicing, reshaping, transposing, and concatenation to quantum data, all operating as zero-cost views (no extra qubits allocated). For arrays of type QuantumFloat, matrix multiplication and tensor contraction are available.


Quantum Environments
====================

:doc:`Quantum environments <Quantum Environments/index>` are context managers that modify how enclosed operations are compiled. They provide structured control flow without manual gate-level bookkeeping. Some of the most important environments:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Environment
     - Purpose
   * - :doc:`ControlEnvironment <Quantum Environments/ControlEnvironment>`
     - Adds quantum or classical control qubits to all enclosed operations
   * - :doc:`InversionEnvironment <Quantum Environments/InversionEnvironment>`
     - Reverses (daggers) the enclosed operations
   * - :doc:`ConjugationEnvironment <Quantum Environments/ConjugationEnvironment>`
     - Applies :math:`U \cdot \mathrm{body} \cdot U^\dagger` patterns

Environments compose naturally. A controlled operation inside another ControlEnvironment is handled by Qrisp's compiler intelligently - it does not naively synthesize a double-controlled version of every gate but instead leverages the structure for more efficient circuits.

::

    from qrisp import QuantumFloat, control, QFT, h

    def QPE(psi, U, precision):
        res = QuantumFloat(precision, -precision)
        h(res)
        for i in range(precision):
            with control(res[i]):
                for _ in range(2**i):
                    U(psi)
        QFT(res, inv=True)
        return res


Arithmetic
==========

Qrisp supports infix arithmetic on quantum types. Expressions like ``qf_a + qf_b``, ``qf * qf``, or ``qf == 0.25`` are compiled into efficient reversible circuits, with intermediate ancilla qubits automatically managed and recycled:

::

    from qrisp import QuantumFloat, auto_uncompute, z

    @auto_uncompute
    def sqrt_oracle(qf):
        is_quarter = (qf * qf == 0.25)
        z(is_quarter)

The ``auto_uncompute`` decorator ensures that all temporary quantum variables created during the computation (the multiplication result, the comparison result) are properly reversed and their qubits freed - a critical requirement for algorithms like Grover's where the search register must be disentangled before the diffuser acts.

For cases requiring fine-grained control over precision and output types, the :doc:`prefix arithmetic <Primitives/Prefix arithmetic>` module offers explicit functions as an alternative to infix operators.


Memory Management
=================

Quantum resources are scarce and cannot be freed by simply discarding a reference - measurement would collapse entangled partners. Qrisp provides two mechanisms:

* **QuantumVariable.delete** - Deallocates the qubits of a variable, freeing them for reuse. The variable must be in the :math:`|0\rangle` state; with ``verify=True``, Qrisp queries a simulator to confirm disentanglement before proceeding.
* **QuantumVariable.uncompute** - Reverses the computation that created the variable (returning its qubits to :math:`|0\rangle`) and then calls ``delete``. This implements the `Unqomp algorithm <https://github.com/eth-sri/Unqomp>`_, which avoids redundant "un-uncomputation" steps for better gate efficiency. Phase-tolerant gates (e.g., the Margolus gate, requiring only 3 CNOTs instead of 6 for Toffoli) are automatically substituted where their extra phases cancel upon inversion.

The ``auto_uncompute`` decorator applies this logic to all local variables of a function, and the ``recompute`` option enables a qubit-vs-gate tradeoff: variables can be deallocated early and recomputed later if needed, reducing peak qubit count at the cost of additional gates.

For a detailed discussion, see :doc:`Core/Uncomputation`.


Operators Module
================

The :doc:`Operators <Operators/index>` submodule provides a unified framework for describing, optimizing, and simulating quantum Hamiltonians:

* :doc:`Operators/QubitOperator` - Hamiltonians expressed as weighted sums of Pauli strings.
* :doc:`Operators/FermionicOperator` - Hamiltonians using fermionic creation and annihilation operators, with automatic Jordan-Wigner transformation.

These operators serve as inputs to variational algorithms (VQE, QAOA), Hamiltonian simulation, quantum phase estimation, and the block-encoding layer. Example applications span the Heisenberg model, electronic structure calculations, molecular potential energy curves, and the transverse-field Ising model.


Block Encodings
===============

The :doc:`BlockEncoding <Block Encodings/BlockEncoding>` class is the abstraction layer for quantum linear algebra. It embeds a (possibly non-unitary) matrix :math:`A` into a larger unitary as a subblock:

.. math::

    \begin{pmatrix} A/\alpha & \cdot \\ \cdot & \cdot \end{pmatrix}

Three besides the basic signature, three additional constructors cover different use cases:

* ``BlockEncoding.from_array(A)`` - from a NumPy array, for rapid prototyping.
* ``BlockEncoding.from_operator(H)`` - from a QubitOperator or FermionicOperator.
* ``BlockEncoding.from_lcu(coeffs, unitaries)`` - from a custom Linear Combination of Unitaries, typically the most resource-efficient option.

Once encoded, operators can be manipulated with a NumPy-like syntax:

::

    # Classical: C = I + A - 2*A^2 + B^(-1)
    C = np.eye(4) + A - 2 * A @ A + np.linalg.inv(B)

    # Qrisp:
    B_C = B_A.poly([1., 1., -2.]) + B_B.inv(epsilon, kappa)

Two powerful paradigms are available:

* **Matrix arithmetic** - Block encodings can be added, multiplied, scaled, and tensor-producted (``kron``), enabling modular construction of composite operators.
* **Spectral transformation** - Polynomial filters and non-linear functions (matrix inversion via ``.inv``, Hamiltonian simulation via ``.sim``, arbitrary polynomials via ``.poly``) are applied to an operator's spectrum through Generalized Quantum Signal Processing (GQSP).

Application to quantum states uses either ``.apply`` (deterministic, post-selection required) or ``.apply_rus`` (repeat-until-success protocol, fully automated), with resource estimation always available via ``.resources``.

For more details, see :doc:`Block Encodings/index`.


Backend Compatibility
=====================

Qrisp circuits are portable by design. Measurement results can be obtained from any supported backend by passing it to ``get_measurement``:

::

    results = qv.get_measurement(backend=my_backend)

Built-in backend classes include:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Backend
     - Description
   * - :doc:`IQMBackend <Backend Interface/index>`
     - IQM Resonance cloud
   * - :doc:`QiskitBackend <Backend Interface/index>`
     - Any Qiskit-compatible backend (Aer, IBM hardware)
   * - :doc:`AQTBackend <Backend Interface/index>`
     - AQT Arnica cloud
   * - :doc:`QiskitRuntimeBackend <Backend Interface/index>`
     - IBM Qiskit Runtime sessions
   * - :doc:`VirtualBackend <Backend Interface/index>`
     - Run arbitrary dispatch code adhering to the Qrisp interface

Qrisp also supports direct export to Qiskit circuit and other circuit formats for seamless interoperability.


Jasp - The Next-Gen Compilation Pipeline
========================================

Standard Python-based circuit construction hits a wall at scale: a 35-bit modular multiplication already takes ~20 seconds to compile, and typical RSA key sizes reach 2000 bits. Furthermore, many quantum algorithms require *real-time classical computation* - classical logic that executes during the quantum program, faster than decoherence - which an interpreted language fundamentally cannot provide.

Jasp solves both problems by making Qrisp code traceable through `Jax <https://jax.readthedocs.io/>`_, Google's framework for high-performance numerical computing. Instead of executing a function with actual values, Jax sends *tracer* objects through it, recording operations into a functional intermediate representation (Jaxpr). Jasp extends this with quantum primitives, producing a ``Jaspr`` - a Jax-compatible IR that captures both quantum operations (gate applications, measurements) and classical real-time computations (arithmetic, control flow, even neural network inference) in a single representation.


Using Jasp
^^^^^^^^^^

For many use cases, adopting Jasp is a single decorator:

::

    from qrisp import *
    from qrisp.jasp import jaspify

    @jaspify
    def main():
        qf = QuantumFloat(4)
        h(qf[0])
        cx(qf[0], qf[1])
        return measure(qf)

    print(main())

The ``terminal_sampling`` option draws samples directly from the statevector instead of repeated single-shot simulation, offering large speedups for variational and sampling-based algorithms:

::

    @jaspify(terminal_sampling=True)
    def qaoa_optimization():
        ...


Compilation Targets
^^^^^^^^^^^^^^^^^^^

Because ``Jaspr`` objects are embedded in the Jax ecosystem, they can be lowered through multiple compilation paths:

* **QIR** (Quantum Intermediate Representation) via Catalyst/MLIR - targeting the LLVM toolchain for optimized hybrid quantum-classical execution.
* **QuantumCircuit** - standard circuit output for any gate-based backend.
* **MLIR** / **Catalyst MLIR** - intermediate stages for advanced optimization passes.


Key Jasp Features
^^^^^^^^^^^^^^^^^

* **qache** - A caching decorator that traces quantum functions only once per calling signature, then reuses the cached Jaspr. Functions called multiple times incur tracing cost only on the first invocation.
* **jrange** - A traceable loop construct that replaces Python's ``range``, enabling dynamic iteration counts in compiled code.
* **Classical control flow** - Measurement results are native integers (not opaque classical bits), so they can drive ``if``/``else`` branching, arithmetic, or any Jax operation in real time.
* **Repeat-until-success (@RUS)** - A decorator for trial-based quantum protocols that loop until a classical measurement condition is met.
* **sample / minimize** - Traceable primitives for quantum sampling and classical optimization, enabling end-to-end compiled hybrid loops (e.g., full QAOA optimization without Python-level overhead).
* **Boolean simulation** - A specialized fast-path for circuits composed entirely of boolean gates (X, CX, MCX). Jasp transforms the Jaspr into pure boolean Jax logic, then leverages XLA compilation for massive throughput - demonstrated at over a million quantum floating-point operations after a single compilation step.


Real-Time Classical Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Jasp's most distinctive capability is embedding arbitrary classical computation into the quantum program timeline. Because traced classical logic compiles to machine code (not interpreted Python), operations like syndrome decoding, neural network classification, or modular arithmetic can execute within coherence windows. The :ref:`Jasp tutorial <jasp>` demonstrates a neural network binary classifier running inside a repeat-until-success loop, where measurement outcomes feed into ``jax.nn.sigmoid`` and the result determines whether to continue or halt - all within a single compiled program.


Algorithms
==========

Qrisp ships with implementations of major quantum algorithms, all built on the abstractions above:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Algorithm
     - Purpose
   * - :ref:`VQE <VQE>`
     - Ground state energy of Hamiltonians
   * - :ref:`QAOA <QAOA>`
     - Combinatorial optimization
   * - :ref:`Lanczos <lanczos_alg>`
     - Ground state energy via Krylov subspace methods
   * - :ref:`CKS <CKS>`
     - Quantum linear systems
   * - :ref:`GQSP <GQSP>`
     - Quantum signal processing (eigenstate filtering, Hamiltonian simulation, linear systems)
   * - :ref:`Quantum Backtracking <QuantumBacktrackingTree>`
     - Constraint-satisfaction problems (3-SAT, TSP)
   * - :ref:`QITE <QITE>`
     - Quantum imaginary-time evolution
   * - :ref:`QIRO <QIRO>`
     - Combinatorial optimization with quantum-informed updates
   * - :ref:`Shor's Algorithm <Shor>`
     - Integer factoring
   * - :ref:`Grover's Algorithm <grovers_alg>`
     - Unstructured search
   * - :ref:`Quantum Counting <QCounting>`
     - Solution counting for Grover oracles
   * - :ref:`QMCI <QMCI>`
     - Quantum Monte Carlo integration

These implementations benefit directly from Qrisp's compilation infrastructure: Shor's algorithm, for example, produces circuits with significantly reduced resource requirements compared to manual circuit-level implementations, demonstrating that systematic high-level development yields quantitative advantages.


.. toctree::
   :hidden:
   :maxdepth: 2
   
   Core/index
   Quantum Types/index
   Quantum Environments/index
   Primitives/index
   Algorithms/index
   Operators/index
   Block Encodings/index
   Jasp/index
   Circuit Manipulation/index
   Backend Interface/index
   Utilities
   Examples/index
