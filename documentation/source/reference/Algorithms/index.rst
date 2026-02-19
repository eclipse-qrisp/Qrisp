Algorithms
==========

This submodule of Qrisp features a curated collection of quantum algorithms designed for a wide range of applications. Each algorithm comes with comprehensive documentation and brief examples to help you understand its implementation and usage.


Foundational Algorithms
-----------------------

Established quantum algorithms, such as Grover's and Shor's, which build the basis for quantum speedups in search and cryptography.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ALGORITHM
     - USED FOR
   * - :ref:`Grover's Algorithm <grovers_alg>`
     - Finding solutions to unstructured search problems.
   * - :ref:`Quantum Counting <QCounting>`
     - Estimating the number of solutions for a given Grover oracle.
   * - :ref:`Quantum Monte Carlo Integration <QMCI>`
     - Numerical integration using amplitude estimation.
   * - :ref:`Shor's Algorithm <Shor>`
     - Efficiently factoring large numbers.


Combinatorial Optimization
--------------------------

Algorithms designed to solve complex routing, scheduling, and resource allocation problems common in logistics and supply chain management.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ALGORITHM
     - USED FOR
   * - :ref:`QAOA <QAOA>`
     - Solving combinatorial optimization problems.
   * - :ref:`QIRO <QIRO>`
     - Solving combinatorial optimization problems, with quantum informed update rules.
   * - :ref:`COLD <DCQO>`
     - Solving QUBO optimization problems with counterdiabatic driving.
   * - :ref:`Quantum Backtracking Algorithms <QuantumBacktrackingTree>`
     - Solving constraint-satisfaction problems like 3-SAT or the :ref:`Traveling Salesman Problem (TSP) <EfficientTSP>`.


Chemistry, Physics and Scientific Computing
-------------------------------------------

Algorithms focused on simulating quantum systems to discover new materials, optimize chemical reactions, and explore fundamental physics.
Beyond simulation, this section includes advanced solvers for linear systems of equations, providing an exponential leap in performance for scientific computing, fluid dynamics, and complex engineering simulations.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ALGORITHM
     - USED FOR
   * - :ref:`VQE <VQE>`
     - Finding the ground state energy of a Hamiltonian.
   * - :ref:`QITE <QITE>`
     - Performing quantum imaginary-time evolution for ground state preparation.
   * - :ref:`Lanczos Algorithm <lanczos_alg>`
     - Finding the ground state energy of a Hamiltonian.
   * - :ref:`CKS Algorithm <CKS>`
     - Solving quantum linear systems problems.
   * - :ref:`Generalized Quantum Signal Processing <GQSP>`
     - Performing quantum signal processing for eigenstate filtering, Hamiltonian simulation, or solving quantum linear systems problems.


We encourage you to explore these algorithms, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:

   Grover
   quantum_counting   
   QMCI
   Shor

   qaoa/QAOA
   qiro/QIRO
   dcqo/DCQO
   QuantumBacktrackingTree

   vqe/VQE
   QITE
   Lanczos
   CKS
   GQSP
