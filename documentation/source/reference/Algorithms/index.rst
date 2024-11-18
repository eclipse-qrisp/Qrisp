Algorithms
==========

This algorithms submodule of Qrisp provides a collection of commonly used quantum algorithms that can be used to solve a variety of computational problems. Each algorithm comes with comprehensive documentation and brief examples to help you understand its implementation and usage:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ALGORITHM
     - USED FOR
   * - :ref:`VQE <VQE>`
     - finding the ground state energy of a Hamiltonian
   * - :ref:`QAOA <QAOA>`
     - solving combinatorial optimization problems
   * - :ref:`QIRO <QIRO>`
     - solving combinatorial optimization problems, with quantum informed update rules
   * - :ref:`Shor's Algorithm <Shor>`
     - efficiently factoring large numbers
   * - :ref:`Grover's Algorithm <grovers_alg>`
     - unstructured search
   * - :ref:`Quantum Backtracking Algorithms <QuantumBacktrackingTree>`
     - solving constraint-satisfaction problems like 3-SAT or the :ref:`Traveling Salesman Problem (TSP) <tsp>`
   * - :ref:`Quantum Counting <QCounting>`
     - estimating the amount of solutions for a given Grover oracle


We encourage you to explore these algorithms, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   vqe/VQE
   qaoa/QAOA
   qiro/QIRO
   Shor
   Grover
   QuantumBacktrackingTree
   quantum_counting   
