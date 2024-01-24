Algorithms
==========

This algorithms submodule of Qrisp provides a collection of commonly used quantum algorithms that can be used to solve a variety of computational problems. Each algorithm comes with comprehensive documentation and brief examples to help you understand its implementation and usage:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ALGORITHM
     - USED FOR
   * - :ref:`Quantum Fourier Transform <QFT>`
     - periodicity detection and phase estimation
   * - :ref:`Quantum Phase Estimation <QPE>`
     - estimating the eigenvalues of a unitary operator
   * - :ref:`Quantum Amplitude Amplification <AA>`
     - enhancing amplitude of a target state
   * - :ref:`Quantum Amplitude Estimation <QAE>`
     - estimating the amplitude of a target state
   * - :ref:`QAOA <QAOA>`
     - solving combinatorial optimizatin problems
   * - :ref:`Shor's Algorithm <Shor>`
     - efficiently factoring large numbers
   * - :ref:`Grover's Algorithm <Grover>`
     - unstructured search
   * - :ref:`Quantum Backtracking Algorithms <QuantumBacktrackingTree>`
     - solving constraint-satisfaction problems like 3-SAT or the :ref:`Traveling Salesman Problem (TSP) <tsp>`
   * - :ref:`Quantum Counting <QCounting>`
     - estimating the amount of solutions for a given Grover oracle
   * - Iterable :ref:`Demuxing <ItDemuxing>`, :ref:`Shifting <ItShifting>`, and :ref:`Permutation <ItPermutation>`
     - low-level manipulations of quantum arguments like :ref:`QuantumVariable <QuantumVariable>` or :ref:`QuantumArray <QuantumArray>`


We encourage you to explore these algorithms, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   QFT
   QPE
   amplitude_amplification
   QAE
   QAOA
   Shor
   grover.grovers_alg  
   QuantumBacktrackingTree
   quantum_counting
   demux
   cyclic_shift
   iterable_permutation
   
