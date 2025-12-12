Primitives
==========

This submodule of Qrisp provides a collection of commonly used buildings blocks to build larger algorithms. Each function comes with comprehensive documentation and brief examples to help you understand its implementation and usage:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - ALGORITHM
     - USED FOR
   * - :ref:`Gate functions <gate_application_functions>`
     - application of (elementary) quantum gates
   * - :ref:`Prefix arithmetic <prefix_arithmetic>`
     - several arithmetic functions that allow better control over precision and output types than the infix version
   * - :ref:`Quantum Fourier Transform <QFT>`
     - periodicity detection and phase estimation
   * - :ref:`Quantum Phase Estimation <QPE>`
     - estimating the eigenvalues of a unitary operator
   * - :ref:`Reflection <reflection>`
     - reflection around a quantum state
   * - :ref:`Quantum Amplitude Amplification <AA>`
     - enhancing amplitude of a target state
   * - :ref:`Quantum Amplitude Estimation <QAE>`
     - estimating the amplitude of a target state
   * - :ref:`Iterative Quantum Amplitude Estimation <IQAE>`
     - resource efficient quantum amplitude estimation
   * - :ref:`Iterative Quantum Phase Estimation <IQPE>`
     - resource efficient quantum phase estimation
   * - :ref:`Quantum State Preparation <prepare>`
     - prepare a quantum state with given amplitudes
   * - :ref:`Dicke state preparation <DickeStates>`
     - preparation of Dicke states, i.e., states with a given Hamming weight 
   * - :ref:`Quantum Switch Case <qswitch>`
     - Executes a `switch statement <https://en.wikipedia.org/wiki/Switch_statement>`_. The condition can be a :ref:`QuantumVariable`.
   * - :ref:`Linear Combination of Unitaries <LCU>`
     - implements the prepare-select-unprepare structure (block encoding) for linear combinations of unitaries; used in Hamiltonian simulation (QSP, QSVT, LCHS)  
   * - :ref:`Grover tools <grover_tools>`
     - unstructured search
   * - :ref:`phase_polynomials` 
     - provides functions for applying diagonal Hamiltonians given by polynomials
   * - Iterable :ref:`Demuxing <ItDemuxing>`, :ref:`Shifting <ItShifting>`, and :ref:`Permutation <ItPermutation>`
     - low-level manipulations of quantum arguments like :ref:`QuantumVariable <QuantumVariable>` or :ref:`QuantumArray <QuantumArray>`


We encourage you to explore these algorithms, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   Gate functions
   Prefix arithmetic
   QFT
   QPE
   reflection
   amplitude_amplification
   QAE
   IQAE
   IQPE
   prepare
   qswitch
   LCU
   DickeStates
   Grover tools
   Phase polynomial tools
   demux
   cyclic_shift
   iterable_permutation
