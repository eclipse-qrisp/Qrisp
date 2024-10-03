Primitives
==========

This submodule of Qrisp provides a collection of commonly used buildings blocks to build larger algorithms. Each function comes with comprehensive documentation and brief examples to help you understand its implementation and usage:

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
   * - :ref:`phase_polynomials` 
     - provides functions for applying diagonal Hamiltonians given by polynomials.
   * - :ref:`Grover's Algorithm <grover_tools>`
     - unstructured search
   * - :ref:`Dicke state preparation <DickeStates>`
     - algorithm for the preparation of Dicke states, i.e. states with a given Hamming weight. 
   * - Iterable :ref:`Demuxing <ItDemuxing>`, :ref:`Shifting <ItShifting>`, and :ref:`Permutation <ItPermutation>`
     - low-level manipulations of quantum arguments like :ref:`QuantumVariable <QuantumVariable>` or :ref:`QuantumArray <QuantumArray>`
   * - :ref:`Prefix arithmetic <prefix_arithmetic>`
     - Several arithmetic functions that allow better control over precision and output types than the infix version.


We encourage you to explore these algorithms, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :maxdepth: 2
   :hidden:
   
   Gate functions
   Prefix arithmetic
   QFT
   QPE
   amplitude_amplification
   QAE
   Phase polynomial tools
   Grover tools
   DickeStates
   demux
   cyclic_shift
   iterable_permutation
