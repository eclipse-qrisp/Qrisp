.. _BlockEncodings:

Block Encodings
===============

.. currentmodule:: qrisp.block_encodings

.. toctree::
   :hidden:
   
   BlockEncoding


**A High-Level Abstraction for Quantum Linear Algebra**

The :ref:`BlockEncoding` provides a powerful programming abstraction for Quantum Linear Algebra. 
While traditional quantum programming often focuses on low-level gate operations (in qrisp: higher-level operations using functions and variables), 
this interface allows developers to work directly with non-unitary matrices by embedding them into larger unitary operators.

**Key Concepts**

- **Matrix-Centric Design:** Instead of manually constructing circuits, users define a target matrix :math:`A`. The module automatically handles the construction of a unitary :math:`U` such that :math:`A` is represented in the top-left block (scaled by a factor :math:`\alpha`):

.. math::

    U = \begin{pmatrix} A/\alpha & \cdot\\ \cdot & \cdot \end{pmatrix}

- **Algebraic Composability:** Perform complex operations—such as the addition or multiplication of linear operators — through a modular interface that manages underlying ancilla qubits and normalization factors.

**Unified Algorithmic Framework**

By utilizing block encodings as a foundation, this abstraction enables the realization of nearly all fundamental tasks in quantum computation through Quantum Signal Processing (QSP) and Quantum Singular Value Transformation (QSVT) [Gilyén2019]_, [Martyn2021]_, [Low2019]_.

- **Matrix Inversion:** Solves the Quantum Linear Systems Problem (QLSP) by applying a :math:`1/x` polynomial transformation to the singular values of an encoded matrix.

- **Ground State Preparation:** Efficiently prepares eigenstates using eigenstate filtering—applying a polynomial that acts as a band-pass filter on the spectrum of the Hamiltonian.

- **Hamiltonian Simulation:** Implements time-evolution :math:`e^{-iHt}` by approximating the exponential function with a low-degree polynomial.

**References**

To understand the theoretical framework underpinning this module, refer to the following seminal works:

.. [Gilyén2019] Gilyén et al. (2019). `Quantum singular value transformation and beyond <https://dl.acm.org/doi/10.1145/3313276.3316366>`_.
.. [Martyn2021] Martyn et al. (2021). `Grand Unification of Quantum Algorithms <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`_.
.. [Low2019] Low & Chuang (2019). `Hamiltonian Simulation by Qubitization <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.