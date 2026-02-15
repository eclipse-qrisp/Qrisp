.. _block_encodigns_applications:

Algorithms & Applications
=========================

.. currentmodule:: qrisp.block_encodings

Utilizing block-encodings as programming abstraction -- in principle -- enables the realization of a vast array of fundamental tasks in quantum computation [Martyn2021]_.
Through (Generalized) Quantum Signal Processing (QSP) [Motlagh2024]_ and its derivatives Quantum Eigenvalue Transform (QET) and Quantum Singular Value Transformation (QSVT) [Gilyén2019]_, [Sünderhauf2023]_,
diverse algorithms -- ranging from Hamiltonian simulation to linear systems solvers -- can be expressed as polynomial transformations of a block-encoded matrix.

While it serves as a powerful toolbox for designing algorithms with rigorous performance bounds, its practical viability depends heavily on the specific application and the underlying hardware constraints. 
In many scenarios, established alternatives remain highly competitive: for example, Trotter-Suzuki decompositions are often more straightforward for near-term Hamiltonian simulation, 
variational methods may be better suited for noisy hardware, and traditional Quantum Phase Estimation (QPE) remains a standard for Shor's algorithm. 
Ultimately, block-encodings should be viewed as a modular and expressive language that complements, rather than replaces, the existing suite of quantum algorithmic strategies.

Key applications of block-encodings are:

- **Matrix Inversion:** Solves the Quantum Linear Systems Problem (QLSP) by applying a polynomial transformation approximating :math:`1/x` to the eigenvalues of an encoded matrix [Childs2017]_.

- **Ground State Preparation:** Efficiently prepares eigenstates using eigenstate filtering -- applying a polynomial that acts as a band-pass filter on the spectrum of the Hamiltonian.

- **Hamiltonian Simulation:** Implements time-evolution :math:`e^{-iHt}` by approximating the exponential function with a low-degree polynomial [Low2019]_.


.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :func:`~qrisp.block_encodings.BlockEncoding.inv`
     - Returns a BlockEncoding approximating the matrix inversion of the operator.
   * - :func:`~qrisp.block_encodings.BlockEncoding.poly`
     - Returns a BlockEncoding representing a polynomial transformation of the operator.
   * - :func:`~qrisp.block_encodings.BlockEncoding.sim`
     - Returns a BlockEncoding approximating Hamiltonian simulation of the operator.

.. toctree::
   :hidden:

   methods/inv
   methods/poly
   methods/sim


**References**

To understand the theoretical framework underpinning these algorithms, refer to the following seminal works:

.. [Childs2017] Childs et al. (2017). `Quantum algorithm for systems of linear equations with exponentially improved dependence on precision <https://arxiv.org/abs/1511.02306>`_
.. [Gilyén2019] Gilyén et al. (2019). `Quantum singular value transformation and beyond <https://dl.acm.org/doi/10.1145/3313276.3316366>`_.
.. [Low2019] Low & Chuang (2019). `Hamiltonian Simulation by Qubitization <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.
.. [Martyn2021] Martyn et al. (2021). `Grand Unification of Quantum Algorithms <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`_.
.. [Motlagh2024] Motlagh & Wiebe (2024). `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_
.. [Sünderhauf2023] Sünderhauf (2023). `Generalized Quantum Singular Value Transformation <https://arxiv.org/pdf/2312.00723>`_.

