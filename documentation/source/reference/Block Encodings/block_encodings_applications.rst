.. _block_encodigns_applications:

Algorithms & Applications
=========================

.. currentmodule:: qrisp.block_encodings

**A Unified Algorithmic Framework**

By utilizing block-encodings as a foundation, this abstraction enables the realization of nearly all fundamental tasks in quantum computation [Martyn2021]_,
through (Generalized) Quantum Signal Processing (QSP) [Motlagh2024]_ and its derivatives Quantum Eigenvalue Transform (QET) and Quantum Singular Value Transformation (QSVT) [Gilyén2019]_, [Sünderhauf2023]_.

- **Matrix Inversion:** Solves the Quantum Linear Systems Problem (QLSP) by applying a polynomial transformation approximating :math:`1/x` to the eigenvalues of an encoded matrix [Childs2017]_.

- **Ground State Preparation:** Efficiently prepares eigenstates using eigenstate filtering -- applying a polynomial that acts as a band-pass filter on the spectrum of the Hamiltonian.

- **Hamiltonian Simulation:** Implements time-evolution :math:`e^{-iHt}` by approximating the exponential function with a low-degree polynomial [Low2019]_.


.. autosummary::
   :toctree: generated/
   
   BlockEncoding.inv
   BlockEncoding.poly
   BlockEncoding.sim


**References**

To understand the theoretical framework underpinning these algorithms, refer to the following seminal works:

.. [Childs2017] Childs et al. (2017). `Quantum algorithm for systems of linear equations with exponentially improved dependence on precision <https://arxiv.org/abs/1511.02306>`_
.. [Gilyén2019] Gilyén et al. (2019). `Quantum singular value transformation and beyond <https://dl.acm.org/doi/10.1145/3313276.3316366>`_.
.. [Low2019] Low & Chuang (2019). `Hamiltonian Simulation by Qubitization <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.
.. [Martyn2021] Martyn et al. (2021). `Grand Unification of Quantum Algorithms <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`_.
.. [Motlagh2024] Motlagh & Wiebe (2024). `Generalized Quantum Signal Processing <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.020368>`_
.. [Sünderhauf2023] Sünderhauf (2023). `Generalized Quantum Singular Value Transformation <https://arxiv.org/pdf/2312.00723>`_.

