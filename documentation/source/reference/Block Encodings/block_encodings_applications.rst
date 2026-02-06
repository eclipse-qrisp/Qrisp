.. _block_encodigns_applications:

Algorithms & Applications
=========================

.. currentmodule:: qrisp.block_encodings

**Unified Algorithmic Framework**

By utilizing block encodings as a foundation, this abstraction enables the realization of nearly all fundamental tasks in quantum computation through Quantum Signal Processing (QSP) and Quantum Singular Value Transformation (QSVT) [Gilyén2019]_, [Martyn2021]_, [Low2019]_.

- **Matrix Inversion:** Solves the Quantum Linear Systems Problem (QLSP) by applying a :math:`1/x` polynomial transformation to the singular values of an encoded matrix.

- **Ground State Preparation:** Efficiently prepares eigenstates using eigenstate filtering—applying a polynomial that acts as a band-pass filter on the spectrum of the Hamiltonian.

- **Hamiltonian Simulation:** Implements time-evolution :math:`e^{-iHt}` by approximating the exponential function with a low-degree polynomial.


.. autosummary::
   :toctree: generated/
   
   BlockEncoding.inv
   BlockEncoding.poly
   BlockEncoding.sim


**References**

To understand the theoretical framework underpinning this module, refer to the following seminal works:

.. [Gilyén2019] Gilyén et al. (2019). `Quantum singular value transformation and beyond <https://dl.acm.org/doi/10.1145/3313276.3316366>`_.
.. [Martyn2021] Martyn et al. (2021). `Grand Unification of Quantum Algorithms <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`_.
.. [Low2019] Low & Chuang (2019). `Hamiltonian Simulation by Qubitization <https://quantum-journal.org/papers/q-2019-07-12-163/>`_.



