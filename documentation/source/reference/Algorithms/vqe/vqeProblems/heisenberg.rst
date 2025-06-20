.. _VQEHeisenberg:

VQE Heisenberg model
====================

.. currentmodule:: qrisp.vqe.problems.heisenberg

We proivde an implementation of the Variational Quantum Eigensolver for Heisenberg models following `Bosse and Montanaro <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.105.094409>`_ and `Kattemölle and van Wezel <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.214429>`_.

Heisenberg problem
------------------

.. autofunction:: heisenberg_problem

Hamiltonian
-----------

.. autofunction:: create_heisenberg_hamiltonian

Ansatz
------

.. autofunction:: create_heisenberg_ansatz

.. autofunction:: create_heisenberg_init_function