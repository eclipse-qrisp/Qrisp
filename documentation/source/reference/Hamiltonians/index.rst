Hamiltonians
============

.. note::

    This module is still under heavy development and the interface can therefore change at any time!

This Hamiltonians submodule of Qrisp provides a unified framework to describe, optimize and simulate quantum Hamiltonians.
It provides a collection of different types of Hamiltonians that can be used to model and solve a variety of problems in physics, chemistry, or optimization.
(Up to now, Pauli Hamiltonians and Fermionic Hamiltonians have been implemented, but stay tuned for future updates!)
Each type of Hamiltonian comes with comprehensive documentation and brief examples to help you understand its implementation and usage:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Hamiltonian
     - USED FOR
   * - :ref:`QubitOperator <QubitOperator>`
     - describe Hamiltonians in terms of Qubit operators
   * - :ref:`FermionicHamiltonian <FermionicHamiltonian>`
     - describe Hamiltonians in terms of fermionic ladder operators

We encourage you to explore these Hamiltonians, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :hidden:
   
   QubitOperator
   FermionicHamiltonian


Examples
========

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Title
     - Description
   * - :ref:`VQEHeisenberg` 
     - Investigating the antiferromagnetic Heisenberg model with :ref:`VQE <VQE>`.
   * - :ref:`VQEElectronicStructure` 
     - Solving the electronic structure problem with :ref:`VQE <VQE>`.
   * - :ref:`MolecularPotentialEnergyCurve` 
     - Calculating molecular potential energy curves with :ref:`VQE <VQE>`.
   * - :ref:`GroundStateEnergyQPE` 
     - Calculating ground state energies with :ref:`quantum phase estimation <QPE>`.                    
   * - :ref:`IsingModel` 
     - Hamiltonian simulation of the transverse field Ising model.

