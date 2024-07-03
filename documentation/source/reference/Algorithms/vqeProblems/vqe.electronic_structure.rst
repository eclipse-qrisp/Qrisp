.. _VQEElectronicStructure:

VQE Electronic structure problem implementation
===============================================

.. currentmodule:: vqe.problems.electronic_structure

Problem description
-------------------

.. math::
    
    H=\sum\limits_{i,j=1}^{M}h_{ij}a_i^{\dagger}a_j+\frac{1}{2}\sum\limits_{i,j,k,l}^{M}h_{i,j,k,l}a_i^{\dagger}a_j^{\dagger}a_ka_l

The coefficients :math:`h_{ij}` and :math:`h_{ijkl}` are one- and two-electron integrals which can be computed classically.

Solving the electronic structure problem on a quantum computer requires transforming the fermionic representation into a qubit representation.
This is achieved by, e.g., the Jordan-Wigner, Parity, or Bravyi-Kitaev transformations.



Cost operator
-------------

todo


Classical cost function
-----------------------

todo