.. _VQEElectronicStructure:

VQE Electronic structure problem 
================================

.. currentmodule:: vqe.problems.electronic_structure

Problem description
-------------------

Consider the problem of finding approximate solutions to the non-relativistic time-independent Schrödinger equation

.. math::

    H\ket{\Psi}=E\ket{\Psi}

The Hamiltonian $H$ for a system of $M$ nuclei and $N$ electrons with coordinates $R_A\in\mathbb R^3$ and $r_i\in\mathbb R^3$, respectively,
is defined as

.. math::

    H= -\sum_{i=1}^N\frac12\nabla_i^2-\sum_{i=1}^M\frac{1}{2M_A}\nabla_A^2-\sum_{i=1}^N\sum_{A=1}^M\frac{Z_A}{r_{iA}}+\sum_{i=1}^N\sum_{j>i}^N\frac{1}{r_{ij}}+\sum_{A=1}^M\sum_{B>A}^M\frac{Z_AZ_B}{R_{AB}}

where $r_{iA}=|r_i-R_A|$, $r_{ij}=|r_i-r_j|$, and $R_{AB}=|R_A-R_B|$, are the distances between electrons and nuclei, $M_A$ is the ratio of the mass of nucleus $A$ to an electron, 
and $Z_A$ is the atomic number of nulecus $A$. 

Within the Born-Oppenheimer approximation, one considers the electrons in a molecule to be moving a the field of fixed nuclei. 
Therefore, the kinetic energy of the nuclei is neglected, and the nuclear repulsion energy is constant.

The Hamiltonian describing the motion of $N$ electrons in the field of $M$ fixed nuclei is

.. math::

    H_{\text{el}} = -\sum_{i=1}^N\frac12\nabla_i^2-\sum_{i=1}^N\sum_{A=1}^M\frac{Z_A}{r_{iA}}+\sum_{i=1}^N\sum_{j>i}^N\frac{1}{r_{ij}}

The **electronic stucture problem** consists of finding solutions to the Schrödinger equation for the electronic Hamiltonian

.. math::

    H_{\text{el}}\ket{\Psi_{\text{el}}}=E_{\text{el}}\ket{\Psi_{\text{el}}}

where $\ket{\Psi_{\text{el}}}$ is the electronic wave function.

In the following, we focus on finding the ground state energy

.. math::

        E_0 = \braket{\Psi_0|H_{\text{el}}|\Psi_0}

where $\ket{\Psi_0}$ is a ground state wave function of the system.

A starting point for solving this problem is the (restricted) **Hartree-Fock** method which produces a set $\{\psi_i\}$ of $K$ spacial orbitals
that correspond to $M=2K$ spin orbitals $\chi_{2i}=\psi_{i}\alpha$ and $\chi_{2i+1}=\psi_{i}\beta$, for $i=0,\dotsc,K-1$, for the $N$ electrons 
of the molecule.

Utilizing the second quantization formalism, the electronic Hamiltonian is then expressed 
in the basis of the solutions (i.e., the spin orbitals) of the Hartree-Fock method: 

.. math::
    
    \hat H_{\text{el}} = \sum\limits_{i,j=0}^{M-1}h_{ij}a_i^{\dagger}a_j+\frac{1}{2}\sum\limits_{i,j,k,l=0}^{M-1}h_{i,j,k,l}a_i^{\dagger}a_j^{\dagger}a_ka_l

where $a_i^{\dagger}$ and $a_i$ are the fermionic creation and annihilation operators, respectively.

Here, the coefficients :math:`h_{ij}` and :math:`h_{ijkl}` are one- and two-electron integrals which can be computed classically:

- One-electron integrals:

    .. math::

        h_{i,j}=\int\mathrm dx \chi_i^*(x)\chi_j(x)

- Two-electron integrals:

    .. math::

        h_{i,j,k,l} = \int\mathrm dx_1\mathrm dx_2\chi_i^*(x_1)\chi_j^*(x_2)\frac{1}{|r_1-r_2|}\chi_k(x_1)\chi_l(x_2)


Note: There is a difference between the **physicists's notation** (above) and the **chemists' notation** for the two-electron integrals!

The Hartree-Fock state

.. math::

    \ket{\Psi_{\text{HF}}} = \ket{1_0,\dotsc,1_{N-1},0_N,\dotsc,0_{M-1}}

where the first (i.e., the lowest energy) $N$ orbitals are occupied 
is the best approximation of the ground state $\ket{\hat\Psi_0}$ of the Hamiltonian $\hat H_{\text{el}}$ in this form.

All feasible $N$-electron states are expressed as a superposition of the Hartree-Fock state and

- single electron excitation states:

    .. math::

        \ket{\Psi_i^r}=\ket{1_0,\dotsc,0_i,\dotsc,1_{N-1}0_{N},\dotsc,1_r,\dotsc,0_{M-1}}

- double electron excitation states:

    .. math::

        \ket{\Psi_{ij}^{rs}}=\ket{1_0,\dotsc,0_i,\dotsc,0_j,\dotsc,1_{N-1},0_{N},\dotsc,1_r,\dotsc,1_s,\dotsc,0_{M-1}}

- higher order (triple, quadruple, ect.) excitation states.

That is, a ground state can be expressed as

.. math::

    \ket{\hat\Psi_{0}}=c_0\ket{\Psi_{\text{HF}}}+\sum_{i,r}c_i^r\ket{\Psi_i^r}+\sum_{i<j,r<s}c_{ij}^{rs}\ket{\Psi_i^r}+\dotsb

Solving the electronic structure problem, i.e., finding a ground state $\ket{\hat\Psi_0}$ of the electronic Hamiltonian $\hat H_{\text{el}}$ within the feasible $N$-electron subspace,
with a quantum computer requires transforming the fermionic representation into a qubit representation.
This is achieved by, e.g., the Jordan-Wigner, Parity, or Bravyi-Kitaev transformation.

Electronic structure problem
----------------------------

.. autofunction:: electronic_structure_problem

Helper functions
----------------

.. autofunction:: electronic_data

Hamiltonian
-----------

.. autofunction:: create_electronic_hamiltonian

Ansatz
------

.. autofunction:: create_QCCSD_ansatz

.. autofunction:: create_hartree_fock_init_function

