.. _VQEElectronicStructure:

VQE Electronic structure problem implementation
===============================================

.. currentmodule:: vqe.problems.electronic_structure

Problem description
-------------------


We consider finding approximate solutions to the non-relativistic time-independent Schrödinger equation

.. math::

    H\ket{\Psi}=E\ket{\Psi}

where $H$ is the Hamitonian operator for a system of $M$ nuclei and $N$ electrones with coordinates $R_A\in\mathbb R^3$ and $r_i\in\mathbb R^3$, respectively, is given by:

.. math::

    H= -\sum_{i=1}^N\frac12\nabla_i^2-\sum_{i=1}^M\frac{1}{2M_A}\nabla_A^2-\sum_{i=1}^N\sum_{A=1}^M\frac{Z_A}{r_{iA}}+\sum_{i=1}^N\sum_{j>i}^N\frac{1}{r_{ij}}+\sum_{A=1}^M\sum_{B>A}^M\frac{Z_AZ_B}{R_{AB}}

where $r_{iA}=|r_i-R_A|$, $r_{ij}=|r_i-r_j|$, and $R_{AB}=|R_A-R_B|$, are the distances between electrons and nuclei, $M_A$ is the ratio of the mass of nucleus $A$ to an electron, 
and $Z_A$ is the atomic number of nulecus $A$. 

Within the Born-Oppenheimer approximation, one considers the electrons in a molecule to be moving a the field of fixed nuclei. 
Therefore, the kinetic energy of the nuclei is neglected, and the nuclear repulsion energy is constant.

The Hamiltonioan describing the motion of $N$ electrons in the field of $M$ fixed nuclei is

.. math::

    H_{\text{el}} = -\sum_{i=1}^N\frac12\nabla_i^2-\sum_{i=1}^N\sum_{A=1}^M\frac{Z_A}{r_{iA}}+\sum_{i=1}^N\sum_{j>i}^N\frac{1}{r_{ij}}

The electronic problem is the solution to the Schrödinger equation for the electronic Hamiltonian,

.. math::

    H_{\text{el}}\ket{\Psi_{\text{el}}}=E_{\text{el}}\ket{\Psi_{\text{el}}}

is the electronic wave function $\ket{\Psi_{\text{el}}}$.

The total energy of the system also includes the nuclear repulsion energy:

.. math::

    E_{\text{tot}} = E_{\text{el}} + \sum_{A=1}^M\sum_{B>A}^M\frac{Z_AZ_B}{R_{AB}}


Spatial orbitals $\psi_i$

Spin orbitals $\chi_i$

$N$-electron wave functions




The Hartree-Fock method produces a set $\{\chi_i\}$ of $2K$ spin orbitals. The Hartree-Fock ground state

.. math::

    \ket{\Psi_0} = \ket{\chi_1,\chi_2,\dotsc,\chi_a,\chi_b,\dotsc,\chi_N}

is the best approximation of the ground state in form of a single Slater determinant.

Clearly, it is just one of the ${n \choose K}$ determinant that can be built form $2N$ orbitals and $N$ electrons.

A singly excited determinant

.. math::

    \ket{\Psi_a^r} = \ket{\chi_1,\chi_2,\dotsc,\chi_r,\chi_b,\dotsc,\chi_N}

A doubly excited determinant 

.. math::

    \ket{\Psi_{ab}^{rs}} = \ket{\chi_1,\chi_2,\dotsc,\chi_r,\chi_s,\dotsc,\chi_N}

All determinants can be classified as the Hartree-Fock ground state or singly, doubly, triply, ect.
excited states with their relevance deminishing in that order.

Then, within the given approximation, the excat wave function for any sate of the system is
a superposition of the HF gs and excited states, i.e.,

.. math::

    \ket{\Psi} = c_0\ket{\Psi_0}+\sum_{a,r}c_a^r\ket{\Psi_a^r}+\sum_{a<b,r<s}c_{ab}^{rs}\ket{\Psi_{ab}^{rs}}+\dotsb



The electronic Hamiltonian is decomposed 


.. math::

    H_1 &= -\sum_{i=1}^N\frac12\nabla_i^2-\sum_{i=1}^N\sum_{A=1}^M\frac{Z_A}{r_{iA}}\\
    H_2 &= \sum_{i=1}^N\sum_{j>i}^N\frac{1}{r_{ij}}
    

.. math::

    h(i) = -\frac12\nabla_i^2-\sum_A\frac{Z_A}{r_{iA}}


.. math::

    \braket{i|h|j} &= \int\mathrm dx_1\chi_i^*(x_1)h(r_1)\chi_j(x_1)\\

    \braket{ij|kl} &= \int\mathrm dx_1\mathrm dx_2\chi_i^*(x_1)\chi_j^*(x_2)r_{12}^{-1}\chi_k(x_1)\chi_k(x_2)



Second quantization

The second quantization is a formalism that incorporates the anti symmery rperty od the wave functions into algebraic 
properties of certain operations, therefore, an elgant formalism for treating many-electron systems.


todo




Creation and Annihilation operators


.. math::

    \mathcal O_1 &= \sum_{ij}\braket{i|h|j}a_i^{\dagger}a_j

    \mathcal O_2 &= \frac12\sum_{ijkl}\braket{ij|kl}a_i^{\dagger}a_j^{\dagger}a_ka_l

Finally, in second quantization, the electronic Hamiltonian is expressed as:

.. math::

    H_{\text{el}} = \mathcal O_1 + \mathcal O_2 = \sum_{ij}\braket{i|h|j}a_i^{\dagger}a_j + \frac12\sum_{ijkl}\braket{ij|kl}a_i^{\dagger}a_j^{\dagger}a_ka_l




hello






.. math::
    
    H=\sum\limits_{i,j=1}^{M}h_{ij}a_i^{\dagger}a_j+\frac{1}{2}\sum\limits_{i,j,k,l}^{M}h_{i,j,k,l}a_i^{\dagger}a_j^{\dagger}a_ka_l

The coefficients :math:`h_{ij}` and :math:`h_{ijkl}` are one- and two-electron integrals which can be computed classically.

Solving the electronic structure problem on a quantum computer requires transforming the fermionic representation into a qubit representation.
This is achieved by, e.g., the Jordan-Wigner, Parity, or Bravyi-Kitaev transformations.

Electronic Structure problem
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

