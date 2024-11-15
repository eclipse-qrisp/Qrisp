"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

import numpy as np

from qrisp import h, x, cx, ry, control, conjugate
from qrisp.operators.qubit import QubitOperator, QubitTerm
from qrisp.operators.fermionic import *
from functools import cache
import itertools
import math

#
# helper functions
#

def verify_symmetries(two_int):
    """
    Checks the symmetries of the two_electron integrals tensor in physicist's notation.

    Parameters
    ----------
    int_two : numpy.ndarray
        The two-electron integrals w.r.t. spin orbitals in physicist's notation.

    Returns
    -------

    """

    M = two_int.shape[0]
    for i in range(M):
        for j in range(M):
            for k in range(M):
                for l in range(M):
                    test1 = abs(two_int[i][j][k][l]-two_int[j][i][l][k])
                    test2 = abs(two_int[i][j][k][l]-two_int[k][l][i][j])
                    test3 = abs(two_int[i][j][k][l]-two_int[l][k][j][i])
                    if test1>1e-9 or test2>1e-9 or test3>1e-9:
                        return False
    return True

def delta(i,j):
    if i==j:
        return 1
    else:
        return 0

#
# spacial orbitals to spin orbitals
# 

def omega(x):
    return x%2

def spacial_to_spin(one_int,two_int):
    r"""
    Transforms one- and two-electron integrals w.r.t. $M$ spacial orbitals $\psi_0,\dotsc,\psi_{M-1}$ to 
    one- and two-electron integrals w.r.t. $2M$ spin orbitals $\chi_{2i}=\psi_{i}\alpha$, 
    $\chi_{2i+1}=\psi_{i}\beta$ for $i=0,\dotsc,M-1$.

    That is, given one- and two-electron integrals for spacial orbitals:

    .. math::

        h_{ij} &= \braket{\psi_i|h|\psi_j} \\
        h_{ijkl} &= \braket{\psi_i\psi_j|\psi_k\psi_l}

    this methods computes one- and two-electron integrals for spin orbitals:

    .. math::

        \tilde h_{ijkl} &= \braket{\chi_i|h|\chi_j} \\
        \tilde h_{ijkl} &= \braket{\chi_i\chi_j|\chi_k\chi_l}

    Parameters
    ----------
    one_int : numpy.ndarray
        The one-electron integrals w.r.t. spacial orbitals.
    two_int : numpy.ndarray
        The two-electron integrals w.r.t. spacial orbitals.

    Returns
    -------
    one_int_spin : numpy.ndarray
        The one-electron integrals w.r.t. spin orbitals.
    two_int_spin : numpy.ndarray
        The two-electron integrals w.r.t. spin orbitals.

    """

    num_spacial_orbs = one_int.shape[0]
    num_spin_orbs = 2*num_spacial_orbs

    # Initialize the spin-orbital one-electron integral tensor
    one_int_spin = np.zeros((num_spin_orbs, num_spin_orbs))

    for i in range(num_spacial_orbs):
        for j in range(num_spacial_orbs):

            one_int_spin[2*i][2*j] = one_int[i][j]
            
            one_int_spin[2*i+1][2*j+1] = one_int[i][j]

    # Initialize the spin-orbital two-electron integral tensor
    two_int_spin = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))

    for i in range(num_spacial_orbs):
        for j in range(num_spacial_orbs):  
            for k in range(num_spacial_orbs):
                for l in range (num_spacial_orbs):

                    two_int_spin[2*i][2*j+1][2*k+1][2*l] = two_int[i][j][k][l]

                    two_int_spin[2*i+1][2*j][2*k][2*l+1] = two_int[i][j][k][l]

                    two_int_spin[2*i][2*j][2*k][2*l] = two_int[i][j][k][l]

                    two_int_spin[2*i+1][2*j+1][2*k+1][2*l+1] = two_int[i][j][k][l]
    
    return one_int_spin, two_int_spin

def electronic_data(mol):
    """
    A function that utilizes `restricted Hartree-Fock (RHF) <https://pyscf.org/user/scf.html>`_
    calculation in the `PySCF <https://pyscf.org>`_ quantum chemistry package to obtain the electronic data for
    defining an electronic structure problem.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The `molecule <https://pyscf.org/user/gto.html#>`_.

    Returns
    -------
    data : dict
        A dictionary specifying the electronic data for a molecule. The following data is provided:
        
        * ``one_int`` : numpy.ndarray
            The one-electron integrals w.r.t. spin orbitals (in physicists' notation).
        * ``two_int`` : numpy.ndarray
            The two-electron integrals w.r.t. spin orbitals (in physicists' notation).
        * ``num_orb`` : int
            The number of spin orbitals.
        * ``num_elec`` : int
            The number of electrons.
        * ``energy_nuc``
            The nuclear repulsion energy.
        * ``energy_hf``    
            The Hartree-Fock ground state energy.

    """

    from pyscf import scf, ao2mo

    data = {}

    threshold = 1e-9
    def apply_threshold(matrix, threshold):
        matrix[np.abs(matrix) < threshold] = 0
        return matrix
    
    # Set verbosity level to 0 to suppress output
    mol.verbose = 0
    
    # Perform a Hartree-Fock calculation
    mf = scf.RHF(mol)
    energy_hf = mf.kernel()

    # Extract one-electron integrals
    one_int = apply_threshold(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff,threshold)
    # Extract two-electron integrals (electron repulsion integrals)
    two_int = ao2mo.kernel(mol,mf.mo_coeff)
    # Full tensor with chemist's notation
    two_int = apply_threshold(ao2mo.restore(1,two_int,mol.nao_nr()),threshold)
    # Full tensor with physicist's notation
    two_int = np.transpose(two_int,(0,2,3,1))

    # Convert spacial orbital to spin orbitals
    one_int, two_int  = spacial_to_spin(one_int,two_int)

    data['mol'] = mol
    data['one_int'] = one_int
    data['two_int'] = two_int
    data['num_orb'] = 2*mf.mo_coeff.shape[0]  # Number of spin orbitals
    data['num_elec'] = mol.nelectron
    data['energy_nuc'] = mol.energy_nuc()
    data['energy_hf'] = energy_hf

    return data

def create_electronic_hamiltonian(arg, active_orb=None, active_elec=None):
    """
    Creates the qubit Hamiltonian for an electronic structure problem. 
    If an Active Space (AS) is specified, the Hamiltonian is calculated following this `paper <https://arxiv.org/abs/2009.01872>`_.
    
    Parameters
    ----------
    arg : pyscf.gto.Mole or dict
        A PySCF `molecule <https://pyscf.org/user/gto.html#>`_ or
        a dictionary specifying the electronic data for a molecule. The following data is required:
        
        * ``one_int`` : numpy.ndarray
            The one-electron integrals w.r.t. spin orbitals (in physicists' notation).
        * ``two_int`` : numpy.ndarray
            The two-electron integrals w.r.t. spin orbitals (in physicists' notation).
        * ``num_orb`` : int
            The number of spin orbitals.
        * ``num_elec`` : int
            The number of electrons.

    active_orb : int, optional
        The number of active spin orbitals.
    active_elec : int, optional
        The number of active electrons.

    Returns
    -------
    H : :ref:`FermionicOperator`
        The fermionic Hamiltonian.
    
    Examples
    --------

    We calucalte the fermionic Hamiltonian for the Hydrogen molecule, and transform it to a Pauli Hamiltonian via Jordan-Wigner transform.

    ::

        from pyscf import gto
        from qrisp.vqe.problems.electronic_structure import *

        mol = gto.M(
            atom = '''H 0 0 0; H 0 0 0.74''',
            basis = 'sto-3g')

        H = create_electronic_hamiltonian(mol)
        H.to_qubit_operator()   

    Yields:

    .. math::

        -&0.812170607248714 - 0.0453026155037992 X_0X_1Y_2Y_3 + 0.0453026155037992 X_0Y_1Y_2X_3 - 0.0453026155037992 Y_0Y_1X_2X_3

        &+0.171412826447769 Z_0 + 0.168688981703612 Z_0Z_1 + 0.120625234833904 Z_0Z_2 + 0.165927850337703 Z_0Z_3 + 0.171412826447769 Z_1

        &+0.165927850337703 Z_1Z_2 + 0.120625234833904 Z_1Z_3 - 0.223431536908133 Z_2 + 0.174412876122615Z Z_2Z_3 - 0.223431536908133 Z_3

    """

    import pyscf

    if isinstance(arg,pyscf.gto.Mole):
        data = electronic_data(arg)
    elif isinstance(arg,dict):
        data = arg
        if not verify_symmetries(data['two_int']):
            raise Warning("Failed to verify symmetries for two-electron integrals")
    else:
        raise TypeError("Cannot create electronic Hamiltonian from type "+str(type(arg)))

    one_int = data['one_int']
    two_int = data['two_int']
    M = data['num_orb']
    N = data['num_elec']
    K = active_orb
    L = active_elec

    if K is None or L is None:
        K = M
        L = N

    if L>N or K>M or K<L or K+N-L>M:
        raise Exception("Invalid number of active electrons or orbitals")

    # number of inactive electrons 
    I = N-L

    # inactive Fock operator
    F = one_int.copy()
    for p in range(M):
        for q in range(M):
            for i in range(I):
                #F[p][q] += (two_int[i][p][i][q]-two_int[i][q][p][i])
                F[p][q] += (two_int[i][p][q][i]-two_int[i][q][i][p])

    # inactive energy
    E = 0
    for j in range(I):
        E += (one_int[j][j]+F[j][j])/2

    # Hamiltonian
    H=E
    for i in range(K):
        for j in range(K):
            if F[I+i][I+j]!=0:
                H += F[I+i][I+j]*c(i)*a(j)
    
    for i in range(K):
        for j in range(K): 
            for k in range(K):
                for l in range(K):
                    if two_int[I+i][I+j][I+k][I+l]!=0 and i!=j and k!=l:
                        H += (0.5*two_int[I+i][I+j][I+k][I+l])*c(i)*c(j)*a(k)*a(l)

    return H.reduce()

#
# ansatz
#

def conjugator(i,j):
    h(i)
    cx(i,j)

def pswap(phi,i,j):
    with conjugate(conjugator)(i,j):
        ry(-phi/2, [i,j])  

def conjugator2(i,j,k,l):
    cx(i,j)
    cx(k,l) 

def pswap2(phi,i,j,k,l):
    with conjugate(conjugator2)(i,j,k,l):
        with control([j,l],ctrl_state='00'):
            pswap(phi,i,k)

def create_QCCSD_ansatz(M,N):
    r"""
    This method creates a function for applying one layer of the `QCCSD ansatz <https://arxiv.org/abs/2005.08451>`_.

    The chemistry-inspired Qubit Coupled Cluster Single Double (QCCSD) ansatz evolves the initial state, 
    usually, the Hartree-Fock state

    .. math::

        \ket{\Psi_{\text{HF}}}=\ket{1_0,\dotsc,1_N,0_{N+1},\dotsc,0_M}
    
    under the action of parametrized (non-commuting) single and double excitation unitaries.
    
    The single (S) excitation unitaries $U_i^r$ implement a continuous swap for qubits $i$ and $r$:
    
    .. math::

        U_i^r(\theta) = \begin{pmatrix}
                        1&0&0&0\\
                        0&\cos(\theta)&-\sin(\theta)&0\\
                        0&\sin(\theta)&\cos(\theta)&0\\
                        0&0&0&1
                        \end{pmatrix}

    Similarly, the double (D) excitation unitaries $U_{ij}^{rs}(\theta)$ implement a continuous swap 
    for qubit pairs $i,j$ and $r,s$.
        
    Parameters
    ----------
    M : int
        The number of (active) spin orbitals.
    N : int
        The number of (active) electrons.

    Returns
    -------
    ansatz : function
        A function that can be applied to a :ref:`QuantumVariable` and a list of parameters.
    num_params : int
        The number of parameters.
    
    """

    spin_down_occupied = [i for i in range(N) if i%2==0]
    spin_down_virtual = [i for i in range(N,M) if i%2==0]
    spin_up_occupied = [i for i in range(N) if i%2==1]
    spin_up_virtual = [i for i in range(N,M) if i%2==1]

    num_singles = len(spin_down_occupied)*len(spin_down_virtual) + len(spin_up_occupied)*len(spin_up_virtual)

    num_doubles = len(spin_down_occupied)*len(spin_up_occupied)*len(spin_down_virtual)*len(spin_up_virtual) \
                    +math.comb(len(spin_down_occupied),2)*math.comb(len(spin_down_virtual),2) \
                    +math.comb(len(spin_up_occupied),2)*math.comb(len(spin_up_virtual),2)
    
    num_params = num_singles + num_doubles

    def ansatz(qv, theta):

        num_params = 0
        # Single excitations
        for i in spin_down_occupied:
            for j in spin_down_virtual:
                    pswap(theta[num_params],qv[i],qv[j])
                    num_params += 1
        
        for i in spin_up_occupied:
            for j in spin_up_virtual:
                    pswap(theta[num_params],qv[i],qv[j])
                    num_params += 1
        
        # Double excitation
        for i in spin_down_occupied:
            for j in spin_up_occupied:
                for k in spin_down_virtual:
                    for l in spin_up_virtual:
                        pswap2(theta[num_params],qv[i],qv[j],qv[k],qv[l])
                        num_params += 1

        for i,j in itertools.combinations(spin_down_occupied,2):
            for k,l in itertools.combinations(spin_down_virtual,2):
                pswap2(theta[num_params],qv[i],qv[j],qv[k],qv[l])
                num_params += 1

        for i,j in itertools.combinations(spin_up_occupied,2):
            for k,l in itertools.combinations(spin_up_virtual,2):
                pswap2(theta[num_params],qv[i],qv[j],qv[k],qv[l])
                num_params += 1

    return ansatz, num_params

def create_hartree_fock_init_function(M, N, mapping_type='jordan_wigner'):
    """
    Creates the function that, when applied to a :ref:`QuantumVariable`, initializes the Hartee-Fock state:
    If the mapping type is ``jordan_wigner``, the first ``N`` qubits are initialized in the $\ket{1}$ state.
    If the mapping type is ``parity``, the first ``N`` qubits with even index are initialized in the $\ket{1}$ state.

    Parameters
    ----------
    M : int
        The number of (active) spin orbitals.
    N : int
        The number of (active) electrons.
    mapping_type : string, optinal
        The mapping from fermionic Hamiltonian to qubit Hamiltonian. Available are ``jordan_wigner``, ``parity``.
        The default is ``jordan_wigner``.

    Returns
    -------
    init_function : function
        A function that can be applied to a :ref:`QuantumVariable`.

    """

    if mapping_type=='jordan_wigner':
        def init_function(qv):
            for i in range(N):
                x(qv[i])

    if mapping_type=='parity':
        def init_function(qv):
            for i in range(N//2):
                x(qv[2*i])
            # odd number of electrons
            if N%2==1:
                for i in range(N-1,M):
                    x(qv[i])

    return init_function


def electronic_structure_problem(arg, active_orb=None, active_elec=None, mapping_type='jordan_wigner', ansatz_type='QCCSD', threshold=1e-4):
    r"""
    Creates a VQE problem instance for an electronic structure problem defined by the 
    one-electron and two-electron integrals for the spin orbitals (in physicists' notation).

    The problem Hamiltonian is given by:

    .. math::

        H = \sum\limits_{i,j=0}^{M-1}h_{i,j}a^{\dagger}_ia_j + \sum\limits_{i,j,k,l=0}^{M-1}h_{i,j,k,l}a^{\dagger}_ia^{\dagger}_ja_ka_l
    
    for one-electron integrals:

    .. math::

        h_{i,j}=\int\mathrm dx \chi_i^*(x)\chi_j(x)

    and two-electron integrals:

    .. math::

        h_{i,j,k,l} = \int\mathrm dx_1\mathrm dx_2\chi_i^*(x_1)\chi_j^*(x_2)\frac{1}{|r_1-r_2|}\chi_k(x_1)\chi_l(x_2)

    Parameters
    ----------
    arg : pyscf.gto.Mole or dict
        A PySCF `molecule <https://pyscf.org/user/gto.html#>`_ or
        a dictionary specifying the electronic data for a molecule. The following data is required:
        
        * ``one_int`` : numpy.ndarray
            The one-electron integrals w.r.t. spin orbitals (in physicists' notation).
        * ``two_int`` : numpy.ndarray
            The two-electron integrals w.r.t. spin orbitals (in physicists' notation).
        * ``num_orb`` : int
            The number of spin orbitals.
        * ``num_elec`` : int
            The number of electrons.

    active_orb : int, optional
        The number of active spin orbitals.
    active_elec : int, optional
        The number of active electrons.
    mapping_type : string, optinal
        The mapping from fermionic Hamiltonian to qubit Hamiltonian. Available are ``jordan_wigner``, ``parity``.
        The default is ``jordan_wigner``.
    ansatz_type : string, optional
        The ansatz type. Availabe is ``QCCSD``. The default is ``QCCSD``.
    threshold : float, optional
        The threshold for the absolute value of the coefficients of Pauli products in the quantum Hamiltonian. The default is 1e-4.

    Returns
    -------
    VQEProblem
        The VQE problem instance.

    Examples
    --------

    We calculate the electronic energy for the Hydrogen molecule at bond distance 0.74 angstroms:

    ::

        from pyscf import gto
        from qrisp import QuantumVariable
        from qrisp.vqe.problems.electronic_structure import *

        mol = gto.M(
            atom = '''H 0 0 0; H 0 0 0.74''',
            basis = 'sto-3g')

        vqe = electronic_structure_problem(mol)
        vqe.set_callback()

        energy = vqe.run(QuantumVariable(4),depth=1,max_iter=50)
        print(energy)
        #Yields -1.8461290172512965
    
    """
    from qrisp.vqe import VQEProblem
    import pyscf
    if isinstance(arg,pyscf.gto.Mole):
        data = electronic_data(arg)
    elif isinstance(arg,dict):
        data = arg
        if not verify_symmetries(data['two_int']):
            raise Warning("Failed to verify symmetries for two-electron integrals")
    else:
        raise TypeError("Cannot instantiate VQEProblem from type "+str(type(arg)))
    
    M = data['num_orb']
    N = data['num_elec']
    K = active_orb
    L = active_elec

    if K is None or L is None:
        K = M
        L = N

    ansatz, num_params = create_QCCSD_ansatz(K,L)

    fermionic_hamiltonian = create_electronic_hamiltonian(data,K,L)
    hamiltonian = fermionic_hamiltonian.to_qubit_operator(mapping_type=mapping_type)
    hamiltonian.apply_threshold(threshold)

    return VQEProblem(hamiltonian, ansatz, num_params, init_function=create_hartree_fock_init_function(K,L,mapping_type))