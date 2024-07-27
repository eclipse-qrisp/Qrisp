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

#todo
import sympy as sp
from sympy import I
from sympy import *
from qrisp.misc.spin import *
from qrisp.misc.pauli_operator import *
from functools import cache

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
# \Psi_1,...,\\

# spin to spacial
#def sp(x):
#    return x//2

def omega(x):
    return x%2

def spacial_to_spin(int_one,int_two):
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
    int_one : numpy.ndarray
        The one-electron integrals w.r.t. spacial orbitals.
    int_two : numpy.ndarray
        The two-electron integrals w.r.t. spacial orbitals.

    Returns
    -------
    int_one_spin : numpy.ndarray
        The one-electron integrals w.r.t. spin orbitals.
    int_two_spin : numpy.ndarray
        The two-electron integrals w.r.t. spin orbitals.

    """

    n_spin_orbs = 2*int_one.shape[0]

    # Initialize the spin-orbital one-electron integral tensor
    int_one_spin = np.zeros((n_spin_orbs, n_spin_orbs))
    for i in range(n_spin_orbs):
        for j in range(n_spin_orbs):
            int_one_spin[i][j] = delta(omega(i),omega(j))*int_one[i//2][j//2]

    # Initialize the spin-orbital two-electron integral tensor
    int_two_spin = np.zeros((n_spin_orbs, n_spin_orbs, n_spin_orbs, n_spin_orbs))
    for i in range(n_spin_orbs):
        for j in range(n_spin_orbs):
            for k in range(n_spin_orbs):
                for l in range (n_spin_orbs):
                    int_two_spin[i][j][k][l] = delta(omega(i),omega(j))*delta(omega(k),omega(l))*int_two[i//2][j//2][k//2][l//2]

    return int_one_spin, int_two_spin

def electronic_data(mol):
    """
    A wrapper function that utilizes restricted Hartree-Fock (RHF) calculation 
    in the pyscf quantum chemistry package to obtain the electronic data for
    defining an electronic structure problem.

    Parameters
    ---------
    mol : pyscf.gto.Mole
        The `molecule <https://pyscf.org/user/gto.html#>`_.

    Returns
    -------
    data : dict
        
        * ``mol`` 
        * ``one_int``
        * ``two_int``
        * ``num_orb``
        * ``num_elec``
        * ``energy_nuc``
        * ``energy_scf``

    """
    from pyscf import gto, scf, ao2mo

    data = {}

    threshold = 1e-9
    def apply_threshold(matrix, threshold):
        matrix[np.abs(matrix) < threshold] = 0
        return matrix
    
    # Perform a Hartree-Fock calculation
    mf = scf.RHF(mol)
    energy_scf = mf.kernel()

    # Extract one-electron integrals
    one_int = apply_threshold(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff,threshold)
    # Extract two-electron integrals (electron repulsion integrals)
    two_int = ao2mo.kernel(mol,mf.mo_coeff)
    # Full tensor with chemist's notation
    two_int = apply_threshold(ao2mo.restore(1,two_int,mol.nao_nr()),threshold)
    # Convert spacial orbital to spin orbitals
    one_int, two_int  = spacial_to_spin(one_int,two_int)

    data['mol'] = mol
    data['one_int'] = one_int
    data['two_int'] = np.transpose(two_int,(0,2,1,3))
    data['num_orb'] = 2*mf.mo_coeff.shape[0]  # Number of spin orbitals
    data['num_elec'] = mol.nelectron
    data['energy_nuc'] = mol.energy_nuc()
    data['energy_scf'] = energy_scf

    return data

#
# Fermion to qubit mappings
#

# Jordan-Wigner annihilation operaror 
@cache
def a_jw(j):
    return PauliOperator({tuple([(i,"Z") for i in range(j)]+[(j,"X")]):0.5,tuple([(i,"Z") for i in range(j)]+[(j,"Y")]):0.5j})

# Jordan-Wigner creation operator 
@cache
def c_jw(j):
    return PauliOperator({tuple([(i,"Z") for i in range(j)]+[(j,"X")]):0.5,tuple([(i,"Z") for i in range(j)]+[(j,"Y")]):-0.5j})

# Parity annihilation operator
@cache
def a_par(j,M):
    if j>0:
        return PauliOperator({tuple([(j-1,"Z"),(j,"X")]+[(i,"X") for i in range(j+1,M)]):0.5,tuple([(j,"Y")]+[(i,"X") for i in range(j+1,M)]):0.5j})
    else:
        return PauliOperator({tuple([(j,"X")]+[(i,"X") for i in range(j+1,M)]):0.5,tuple([(j,"Y")]+[(i,"X") for i in range(j+1,M)]):0.5j})

# Parity creation operator
@cache
def c_par(j,M):
    if j>0:
        return PauliOperator({tuple([(j-1,"Z"),(j,"X")]+[(i,"X") for i in range(j+1,M)]):0.5,tuple([(j,"Y")]+[(i,"X") for i in range(j+1,M)]):-0.5j},0)
    else:
        return PauliOperator({tuple([(j,"X")]+[(i,"X") for i in range(j+1,M)]):0.5,tuple([(j,"Y")]+[(i,"X") for i in range(j+1,M)]):-0.5j},0)

@cache
def ann(i,M,mapping_type):
    """
    Returns the qubit operator for the fermionic annihilation operator $a_i$.

    Parameters
    ----------
    i : int
        The index of the annihilation operator $a_i$.
    M: int
        The number of fermions.
    mapping_type : str
        The mapping type. Available are ``jordan_wigner``, ``parity``.

    Returns
    -------
    PauliOperator
        The qubit PauliOperator for the annihilation operator $a_i$.

    """

    if mapping_type=='jordan_wigner':
        return a_jw(i)
    if mapping_type=='parity':
        return a_par(i,M)

@cache
def cre(i,M,mapping_type):
    """
    Returns the qubit operator for the fermionic creation operator $a_i^{\dagger}$.

    Parameters
    ----------
    i : int
        The index of the annihilation operator $a_i^{\dagger}$.
    M: int
        The number of fermions.
    mapping_type : str
        The mapping type. Available are ``jordan_wigner``, ``parity``.

    Returns
    -------
    PauliOperator
        The qubit PauliOperator for the annihilation operator $a_i$.

    """

    if mapping_type=='jordan_wigner':
        return c_jw(i)
    if mapping_type=='parity':
        return c_par(i,M)

@cache
def cre2(i,j,M,mapping_type):

    if mapping_type=='jordan_wigner':
        return c_jw(i)*c_jw(j)
    if mapping_type=='parity':
        return c_par(i,M)*c_par(j,M)

@cache
def ann2(i,j,M,mapping_type):

    if mapping_type=='jordan_wigner':
        return a_jw(i)*a_jw(j)
    if mapping_type=='parity':
        return a_par(i,M)*a_par(j,M)

#
# Hamiltonian
#

def create_electronic_hamiltonian(one_int, two_int, M, N, K=None, L=None, mapping_type='jordan_wigner', threshold=1e-4):
    """
    Creates the qubit Hamiltonian for an electronic structure problem. 
    If an Active Space (AS) is specified, the Hamiltonian is calculated following this `paper <https://arxiv.org/abs/2009.01872>`_.
    
    Parameters
    ----------
    int_one : numpy.ndarray
        The one-electron integrals w.r.t. spin orbitals (in physicists' notation).
    int_two : numpy.ndarray
        The two-electron integrals w.r.t. spin orbitals (in physicists' notation).
    M : int
        The number of spin orbitals.
    N : int
        The number of electrons.
    K : int, optional
        The number of active spin orbitals.
    L : int, optional
        The number of active electrons.
    mapping_type : string, optinal
        The mapping from the fermionic Hamiltonian to the qubit Hamiltonian. Available are ``jordan_wigner``, ``parity``.
        The default is ``jordan_wigner``.
    threshold : float, optional
        The threshold for the absolute value of the coefficients of Pauli products in the quantum Hamiltonian. The default is 1e-4.

    Returns
    -------
    H : PauliOperator
        The qubit Hamiltonian.
    
    """

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
                F[p][q] += (two_int[i][p][i][q]-two_int[i][q][p][i])

    # inactive energy
    E = 0
    for j in range(I):
        E += (one_int[j][j]+F[j][j])/2

    # Hamiltonian
    H = PauliOperator({():E})
    for i in range(K):
        for j in range(K):
            if F[I+i][I+j]!=0:
                H += F[I+i][I+j]*cre(i,K,mapping_type)*ann(j,K,mapping_type)
    
    for i in range(K):
        for j in range(K): 
            for k in range(K):
                for l in range(K):
                    if two_int[I+i][I+j][I+k][I+l]!=0 and i!=j and k!=l:
                        H += -0.5*two_int[I+i][I+j][I+k][I+l]*cre2(i,j,K,mapping_type)*ann2(k,l,K,mapping_type)

    # apply threshold
    H.apply_threshold(threshold)
    
    return H

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
    This method implements the `QCCSD ansatz <https://arxiv.org/abs/2005.08451>`_.

    A chemistry-inspired Coupled Cluster Single Double (CCSD) ansatz assumes that the ground state of the electronic Hamiltonian 
    is a superposition of the Hartree-Fock state:

    .. math::

        \ket{\Psi_{\text{HF}}}=\ket{1_0,\dotsc,1_N,0_{N+1},\dotsc,0_M}
    
    and the single (S) electron excitation states:

    .. math::

        \ket{\Psi_i^r}=\ket{1_0,\dotsc,0_i,\dotsc,1_N,0_{N+1},\dotsc,1_r,\dotsc,0_M}

    and the double (D) electron excitation states:

    .. math::

        \ket{\Psi_{ij}^{rs}}=\ket{1_0,\dotsc,0_i,\dotsc,0_j,\dotsc,1_N,0_{N+1},\dotsc,1_r,\dotsc,1_s,\dotsc,0_M}

    That is, the ansatz assumes the following form of the ground state:

    .. math::

        \ket{\Psi_{\text{CCSD}}}=c_0\ket{\Psi_{\text{HF}}}+\sum_{i,r}c_i^r\ket{\Psi_i^r}+\sum_{i<j,r<s}c_{ij}^{rs}\ket{\Psi_i^r}

    Parameters
    ----------
    M : int
        The number of (active) spin orbitals.
    N : int
        The number of (active) electrons.

    Returns
    -------
    ansatz : function
        This method creates a function for applying one layer of the ansatz.
    num_params : int
        The number of parameters.
    
    """

    num_params = N*(M-N) + N*(N-1)*(M-N)*(M-N-1)//4

    def ansatz(qv, theta):

        num_params = 0
        # Single excitations
        for i in range(N):
            for j in range(N,M):
                pswap(theta[num_params],qv[i],qv[j])
                num_params += 1
        
        # Double excitations
        for i in range(N-1):
            for j in range(i+1,N):
                for k in range(N,M-1):
                    for l in range(k+1,M):
                        pswap2(theta[num_params],qv[i],qv[j],qv[k],qv[l])
                        num_params += 1

    return ansatz, num_params

def create_hartree_fock_init_function(N, mapping_type='jordan_wigner'):
    """
    Creates a function for initializing the Hartee-Fock state, i.e., the first ``N`` qubits are initialized in the $\ket{1}$ state.

    Parameters
    ----------
    N : int
        The number of (active) electrons.
    
    Returns
    -------
    init_function : function
        A function for initializing the Hartee-Fock state.

    """

    if mapping_type=='jordan_wigner':
        def init_function(qv):
            for i in range(N):
                x(qv[i])

    if mapping_type=='parity':
        def init_function(qv):
            for i in range(N//2):
                x(qv[2*i])

    return init_function


def electronic_structure_problem(one_int, two_int, M, N, K=None, L=None, mapping_type='jordan_wigner', ansatz_type='QCCSD', threshold=1e-4):
    r"""
    Creates a VQE problem instance for an electronic structure problem defined by the 
    one-electron and two-electron integrals for the spin orbitals (in physicists' notation).

    The problem Hamiltonian is given by:

    .. math::

        H = \sum\limits_{i,j=1}^{M}h_{i,j}a^{\dagger}_ia_j + \sum\limits_{i,j,k,l=1}^{M}h_{i,j,k,l}a^{\dagger}_ia^{\dagger}_ja_ka_l
    
    for one-electron integrals:

    .. math::

        h_{i,j}=\int\mathrm dx \chi_i^*(x)\chi_j(x)

    and two-electron integrals:

    .. math::

        h_{i,j,k,l} = \int\mathrm dx_1\mathrm dx_2\chi_i^*(x_1)\chi_j^*(x_2)\frac{1}{|r_1-r_2|}\chi_k(x_1)\chi_l(x_2)

    Parameters
    ----------
    int_one : numpy.ndarray
        The one-electron integrals w.r.t. spin orbitals (in physicists' notation).
    int_two : numpy.ndarray
        The two-electron integrals w.r.t. spin orbitals (in physicists' notation).
    M : int
        The number of spin orbitals.
    N : int
        The number of electrons.
    K : int, optional
        The number of active spin orbitals.
    L : int, optional
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
        data = electronic_data(mol)

        vqe = electronic_structure_problem(data['one_int'],data['two_int'],data['num_orb'],data['num_elec'])
        vqe.set_callback()

        energy = vqe.run(QuantumVariable(4),depth=1,max_iter=50,mes_kwargs={'method':'QWC'})
        print(energy)
        #Yields -1.8461290172512965
    
    """
    from qrisp.vqe import VQEProblem

    if K is None or L is None:
        K = M
        L = N

    ansatz, num_params = create_QCCSD_ansatz(K,L)

    return VQEProblem(create_electronic_hamiltonian(one_int,two_int,M,N,K,L,mapping_type=mapping_type,threshold=threshold), ansatz, num_params, init_function=create_hartree_fock_init_function(L,mapping_type))