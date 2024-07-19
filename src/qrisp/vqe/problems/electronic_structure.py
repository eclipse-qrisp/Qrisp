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
from qrisp.misc.pauli_op import *

#
# helper functions
#

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
    data['two_int'] = two_int
    data['num_orb'] = 2*mf.mo_coeff.shape[0]  # Number of spin orbitals
    data['num_elec'] = mol.nelectron
    data['energy_nuc'] = mol.energy_nuc()
    data['energy_scf'] = energy_scf

    return data

#
# Fermion to qubit mappings
#

# annihilation operator
def a(j):
    return sp.prod(Z(i) for i in range(j))*(X(j)+I*Y(j))/2

# creation operator
def A(j):
    return sp.prod(Z(i) for i in range(j))*(X(j)-I*Y(j))/2


def c_jw(j):
    return PauliOperator({tuple([(i,"Z") for i in range(j)]+[(j,"X")]):0.5,tuple([(i,"Z") for i in range(j)]+[(j,"Y")]):-0.5j},0)

def a_jw(j):
    return PauliOperator({tuple([(i,"Z") for i in range(j)]+[(j,"X")]):0.5,tuple([(i,"Z") for i in range(j)]+[(j,"Y")]):0.5j},0)

def jordan_wigner(one_int, two_int):
    """

    Parameters
    ----------
    one_int : list[float]
        The one-body integrals.
    two_int : list[float]
        The two-body integrals.

    Returns
    -------
    H : SymPy expr
        The qubit Hamiltonian for the electronic structure problem.

    """
    M = one_int.shape[0]

    H = 0
    #H += sum(sum(one_int[i][j]*A(i)*a(j) for i in range(M)) for j in range(M))
    #H += -(1/2)*sum(sum(sum(sum(two_int[i][k][j][l]*A(i)*A(j)*a(k)*a(l) for i in range(M)) for j in range(M)) for k in range(M)) for l in range(M))


    #H += sum(sum(simplify_spin(one_int[i][j]*A(i)*a(j)) for i in range(M)) for j in range(M))
    #H += -(1/2)*sum(sum(sum(sum(simplify_spin(two_int[i][k][j][l]*A(i)*A(j)*a(k)*a(l)) for i in range(M)) for j in range(M)) for k in range(M)) for l in range(M))

    """
    H = PauliOperator()
    for i in range(M):
        for j in range(M):
            H.add( PauliOperator.from_expr(one_int[i][j]*A(i)*a(j)))

    for i in range(M):
        for j in range(M): 
            for k in range(M):
                for l in range(M):
                    H.add(PauliOperator.from_expr(-(1/2)*two_int[i][k][j][l]*A(i)*A(j)*a(k)*a(l)))
    

    """

    H = PauliOperator()
    for i in range(M):
        for j in range(M):
            H.add( c_jw(i)*a_jw(j).scalar_mul(one_int[i][j]) )
    
    for i in range(M):
        for j in range(M): 
            for k in range(M):
                for l in range(M):
                    H.add( c_jw(i)*c_jw(j)*a_jw(k)*a_jw(l).scalar_mul(-0.5*two_int[i][k][j][l]) )
    

    print("Hamiltonian defined")

    return H
    #H = simplify_spin(H)

    #return H

# annihilation operator
def b(j,M):
    if j>0:
        return (Z(j-1)*X(j)+I*Y(j))/2*sp.prod(X(i) for i in range(j+1,M))
    else:
        return (X(j)+I*Y(j))/2*sp.prod(X(i) for i in range(j+1,M))


# creation operator
def B(j,M):
    if j>0:
        return (Z(j-1)*X(j)-I*Y(j))/2*sp.prod(X(i) for i in range(j+1,M))
    else:
        return (X(j)-I*Y(j))/2*sp.prod(X(i) for i in range(j+1,M))
    

def parity(one_int, two_int):
    """

    Parameters
    ----------
    one_int : list[float]
        The one-body integrals.
    two_int : list[float]
        The two-body integrals.

    Returns
    -------
    H : SymPy expr
        The qubit Hamiltonian for the electronic structure problem.

    """

    # number of electrons
    M = one_int.shape[0]

    H = 0
    H += sum(sum(one_int[i][j]*B(i,M)*b(j,M) for i in range(M)) for j in range(M))
    H += -(1/2)*sum(sum(sum(sum(two_int[i][k][j][l]*B(i,M)*B(j,M)*b(k,M)*b(l,M) for i in range(M)) for j in range(M)) for k in range(M)) for l in range(M))
    H = simplify_spin(H)

    return H

#
# Hamiltonian
#

def create_electronic_hamiltonian(one_int, two_int, M, N, mapping_type='jordan_wigner'):
    """
    Creates the qubit Hamiltonian for an electronic structure problem defined by the 
    one-electron and two-electron integrals for the spin orbitals (in chemists' notation).
    
    Parameters
    ----------
    int_one : numpy.ndarray
        The one-electron integrals w.r.t. spacial orbitals.
    int_two : numpy.ndarray
        The two-electron integrals w.r.t. spacial orbitals.
    M : int
        The number of spin orbitals.
    N : int
        The number of electrons.
    mapping_type : string, optinal
        The mapping from the fermionic Hamiltonian to the qubit Hamiltonian. Available are ``jordan_wigner``, ``parity``.
        The default is ``jordan_wigner``.

    Returns
    -------
    H : sympr.Expr
        THe electronic Hamiltonian.
    
    """

    H = jordan_wigner(one_int,two_int)

    return H

#
# Ansatz
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
    """
    This method implements the `QCCSD ansatz <https://arxiv.org/abs/2005.08451>`_.

    Parameters
    ----------
    M : int
        The number of spin orbitals.
    N : int
        The number of electrons.

    Returns
    -------
    ansatz : function

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

def create_hartree_fock_init_function(N):
    """
    
    
    """

    def init_function(qv):
        for i in range(N):
            x(qv[i])

    return init_function


def electronic_structure_problem(one_int, two_int, M, N, mapping_type='jordan_wigner', ansatz_type='QCCSD'):
    r"""
    Creates a VQE problem instance for an electronic structure problem defined by the 
    one-electron and two-electron integrals for the spin orbitals (in chemists' notation).

    The problem Hamiltonian is given by:

    .. math::

        H = \sum\limits_{i,j=1}^{M}h_{i,j}a^{\dagger}_ia_j + \sum\limits_{i,j,k,l=1}^{M}h_{i,j,k,l}a^{\dagger}_i\a^{\dagger}_ja_ka_l
    
    for one-electron integrals:

    .. math::

        h_{i,j}=\int\mathrm dx \chi_i^*(x)\chi_j(x)

    and two-electron integrals:

    .. math::

        h_{i,j,k,l} = \int\mathrm dx_1 \mathrm dx_2 

    Parameters
    ----------
    int_one : numpy.ndarray
        The one-electron integrals w.r.t. spacial orbitals.
    int_two : numpy.ndarray
        The two-electron integrals w.r.t. spacial orbitals.
    M : int
        The number of spin orbitals.
    N : int
        The number of electrons.
    mapping_type : string, optinal
        The mapping from fermionic Hamiltonian to qubit Hamiltonian. Available are ``jordan_wigner``, ``parity``.
        The default is ``jordan_wigner``.
    ansatz_type : string, optional
        The ansatz type.

    Returns
    -------
    VQEProblem
        The VQE problem instance.

    Examples
    --------

    We denonstrate how to use pyscf to obtain the 
    
    
    """
    from qrisp.vqe import VQEProblem

    ansatz, num_params = create_QCCSD_ansatz(M,N)

    return VQEProblem(create_electronic_hamiltonian(one_int,two_int,M,N,), ansatz, num_params, init_function=create_hartree_fock_init_function(N))