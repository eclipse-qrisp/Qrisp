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
from qrisp.misc.spin import *

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
def sp(x):
    return x//2

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

    n_spin_orbs = int_one.shape[0]

    # Initialize the spin-orbital one-electron integral tensor
    int_one_spin = np.zeros((n_spin_orbs, n_spin_orbs))
    for i in range(n_spin_orbs):
        for j in range(n_spin_orbs):
            int_one_spin[i][j] = delta(omega(i),omega(j))*int_one[sp(i)][sp(j)]

    # Initialize the spin-orbital two-electron integral tensor
    int_two_spin = np.zeros((n_spin_orbs, n_spin_orbs, n_spin_orbs, n_spin_orbs))
    for i in range(n_spin_orbs):
        for j in range(n_spin_orbs):
            for k in range(n_spin_orbs):
                for l in range (n_spin_orbs):
                    int_two_spin[i][j][k][l] = delta(omega(i),omega(j))*delta(omega(k),omega(l))*int_two[sp(i)][sp(j)][sp(k)][sp(l)]

    return int_one_spin, int_two_spin

#
#
#

# annihilation operator
def a(j):
    return sp.prod(Z(i) for i in range(j))*(X(j)+I*Y(j))/2

# creation operator
def A(j):
    return sp.prod(Z(i) for i in range(j))*(X(j)-I*Y(j))/2

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
    H += sum(sum(one_int[i][j]*A(i)*a(j) for i in range(M)) for j in range(M))
    H += -(1/2)*sum(sum(sum(sum(two_int[i][k][j][l]*A(i)*A(j)*a(k)*a(l) for i in range(M)) for j in range(M)) for k in range(M)) for l in range(M))
    H = simplify_spin(H)

    return H



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




def electronic_structure_problem(one_int, two_int):
    r"""
    Creates a VQE problem instance for a electronic structure problem (in terms of spin orbitals)
    defined by the one-electron and two-electron integrals for the spacial orbitals (in chemists' notation).

    The problem Hamiltonian is given by:

    .. math::

        H = \sum\limits_{i,j=1}^{M}h_{i,j}a^{\dagger}_ia_j + \sum\limits_{i,j,k,l=1}^{M}h_{i,j,k,l}a^{\dagger}_i\a^{\dagger}_ja_ka_l
    
    Here,

    .. math::

        h_{i,j}=\int\mathrm dx \chi_i^*(x)\chi_j(x)

    and

    .. math::

        h_{i,j,k,l} = \int\mathrm dx_1 \mathrm dx_2 
        
    
    """
    return 0