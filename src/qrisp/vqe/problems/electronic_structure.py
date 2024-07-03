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
    H += (1/2)*sum(sum(sum(sum(two_int[i][j][k][l]*A(i)*A(j)*a(k)*a(l) for i in range(M)) for j in range(M)) for k in range(M)) for l in range(M))
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
    H += (1/2)*sum(sum(sum(sum(two_int[i][j][k][l]*B(i,M)*B(j,M)*b(k,M)*b(l,M) for i in range(M)) for j in range(M)) for k in range(M)) for l in range(M))
    H = simplify_spin(H)

    return H




def electronic_structure_problem(one_int, two_int):
    """


    """
    return 0