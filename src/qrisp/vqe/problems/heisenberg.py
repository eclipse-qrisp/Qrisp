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

from qrisp import *
from qrisp.misc.spin import X, Y, Z
import networkx as nx


def greedy_edge_coloring(G):
    """
    This methods computes an edge coloring of a given graph.

    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    
    Returns
    -------
    edge_coloring : list
        An edge coloring of the graph.

    """

    edge_coloring = []

    G = G.copy()
    while G.number_of_edges()>0:
        M = nx.maximal_matching(G)
        edge_coloring.append(M)
        G.remove_edges_from(M)
    
    return edge_coloring


# sing gate corresponding to the singlet state $\ket{10}-\ket{01} of two qubits.
def sing(a,b):
    x(a)
    h(a)
    x(b)
    cx(a,b)

#  heis gate corresponding to the unitary exp(-i*theta*(XX+YY+ZZ)).
def heis(theta,a,b):
    # change of basis
    cx(a,b)
    h(a)

    cx(a,b)
    rz(-theta,b)
    cx(a,b)
    rz(theta,a)
    rz(theta,b)

    # change of basis
    h(a)
    cx(a,b)

def create_heisenberg_spin_operator(G, J, B):
    """
    This method creates the spin operator for the Heisenberg model.

    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    J : Float
        The coupling constant.
    B : FLoat
        The magnetic field strength.

    Returns
    -------
    spin_op : SymPy expression
        The spin operator.

    """

    spin_op = sum(J*X(i)*X(j)+Y(i)*Y(j)+Z(i)*Z(j) for (i,j) in G.edges())
    spin_op += sum(B*Z(i) for i in G.nodes)

    return spin_op


def create_heisenberg_ansatz(G, J, B, M):
    """
    This method creates a function for applying the ansatz.

    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    J : Float
        The coupling constant.
    B : FLoat
        The magnetic field strength.
    M : list
        A list of edges corresponting to a maximal matching of ``G``.

    Returns
    -------

    ansatz : function 
        A function that can be applied to a ``QuantumVariable`` and a list of parameters.
    
    """

    def ansatz(qv,theta):
        # apply H_0
        for (i,j) in M:
            heis(theta[0],qv[i],qv[j])
        
        # apply H_T
        rz(B*theta[1],qv)

        edge_coloring = greedy_edge_coloring(G)
        for edges in edge_coloring:
            for (i,j) in edges:
                heis(J*theta[1],qv[i],qv[j])

    return ansatz


def create_heisenberg_init_function(M):
    """
    Creates the function that, when applied to a ``QuantumVariable``, initializes a tensor product of singlet sates corresponding to a matching.

    Parameters
    ----------
    M : list
        A list of edges corresponding to a maximal matching.

    Returns
    -------
    init_function : function 
        A function that can be applied to a ``QuantumVariable``.

    """

    def init_function(qv):

        # tensor product of singlet states
        for (i,j) in M:
            sing(qv[i],qv[j])

    return init_function


def heisenberg_problem(G, J, B):
    """
    Creates a VQE problem instance for a Heisenberg model defined by a graph $G=(V,E)$,
    the coupling constant $J>0$, and the magnetic field strength $B$. The model Hamiltonian is given by:

    .. math::

        H = J\sum\limits_{(i,j)\in E}(X_iX_j+Y_iY_j+Z_iZ_j)+B\sum\limits_{i\in V}Z_i

    Each term $H_{i,j}=X_iX_j+Y_iY_j+Z_iZ_j$ has the eigenvectors (triplet states) $\ket{00}$, $\ket{11}$,
    $\frac{1}{\sqrt{2}}\left(\ket{10}+\ket{01}\right)$ for eigenvalue $+1$, and the eigenvector (singlet state)
    $\frac{1}{\sqrt{2}}\left(\ket{10}+\ket{01}\right)$ for eigenvalue $-3$.

    Hamiltonian Variational Ansatz 

    .. math::

        H(t) = \left(1-\frac{t}T}\right)H_0 + \frac{t}{T}H,

    where

    .. math::

        H_0 = \sum\limits_{(i,j)\in M}(X_iX_j+Y_iY_j+Z_iZ_j)

    for a maximal matching $M\subset E$ of the graph $G$. 

    In this case, the ground state of the initial Hamiltonian $H_0$ is given by a tensor product of the 
    singlet sates corresponding to the maximal matching $M$.

    The time evolution of $H(t)$ is approximately implemented by trotterization, i.e., applying 
    $e^{-iH_0\Delta t}$ and $e^{-iH\Delta t}$, and also if necessary trotterized $e^{-iH\Delta t}$.

    This yields the unitary ansatz with $p$ layers:

    .. math::

        U(\theta) = \prod_{i=1}^{p}e^{-iH_0\theta_{i,0}}e^{-iH\theta_{i,1}}

    If necessary, the unitary $e^{-iH\Delta t}$ trotterized by:

    .. math::

        e^{-iH\Delta t} = \prod\limits_{k=1}^{q}\prod_{(i,j)\in E_k}e^{-iH_{ij}\Delta t},

    where $E_1,\dotsc,E_q$ is an edge coloing of the graph $G$.



    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    J : Float
        The coupling constant.
    B : FLoat
        The magnetic field strength.

    Returns
    -------
    VQEProblem : function
        VQE problem instance for Heisenberg model.

    """        
    from qrisp.vqe import VQEProblem

    M = nx.maximal_matching(G)

    return VQEProblem(create_heisenberg_spin_operator(G,J,B), create_heisenberg_ansatz(G,J,B,M), num_params=2, init_function=create_heisenberg_init_function(M))