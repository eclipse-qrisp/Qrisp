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

def create_heisenberg_hamiltonian(G, J, B):
    """
    This method creates the Hamiltonian for the Heisenberg model.

    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    J : float
        The positive coupling constant.
    B : float
        The magnetic field strength.

    Returns
    -------
    hamiltonian : sympy.Expr
        The quantum Hamiltonian.

    """

    hamiltonian = sum(J*X(i)*X(j)+Y(i)*Y(j)+Z(i)*Z(j) for (i,j) in G.edges()) + sum(B*Z(i) for i in G.nodes)
    return hamiltonian


def create_heisenberg_ansatz(G, J, B, M, C, ansatz_type="per hamiltonian"):
    """
    This method creates a function for applying one layer of the ansatz.

    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    J : float
        The positive coupling constant.
    B : float
        The magnetic field strength.
    M : list
        A list of edges corresponding to a maximal matching of ``G``.
    C : list
        An edge coloring of the graph ``G`` given by a list of lists of edges.
    ansatz_type : string, optional
        Specifies the Hamiltonian Variational Ansatz. Available are ``per hamiltonian``, ``per edge color``, ``per edge``.
        The default is ``per hamiltonian``.

    Returns
    -------

    ansatz : function 
        A function that can be applied to a ``QuantumVariable`` and a list of parameters.
    
    """

    # per hamiltonian
    def ansatz(qv,theta):
        # apply H_0
        for (i,j) in M:
            heis(theta[0],qv[i],qv[j])
        
        # apply H
        rz(B*theta[1],qv)

        for edges in C:
            for (i,j) in edges:
                heis(J*theta[1],qv[i],qv[j])

    #per edge color
    def ansatz_per_edge_color(qv,theta):
        # apply H_0
        for (i,j) in M:
            heis(theta[0],qv[i],qv[j])
        
        # apply H
        rz(B*theta[1],qv)

        count = 0
        for edges in C:
            for (i,j) in edges:
                heis(J*theta[2+count],qv[i],qv[j])
            count += 1

    #per edge
    def ansatz_per_edge(qv,theta):
        # apply H_0
        for (i,j) in M:
            heis(theta[0],qv[i],qv[j])
        
        # apply H
        rz(B*theta[1],qv)

        count = 0
        for edges in C:
            for (i,j) in edges:
                heis(J*theta[2+count],qv[i],qv[j])
                count += 1

    if ansatz_type=="per edge color":
        return ansatz_per_edge_color
    if ansatz_type=="per edge":
        return ansatz_per_edge
    return ansatz


def create_heisenberg_init_function(M):
    """
    Creates the function that, when applied to a ``QuantumVariable``, initializes a tensor product of singlet sates corresponding to a given matching.

    Parameters
    ----------
    M : list
        A list of edges corresponding to a maximal matching of ``G``.

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


def heisenberg_problem(G, J, B, ansatz_type="per hamiltonian"):
    r"""
    Creates a VQE problem instance for an isotropic Heisenberg model defined by a graph $G=(V,E)$,
    the coupling constant $J>0$ (antiferromagnetic), and the magnetic field strength $B$. The model Hamiltonian is given by:

    .. math::

        H = J\sum\limits_{(i,j)\in E}(X_iX_j+Y_iY_j+Z_iZ_j)+B\sum\limits_{i\in V}Z_i

    Each Hamiltonian $H_{i,j}=X_iX_j+Y_iY_j+Z_iZ_j$ has three eigenvectors for eigenvalue $+1$ (triplet states):

    * $\ket{00}$
    * $\ket{11}$
    * $\frac{1}{\sqrt{2}}\left(\ket{10}+\ket{01}\right)$  

    and one eigenvector for eigenvalue $-3$ (singlet state):

    * $\frac{1}{\sqrt{2}}\left(\ket{10}-\ket{01}\right)$ 

    For the problem specific VQE ansatz, we choose a Hamiltonian Variational Ansatz as proposed `here <https://arxiv.org/abs/2108.08086>`_.
    This ansatz is inspired by the adiabatic theorem of quantum mechanics: A system is prepared in the ground state of 
    an initial Hamiltonian $H_0$ and then slowly evolved under a time-dependet Hamiltoniam $H(t)$. Here, we set

    .. math::

        H(t) = \left(1-\frac{t}{T}\right)H_0 + \frac{t}{T}H

    where

    .. math::

        H_0 = \sum\limits_{(i,j)\in M}(X_iX_j+Y_iY_j+Z_iZ_j)

    for a maximal matching $M\subset E$ of the graph $G$. 

    For $J>0$ the ground state of the initial Hamiltonian $H_0$ is given by a tensor product of
    singlet states corresponding to the maximal matching $M$.

    The time evolution of $H(t)$ is approximately implemented by trotterization, i.e., alternatingly applying 
    $e^{-iH_0\Delta t}$ and $e^{-iH\Delta t}$, and if necessary trotterizing $e^{-iH\Delta t}$.

    In the scope of VQE, the short evolution times $\Delta t$ are replaced by parameters $\theta_i$ which are then optimized.
    This yields the following unitary ansatz with $p$ layers:

    .. math::

        U(\theta) = \prod_{l=1}^{p}e^{-i\theta_{l,0}H_0}e^{-i\theta_{l,1}H}

    The unitary $e^{-i\theta H}$ trotterized by:

    .. math::

        U_H(\theta) = e^{-i\theta H_B}\prod\limits_{k=1}^{q}\prod_{(i,j)\in E_k}e^{-i\theta H_{ij}}

    where $E_1,\dotsc,E_q$ is an edge coloring of the graph $G$, and $H_B$ is the magnetic field Hamiltonian.
    Then all unitaries $e^{-i\theta H_{ij}}$ for $(i,j)\in E_k$ commute. 
    This ansatz can be further generalized by introducing parameters

    * per edge color (one parameter for each color)
    * per edge (one parameter for each edge)

    in the unitary $U_H(\theta)$.

    Parameters
    ----------
    G : nx.Graph
        The graph defining the lattice.
    J : float
        The positive coupling constant.
    B : float
        fhe magnetic field strength.
    ansatz_type : string, optional
        Specifies the Hamiltonian Variational Ansatz. Available are ``per hamiltonian``, ``per edge color``, ``per edge``.
        The default is ``per hamiltonian``.

    Returns
    -------
    VQEProblem
        VQE problem instance for a specific isotropic Heisenberg model.

    Examples
    --------

    ::

        import networkx as nx
        import matplotlib.pyplot as plt

        # Create a graph
        coupling_list = [(0,1),(1,2),(2,3),(0,3)]
        G = nx.Graph()
        G.add_edges_from(coupling_list)

        # Draw the graph with labels
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        plt.show()

    .. figure:: /_static/heisenberg_lattice.png
        :scale: 80%
        :align: center

    ::

        from qrisp.vqe.problems.heisenberg import *

        vqe = heisenberg_problem(G,1,1)
        vqe.set_callback()
        energy = vqe.run(QuantumVariable(G.number_of_nodes()),depth=2,max_iter=50)
        print(energy)
        # Yields -8.061600000000002
    
    We visualize the optimization process:

    >>> vqe.visualize_energy(exact=True)

    .. figure:: /_static/heisenberg_energy.png
        :scale: 80%
        :align: center  

    """        
    from qrisp.vqe import VQEProblem

    M = nx.maximal_matching(G)
    C = greedy_edge_coloring(G)
    num_params = 2
    if ansatz_type=="per edge color":
        num_params = 2+len(C)
    if ansatz_type=="per edge":
        num_params = 2+G.number_of_edges()

    return VQEProblem(create_heisenberg_hamiltonian(G,J,B), create_heisenberg_ansatz(G,J,B,M,C, ansatz_type=ansatz_type), num_params=num_params, init_function=create_heisenberg_init_function(M))