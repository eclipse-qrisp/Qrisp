.. _IsingModel:

Transverse Field Ising Model
============================

.. currentmodule:: qrisp

In this example, we study Hamiltonian dynamics of the transverse field Ising model defined by the Hamiltonian

$$H = -J\\sum_{(i,j)\\in E}Z_iZ_j + B\\sum_{i\\in V}X_i$$

for a lattice graph $G=(V,E)$ and real parameters $J, B$. We investigate the total magnetization of the system as it evolves under the Hamiltonian.

Here, we consider an Ising chain.

::

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    def generate_chain_graph(N):
        coupling_list = [[k,k+1] for k in range(N-1)]
        G = nx.Graph()
        G.add_edges_from(coupling_list)
        return G

    G = generate_chain_graph(6)

We implement methods for creating the Ising Hamiltonian and the total magnetization observable for a given graph.

::

    from qrisp import QuantumVariable
    from qrisp.operators.pauli import X, Y, Z

    def create_ising_hamiltonian(G, J, B):
        H = sum(-J*Z(i)*Z(j) for (i,j) in G.edges()) + sum(B*X(i) for i in G.nodes())
        return H

    def create_magnetization(G):
        H = (1/G.number_of_nodes())*sum(Z(i) for i in G.nodes())
        return H

With all the necessary ingredients, we conduct the experiment: For varying evolution times $T$:

- Prepare the $\ket{0}^{\otimes N}$ state.

- Perform **Hamiltonian simulation** via Trotterization.

- Measure the total magnetization.

::

    T_values = np.arange(0, 2.0, 0.05)
    M_values = []

    M = create_magnetization(G)

    for T in T_values:
        H = create_ising_hamiltonian(G,1.0,1.0)
        U = H.trotterization()

        qv = QuantumVariable(G.number_of_nodes())
        U(qv,t=-T,steps=5)
        M_values.append(M.get_measurement(qv,precision=0.005))

Finally, we visualize the results. As expected, the total magnetization decreases in the presence of a transverse field with increasing evolution time $T$.

::

    import matplotlib.pyplot as plt
    plt.scatter(T_values, M_values, color='#6929C4', marker="o", linestyle='solid', s=10, label='Magnetization')
    plt.xlabel("Time", fontsize=15, color="#444444")
    plt.ylabel("Magnetization", fontsize=15, color="#444444")
    plt.legend(fontsize=12, labelcolor="#444444")
    plt.tick_params(axis='both', labelsize=12)
    plt.grid()
    plt.show()

.. figure:: /_static/Ising_chain_N=6.png
   :alt: Ising chain
   :align: center

