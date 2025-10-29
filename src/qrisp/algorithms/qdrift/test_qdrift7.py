#%%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import timer 

from qrisp import QuantumVariable
from qrisp.operators import X, Y, Z, QubitOperator
from qrisp import *


def sample(probs):
    return np.random.choice(range(len(probs)), p=probs)


def generate_chain_graph(N):
    G = nx.Graph()
    G.add_edges_from([[k,k+1] for k in range(N-1)])
    return G

def create_ising_hamiltonian(G, J, B):
    H = sum(-J*Z(i)*Z(j) for (i,j) in G.edges()) + sum(B*X(i) for i in G.nodes())
    return H

def create_magnetization(G):
    H = (1.0/G.number_of_nodes())*sum(Z(i) for i in G.nodes())
    return H

def qdrift(H, qv, time_simulation, N, use_arctan= False ):

    coeffs = []
    terms = []
    # signs = []

    coeffs = []
    terms = []
    for term, coeff in  H.terms_dict.items():
        terms.append(term)
        coeffs.append(coeff)
    # signs = np.sign(x for x in coeffs)

    # Step 1
    normalisation_factor = sum(abs(c) for c in coeffs)
    if normalisation_factor == 0 or time_simulation == 0:
        return []  

    # Step 3
    tau = normalisation_factor * time_simulation / N
    angle = np.arctan(tau) if use_arctan else tau

    # Step 4
    probs = [abs(c) / normalisation_factor for c in coeffs]

    # Step 5
    i = 0

    while i < N:
        i += 1  # Step 5a
        j = sample(probs)  # Step 5b
        # theta = signs[j] * angle 
        terms[j].simulate(angle, qv)  # Step 5c
    #Step 6
    return qv

T_values = np.arange(0, 2, 0.05)

N=1000
precision_expectation_value = 0.001
G=generate_chain_graph(6)
H = create_ising_hamiltonian(G, J=1.0, B=1.0)
M  = create_magnetization(G)
M_values = []

def psi(t, use_arctan=True,):
    qv = QuantumVariable(G.number_of_nodes())
    qdrift(H, qv, time_simulation=t, N = N, use_arctan=use_arctan)
    return qv  

for t in T_values:

    ev_M = M.expectation_value(psi, precision_expectation_value)  
    print(f"t={t}, <M>={ev_M(t)}")
    magnetization = float(ev_M(t))  
    print(f"Magnetization at time {t}: {magnetization}")
    M_values.append(magnetization)

plt.scatter(T_values, M_values, color='#6929C4', marker="o", linestyle="solid", s=20, label=r"Ising chain")
plt.xlabel(r"Evolution time $T$", fontsize=15, color="#444444")
plt.ylabel(r"Magnetization $\langle M \rangle$", fontsize=15, color="#444444")
plt.legend(fontsize=15, labelcolor="#444444")
plt.tick_params(axis='both', labelsize=12)
plt.grid()
plt.show()

#%%