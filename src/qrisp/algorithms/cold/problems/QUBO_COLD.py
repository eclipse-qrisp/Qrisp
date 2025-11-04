import numpy as np
import sympy as sp
from qrisp.algorithms.cold import DCQOProblem
from qrisp.operators.qubit import X, Y, Z
from qrisp import QuantumVariable


# Define QUBO problem
Q = np.array([[-1.1, 0.6, 0.4, 0.0, 0.0, 0.0],
              [0.6, -0.9,  0.5, 0.0, 0.0, 0.0],
              [0.4, 0.5, -1.0, -0.6, 0.0, 0.0],
              [0.0, 0.0, -0.6, -0.5, 0.6, 0.0],
              [0.0, 0.0, 0.0, 0.6, -0.3, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, -0.4]])
solution = {'101101': -3.4}

# Create all methods and operators needed for the DCQO instance
def create_COLD_instance(Q):

    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi/2 * sp.sin(sp.pi*t/(2*T))**2)**2
        return lam_expr

    def alpha(lam, f, f_deriv):
        A = lam * h + f
        B = 1 - lam
        C = h + f_deriv

        nom = np.sum(A + 4*B*C)
        denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
        alph = nom/denom

        return alph

    # Initial Hamiltonian
    H_init = -1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = (sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) 
              + sum([h[i] * Z(i) for i in range(N)]))
    
    # AGP
    A_lam = sum([Y(i) for i in range(N)])

    # Control Hamiltonian
    H_control = sum([Z(i) for i in range(N)])

    return lam, alpha, H_init, H_prob, A_lam, H_control

# Create DCQO instance
lam, alpha, H_init, H_prob, A_lam, H_control = create_COLD_instance(Q)
COLD_prob = DCQOProblem(lam, alpha, H_init, H_prob, A_lam, H_control)

# Run COLD problem
qarg = QuantumVariable(size=Q.shape[0])
COLD_result = COLD_prob.run(qarg, N_steps=10, T=5, N_opt=1)

# Benchmark result
from qrisp.algorithms.cold.cold_benchmark import *
ar = approx_ratio(Q, COLD_result, solution)
sp = success_prob(COLD_result, solution)
most_likely_3 = most_likely_res(Q, COLD_result, N=3)

print(f'Approximation ratio: {ar}')
print(f'Success probability: {sp}')
print(f'3 most likely: {most_likely_3}')
