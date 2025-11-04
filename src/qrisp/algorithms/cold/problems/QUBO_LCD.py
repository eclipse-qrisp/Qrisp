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
def create_LCD_instance(Q):

    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi/2 * sp.sin(sp.pi*t/(2*T))**2)**2
        return lam_expr

    def alpha(lam):
        A = lam * h 
        B = 1 - lam
        C = h 

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

    return lam, alpha, H_init, H_prob, A_lam

# Create DCQO instance
lam, alpha, H_init, H_prob, A_lam = create_LCD_instance(Q)
LCD_prob = DCQOProblem(lam, alpha, H_init, H_prob, A_lam)

# Run LCD problem
qarg = QuantumVariable(size=Q.shape[0])
LCD_result = LCD_prob.run(qarg, N_steps=10, T=5)

# print(f'LCD result: \n{LCD_result}')
# print(f'Actual solution: {solution}')

# TODO:
from qrisp.algorithms.cold.cold_benchmark import *
ar = approx_ratio(Q, LCD_result, solution)
sp = success_prob(LCD_result, solution)
most_likely_3 = most_likely_res(Q, LCD_result, N=3)

print(f'Approximation ratio: {ar}')
print(f'Success probability: {sp}')
print(f'3 most likely: {most_likely_3}')
