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
def create_nonlocal_LCD_instance(Q):

    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi/2 * sp.sin(sp.pi*t/(2*T))**2)**2
        return lam_expr

    def alpha(lam):
        # lengthy expression for alpha with non-local AGP
        # from the FraunhoferGPT conversation:
        S_hR = sum([sum([J[i][j]**2 *(h[i]+h[j]) for i in range(j)]) for j in range(N)])
        S_hsqR = sum([sum([J[i][j]**2 *(h[i]**2+h[j]**2) for i in range(j)]) for j in range(N)])
        S_2 = sum([sum([J[i][j]**2 for i in range(j)]) for j in range(N)])
        S_4 = sum([sum([J[i][j]**4 for i in range(j)]) for j in range(N)])
        R_i_list = [sum([J[i][j]**2 if j!=i else 0 for j in range(N)])  for i in range(N)]
        S_Rsq = sum(R_i**2 for R_i in R_i_list)
        S_h = sum(h)
        S_hsq = sum(i**2 for i in h)
        c = 0.05
        dc = 0.05

        denom = 4 *(
                c**2 *(N + 4*S_2)
                +c*lam *(2*S_h + 4*S_hR + 12*S_2)
                +lam**2 *(S_hsq + 2*S_hsqR + 6*S_hR + 2*S_Rsq + 4*S_2 + 2*S_4)
                + (1-lam)**2 *N
                + 8 *(1-lam)**2 *S_2
        )

        nom = N*c + S_h + 2*S_2 + N*(1-lam)* dc

        alph = -nom/denom

        return alph

    # Initial Hamiltonian
    H_init = -1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = (sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) 
              + sum([h[i] * Z(i) for i in range(N)]))
    
    # Nonlocal AGP
    A_lam = -2 * (sum([h[i]*Y(i) for i in range(N)])
                  + sum([sum([J[i][j] * (Y(i) * Z(j) + Z(i) * Y(j)) for i in range(j)]) for j in range(N)]))

    return lam, alpha, H_init, H_prob, A_lam

# Create DCQO instance
lam, alpha, H_init, H_prob, A_lam = create_nonlocal_LCD_instance(Q)
nonlocal_prob = DCQOProblem(lam, alpha, H_init, H_prob, A_lam)

# Run COLD problem
qarg = QuantumVariable(size=Q.shape[0])
nonlocal_res = nonlocal_prob.run(qarg, N_steps=10, T=5, N_opt=1)

# Benchmark result
from qrisp.algorithms.cold.cold_benchmark import *
ar = approx_ratio(Q, nonlocal_res, solution)
sp = success_prob(nonlocal_res, solution)
most_likely_3 = most_likely_res(Q, nonlocal_res, N=3)

print(f'Approximation ratio: {ar}')
print(f'Success probability: {sp}')
print(f'3 most likely: {most_likely_3}')
