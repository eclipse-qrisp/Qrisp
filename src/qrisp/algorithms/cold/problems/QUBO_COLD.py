import numpy as np
import sympy as sp
from qrisp.algorithms.cold import DCQOProblem
from qrisp.operators.qubit import X, Y, Z
from qrisp import QuantumVariable


# Create all methods and operators needed for the DCQO instance
def create_COLD_instance(Q, uniform_AGP_coeffs):

    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = t/T
        return lam_expr
    
    def g():
        # Inverse of lam(t) giving t(lam)
        lam, T = sp.symbols("lam T")
        g_expr = lam * T
        return g_expr

    # AGP coefficients            
    if uniform_AGP_coeffs:
        def alpha(lam, f, f_deriv):
            A = lam * h + f
            B = 1 - lam
            C = h + f_deriv

            nom = np.sum(A + 4*B*C)
            denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
            alph = nom/denom
            alph = [alph]*N

            return alph
    else:
        def alpha(lam, f, f_deriv):
            nom = [h[i] + f + (1-lam) * f_deriv 
                for i in range(N)]
            denom = [2 * ((lam*h[i] + f)**2 + (1-lam)**2 + 
                    lam**2 * sum([J[i][j] for j in range(N) if j != i])) 
                    for i in range(N)]

            alph = [nom[i]/denom[i] for i in range(N)]
            return alph


    # Initial Hamiltonian
    H_init = -1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = (sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) 
              + sum([h[i] * Z(i) for i in range(N)]))
    
    # AGP as function of alpha
    def A_lam(alph):
        return sum([alph[i] * Y(i) for i in range(N)])

    # Control Hamiltonian
    H_control = sum([Z(i) for i in range(N)])

    return lam, g, alpha, H_init, H_prob, A_lam, H_control

