import numpy as np
import sympy as sp
from qrisp.algorithms.cold import DCQOProblem
from qrisp.operators.qubit import X, Y, Z
from qrisp import QuantumVariable
from qrisp.algorithms.cold.AGP_params import solve_alpha, solve_alpha_gamma_chi

# Define QUBO problem
Q = np.array([[-1.1, 0.6, 0.4, 0.0, 0.0, 0.0],
              [0.6, -0.9,  0.5, 0.0, 0.0, 0.0],
              [0.4, 0.5, -1.0, -0.6, 0.0, 0.0],
              [0.0, 0.0, -0.6, -0.5, 0.6, 0.0],
              [0.0, 0.0, 0.0, 0.6, -0.3, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, -0.4]])
solution = {'101101': -3.4}

# Create all methods and operators needed for the DCQO instance
def create_LCD_instance(Q, agp_type, uniform_AGP_coeffs=True):

    def build_agp(agp_type):
        
        def order1():
            def A_lam(alph):
                return sum([alph[i] * Y(i) for i in range(N)])
            return A_lam

        def order2():
            def A_lam(alph, gam, chi):
                A =  (sum([alph[i] * Y(i) for i in range(N)]) + 
                      sum([sum([gam[i]*X(i)*Y(j) + gam[j]*Y(i)*X(j) for j in range(i)]) for i in range(N)]) +
                      sum([sum([chi[i]*Z(i)*Y(j) + chi[j]*Y(i)*Z(j) for j in range(i)]) for i in range(N)]))
                return A
            return A_lam
        
        def nested_commutators():
            def A_lam(alph):
                A = -2 * sum([alph[i]*h[i]*Y(i) + 
                              alph[i]*sum([J[i][j]*(Y(i)*Z(j) + Z(i)*Y(j)) for j in range(i)]) for i in range(N)])
                return A
            return A_lam
        
        builders = {"order1": order1(),
                    "order2": order2(),
                    "nc": nested_commutators()}
        
        return builders[agp_type]

    def build_coeffs(agp_type, uniform_AGP_coeffs):

        def order1_uniform():
            def alpha(lam):
                A = lam * h 
                B = 1 - lam
                nom = np.sum(A + 4*B*h)
                denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
                alph = nom/denom
                alph = [[alph]*N]
                return alph
            return alpha

        def order1_nonuniform():
            def alpha(lam):
                denom = [2 * ((lam*h[i])**2 + (1-lam)**2 + 
                        lam**2 * sum([J[i][j] for j in range(N) if j != i])) 
                        for i in range(N)]
                alph = [[h[i]/denom[i] for i in range(N)]]
                return alph
            return alpha

        def order2(uniform):
            def params(lam):
                alpha, gamma, chi = solve_alpha_gamma_chi(h, J, lam, uniform=uniform)
                par = [alpha, gamma, chi]
                return par
            return params
        
        def nc_uniform():
            def alpha(lam):
                S_hR = sum([sum([J[i][j]**2 * (h[i]+h[j]) for i in range(j)]) for j in range(N)])
                S_hsqR = sum([sum([J[i][j]**2 *(h[i]**2+h[j]**2) for i in range(j)]) for j in range(N)])
                S_2 = sum([sum([J[i][j]**2 for i in range(j)]) for j in range(N)])
                S_4 = sum([sum([J[i][j]**4 for i in range(j)]) for j in range(N)])
                R_i_list = [sum([J[i][j]**2 if j!=i else 0 for j in range(N)]) for i in range(N)]
                S_Rsq = sum(R_i**2 for R_i in R_i_list)
                S_h = sum(h)
                S_hsq = sum(i**2 for i in h)

                nom = S_h + 2*S_2
                denom = 4*(lam**2*(S_hsq + 2*S_hsqR + 6*S_hR + 2*S_Rsq + 4*S_2 - 2*S_4)
                           + (1-lam)**2 * (N + 8*S_2))
                
                alph = -nom/denom
                alph = [N*[alph]]
                return alph
            return alpha
        
        def nc_nonuniform():
            def alpha(lam):
                alph = solve_alpha(h, J, lam)
                alph = [alph]
                return alph
            return alpha

        builders = {("order1", True): order1_uniform(),
                    ("order1", False): order1_nonuniform(),
                    ("order2", uniform_AGP_coeffs): order2(uniform_AGP_coeffs),
                    ("nc", True): nc_uniform(),
                    ("nc", False): nc_nonuniform()
                    }        

        return builders[(agp_type, uniform_AGP_coeffs)]


    N = len(Q[0])
    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    def lam():
        t, T = sp.symbols("t T", real=True)
        lam_expr = sp.sin(sp.pi/2 * sp.sin(sp.pi*t/(2*T))**2)**2
        return lam_expr

    # AGP coefficients
    coeff_func = build_coeffs(agp_type, uniform_AGP_coeffs)

    # Initial Hamiltonian
    H_init = -1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = (sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) 
              + sum([h[i] * Z(i) for i in range(N)]))
    
    # AGP
    A_lam = build_agp(agp_type)
    
    return lam, coeff_func, H_init, H_prob, A_lam

# Create DCQO instance for different scenarios
# for agp_type in ["order1", "order2", "nc"]:
#     for uniform in [True, False]:

#         lam, alpha, H_init, H_prob, A_lam = create_LCD_instance(Q, agp_type, uniform)
#         LCD_prob = DCQOProblem(lam, None, alpha, H_init, H_prob, A_lam)

#         # Run LCD problem
#         qarg = QuantumVariable(size=Q.shape[0])
#         LCD_result = LCD_prob.run(qarg, N_steps=10, T=5, method='LCD')

#         from qrisp.algorithms.cold.cold_benchmark import *
#         ar = approx_ratio(Q, LCD_result, solution)
#         sup = success_prob(LCD_result, solution)
#         most_likely_3 = most_likely_res(Q, LCD_result, N=3)

#         print(f'AGP type {agp_type}, uniform coeffs: {uniform}, N = {Q.shape[0]}')
#         print(f'Approximation ratio: {ar}')
#         print(f'Success probability: {sup}')
#         print(f'3 most likely: {most_likely_3}\n')

