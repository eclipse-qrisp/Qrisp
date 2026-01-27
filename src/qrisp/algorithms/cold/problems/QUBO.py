"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
********************************************************************************
"""

# from qrisp import *
import numpy as np
import sympy as sp
from qrisp.core import QuantumVariable
from qrisp.operators.qubit import X, Y, Z
from qrisp.algorithms.cold import DCQOProblem, solve_alpha_gamma_chi, solve_alpha


def create_COLD_instance(Q, uniform_AGP_coeffs):
    """
    Create the necessary parameters and operators to initialize a DCQO problem instance for COLD.

    Parameters
    ----------
    Q : np.array
        The QUBO Matrix to be encoded in the Hamiltonian.
    uniform_AGP_coeffs : bool
        Whether to approximate the AGP with uniform or non-uniform coefficients.

    Returns
    -------
    collected operators : tuple
        Tuple containing the following functions and operators: 
        Scheduling function (lam(t)), function for AGP coefficients (alpha), initial and problem
        Hamiltonian (H_init, H_prob), AGP function (A_lam), Coupling and onsite energies of 
        problem Hamiltonian (J, h), inverse scheduling function (g(lam)), control Hamiltonian (H_control).
    """

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

    collected_operators = (lam, alpha, H_init, H_prob, A_lam, Q, g, H_control)

    return collected_operators


def create_LCD_instance(Q, agp_type, uniform_AGP_coeffs=True):
    """
    Create the necessary parameters and operators to initialize a DCQO problem instance for LCD.

    Parameters
    ----------
    Q : np.array
        The QUBO Matrix to be encoded in the Hamiltonian.
    agp_type : str
        Which approximation of the AGP to use. Can choose between ``order1``, 
        ``order2``, ``nc`` (nested commutators up to first order).
    uniform_AGP_coeffs : bool
        Whether to approximate the AGP with uniform or non-uniform coefficients.

    Returns
    -------
    collected operators : tuple
        Tuple containing the following functions and operators: 
        Scheduling function (lam(t)), function for AGP coefficients (alpha), initial and problem
        Hamiltonian (H_init, H_prob), AGP function (A_lam), Coupling and onsite energies of 
        problem Hamiltonian (J, h), inverse scheduling function (g(lam)), control Hamiltonian (H_control).
    """

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
    
    return lam, coeff_func, H_init, H_prob, A_lam, Q


def solve_QUBO(Q: np.array, problem_args: dict, run_args: dict):
    """
    Solves a QUBO Matrix using counterdiabatic driving. 

    Example usage:
    Q = np.array([
        [-1.0, 0.5, 0.4, 0.0], 
        [0.5, -0.9, 0.6, 0.0], 
        [0.4, 0.6, -0.8, -0.5], 
        [0.0, 0.0, -0.5, 0.2]
        ])

    problem_args = {"method": "COLD", "uniform": False}
    run_args = {"N_steps": 6, "T": 1, "N_opt": 1, "CRAB": False, "objective": "exp_value", "bounds": (-2, 2)}

    result = solve_QUBO(Q, problem_args, run_args)
    
    Q : np.array
        QUBO Matrix to solve.
    problem_args : dict
        Holds arguments for DCQO problem creation (method: str (COLD/LCD), uniform: bool).
    run_args : dict
        Holds arguments for running the DCQO instance (N_steps, T, N_opt, CRAB, optimizer, objective, bounds).
        For all options, see :ref: `DCQOProblam`.

    """

    method = problem_args["method"]


    if method == "LCD":
        problem_operators = create_LCD_instance(
            Q, agp_type=problem_args["agp_type"], uniform_AGP_coeffs=problem_args["uniform"]
        )
        
    elif method == "COLD":
        problem_operators = create_COLD_instance(
            Q, uniform_AGP_coeffs=problem_args["uniform"]
        )

    # Create qarg and problem instrance
    qarg = QuantumVariable(Q.shape[0])
    prob = DCQOProblem(*problem_operators)

    # Run problem
    result = prob.run(qarg, method=method, **run_args)

    return result

