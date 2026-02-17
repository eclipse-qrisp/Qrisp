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
    H_init = 1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = (sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) 
              + sum([h[i] * Z(i) for i in range(N)]))
    
    # AGP as function of alpha
    if uniform_AGP_coeffs:
        A_lam = sum([Y(i) for i in range(N)])
    else:
        A_lam = [Y(i) for i in range(N)]

    # Control Hamiltonian
    H_control = sum([Z(i) for i in range(N)])

    collected_operators = (Q, H_init, H_prob, A_lam, alpha, lam, g, H_control)

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

    def build_agp(agp_type, J, h):
        
        def order1():
            A_lam = [Y(i) for i in range(N)]
            return A_lam

        def nested_commutators(J, h):
            A_lam = -2 * [h[i]*Y(i) + sum([J[i][j]*(Y(i)*Z(j) + Z(i)*Y(j)) for j in range(i)]) for i in range(N)]
            return A_lam
        
        builders = {"order1": order1(),
                    "nc": nested_commutators(J, h)}
        
        return builders[agp_type]

    def build_coeffs(agp_type, uniform_AGP_coeffs, J, h):

        def order1_uniform(J, h):
            def alpha(lam):
                A = lam * h 
                B = 1 - lam
                nom = np.sum(A + 4*B*h)
                denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
                alph = nom/denom
                alph = [alph]*N
                return alph
            return alpha

        def order1_nonuniform(J, h):
            def alpha(lam):
                denom = [2 * ((lam*h[i])**2 + (1-lam)**2 + 
                        lam**2 * sum([J[i][j] for j in range(N) if j != i])) 
                        for i in range(N)]
                alph = [h[i]/denom[i] for i in range(N)]
                return alph
            return alpha

        def nc_uniform(J, h):
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
        
        def nc_nonuniform(J, h):
            def alpha(lam):
                alph = [solve_alpha(h, J, lam)]
                return alph
            return alpha


        builders = {("order1", True): order1_uniform(J, h),
                    ("order1", False): order1_nonuniform(J, h),
                    ("nc", True): nc_uniform(J, h),
                    ("nc", False): nc_nonuniform(J, h)
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
    coeff_func = build_coeffs(agp_type, uniform_AGP_coeffs, J, h)

    # Initial Hamiltonian
    H_init = 1 * sum([X(i) for i in range(N)])

    # Problem Hamiltonian
    H_prob = (sum([sum([J[i][j] * Z(i) * Z(j) for j in range(i)]) for i in range(N)]) 
              + sum([h[i] * Z(i) for i in range(N)]))
    
    # AGP
    A_lam = build_agp(agp_type, J, h)
    
    return Q, H_init, H_prob, A_lam, coeff_func, lam


def solve_QUBO(Q: np.array, problem_args: dict, run_args: dict):
    """
    Solves a QUBO Matrix using counterdiabatic driving. This method uses the pre-defined COLD/LCD operators
    (hamiltonian, scheduling function, AGP parameters) as described in the tutorial. 
    To define your own operators, create a DCQO instance and use the ``run`` method.

    Parameters
    ----------
    
    Q : np.array
        QUBO Matrix to solve.
    problem_args : dict
        Holds arguments for DCQO problem creation (``method``: str ("COLD"/"LCD"), ``uniform``: bool).
    run_args : dict
        Holds arguments for running the DCQO instance (``N_steps``, ``T``, ``N_opt``, ``CRAB``).
        For all options, see :meth:`DCQOProblem.run`.

    Returns
    -------

    result : dict
        The dictionary holding the QUBO vector results, with their probabilitites and cost.
        They are ordered from most to least likely and the dictionary entries are {"state": [prob, cost]}.

    
    Examples
    --------

    ::

        import numpy as np
        from qrisp.algorithms.cold import solve_QUBO

        Q = np.array([[-1.1, 0.6, 0.4, 0.0, 0.0, 0.0],
                    [0.6, -0.9,  0.5, 0.0, 0.0, 0.0],
                    [0.4, 0.5, -1.0, -0.6, 0.0, 0.0],
                    [0.0, 0.0, -0.6, -0.5, 0.6, 0.0],
                    [0.0, 0.0, 0.0, 0.6, -0.3, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.5, -0.4]])

        problem_args = {"method": "COLD", "uniform": False}

        run_args = {"N_steps": 4, "T": 8, "N_opt": 1, "CRAB": False, 
                    "objective": "agp_coeff_magnitude", "bounds": (-3, 3)}

        result = solve_QUBO(Q, problem_args, run_args)

        print(result)

    ::

        {'101101': [0.2749327493274933, np.float64(-3.4)], '011101': [0.14747147471474714, np.float64(-3.0)], '111101': [0.10500105001050011, np.float64(-2.1)], '101100': [0.0888608886088861, np.float64(-3.0)], '011100': [0.04720047200472005, np.float64(-2.6)], '101110': [0.043830438304383046, np.float64(-2.1)], '111100': [0.035760357603576036, np.float64(-1.7000000000000002)], '011110': [0.024410244102441025, np.float64(-1.7)], '100101': [0.02137021370213702, np.float64(-2.0)], '110101': [0.016960169601696017, np.float64(-1.7000000000000002)], '111110': [0.01686016860168602, np.float64(-0.8)], '111001': [0.016390163901639016, np.float64(-0.4)], '001101': [0.014560145601456015, np.float64(-3.1)], '010101': [0.014370143701437016, np.float64(-1.7999999999999998)], '101011': [0.011320113201132012, np.float64(-1.0)], '111000': [0.008620086200862008, np.float64(0.0)], '100001': [0.007290072900729008, np.float64(-1.5)], '000101': [0.006580065800658007, np.float64(-0.9)], '110100': [0.006420064200642007, np.float64(-1.3000000000000003)], '011011': [0.006090060900609006, np.float64(-0.6)], '101000': [0.005780057800578007, np.float64(-1.3)], '100100': [0.0057700577005770064, np.float64(-1.6)], '101001': [0.005720057200572007, np.float64(-1.7000000000000002)], '101111': [0.005310053100531005, np.float64(-1.5)], '100110': [0.005230052300523006, np.float64(-0.7)], '010100': [0.004050040500405004, np.float64(-1.4)], '001100': [0.004040040400404004, np.float64(-2.7)], '011000': [0.003870038700387004, np.float64(-0.9)], '011111': [0.0032400324003240034, np.float64(-1.1)], '100000': [0.0031800318003180035, np.float64(-1.1)], '110010': [0.0029900299002990033, np.float64(-1.1)], '010001': [0.002930029300293003, np.float64(-1.3)], '011001': [0.002850028500285003, np.float64(-1.3)], '110011': [0.002770027700277003, np.float64(-0.5000000000000001)], '001110': [0.002650026500265003, np.float64(-1.8)], '010110': [0.0025900259002590025, np.float64(-0.5)], '110001': [0.0024400244002440027, np.float64(-1.2000000000000002)], '000100': [0.0022800228002280024, np.float64(-0.5)], '100011': [0.002020020200202002, np.float64(-0.8000000000000002)], '111011': [0.001770017700177002, np.float64(0.3)], '010000': [0.0015300153001530014, np.float64(-0.9)], '101010': [0.0014900149001490016, np.float64(-1.6)], '111111': [0.0014200142001420015, np.float64(-0.20000000000000007)], '000010': [0.0010500105001050011, np.float64(-0.3)], '110111': [0.0008300083000830009, np.float64(0.19999999999999984)], '010011': [0.0008200082000820009, np.float64(-0.6)], '001011': [0.0007900079000790008, np.float64(-0.7000000000000001)], '100111': [0.0007200072000720008, np.float64(-0.09999999999999998)], '001000': [0.0006300063000630007, np.float64(-1.0)], '100010': [0.0006000060000600005, np.float64(-1.4000000000000001)], '001111': [0.0005800058000580006, np.float64(-1.2000000000000002)], '010010': [0.0005700057000570006, np.float64(-1.2)], '001001': [0.0005300053000530005, np.float64(-1.4)], '011010': [0.0005200052000520005, np.float64(-1.2)], '000011': [0.0005000050000500006, np.float64(0.3)], '111010': [0.0004300043000430004, np.float64(-0.3)], '110000': [0.00026000260002600025, np.float64(-0.8000000000000002)], '000110': [0.00026000260002600025, np.float64(0.39999999999999997)], '010111': [0.00021000210002100023, np.float64(0.09999999999999998)], '000001': [0.0001700017000170002, np.float64(-0.4)], '001010': [0.00012000120001200013, np.float64(-1.3)], '000000': [9.00009000090001e-05, np.float64(0.0)], '000111': [8.000080000800009e-05, np.float64(1.0)], '110110': [2.0000200002000023e-05, np.float64(-0.4000000000000002)]}

    """

    method = problem_args["method"]


    if method == "LCD":
        # Check if AGP type is specified, otherwise use 1st order
        try: 
            agp_type = problem_args["agp_type"]
        except KeyError:
            agp_type = "order1"
            
        problem_operators = create_LCD_instance(
            Q, agp_type=agp_type, uniform_AGP_coeffs=problem_args["uniform"]
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

