from qrisp.operators.qubit import X, Y, Z
import numpy as np
from qrisp import QuantumVariable
from qrisp import h as hadamard
from qrisp import x
from sympy import Symbol
import scipy
import sympy as sp
from qrisp.algorithms.cold.crab import CRABObjective


# Build initial Hamiltonian
def H_init(N):
    H_i = -1 * sum([X(i) for i in range(N)])
    return H_i

# Build problem Hamiltonian
def H_prob(Q):
    global h
    global J

    h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

    H_p = sum([sum([J[i][j]*Z(i)*Z(j) for j in range(i)]) for i in range(N)]) + sum([h[i]*Z(i) for i in range(N)])

    return H_p

# Precompute pauli sums to avoid redundant computations
def compute_pauli_sums(N):
    sum_z = sum([Z(i) for i in range(N)])
    sum_y = sum([Y(i) for i in range(N)])
    return sum_z, sum_y

# Initialize some global variables for the QUBO matrix Q
# that are needed for further computations
def initialize_qubo_problem(Q):
    global N
    global H_i
    global H_p
    global sum_y
    global sum_z

    N = Q.shape[0]
    H_i = H_init(N)
    H_p = H_prob(Q)
    sum_z, sum_y = compute_pauli_sums(N)

# System Hamiltonian w/o optimizable parameters
def H_0(lam):
    H = (1-lam) * H_i + lam*H_p
    return H

# Control Hamiltonian
def H_control(f, CRAB):
    if CRAB:
        H = sum(f[i, 0] * sum_z for i in range(f.rows))
    else:
        H = f * sum_z
    return H

# Control pulse weighted by optimizable parameters beta
def f_opt(t, T, beta):
    return sum([beta[k] * np.sin(np.pi*(k+1)*t/T) for k in range(len(beta))])

# Time derivative of control pulse
def f_opt_deriv(t, T, beta):
    # g_lam = t/T --> df/dt = 1/T df/dg
    return sum([1/T * beta[k] * np.pi*(k+1) * np.cos(np.pi*(k+1)*t/T) for k in range(len(beta))])

# Control pulse with randomized component (CRAB)
def f_opt_crab(t, T, beta):
    r = np.random.uniform(-0.5, 0.5, len(beta))
    f = sum([beta[k] * np.sin(np.pi*(k+1+r[k])*t/T) for k in range(len(beta))])
    return f, r

# Time derivative of CRAB control pulse
def f_opt_crab_deriv(t, T, beta, r):
    # g_lam = t/T --> df/dt = 1/T df/dg
    return sum([1/T * beta[k] * np.pi*(k+1+r[k]) * np.cos(np.pi*(k+1+r[k])*t/T) for k in range(len(beta))])

# Function lambda of time
def lam_t(t, T):
    return np.sin(np.pi/2 * np.sin(np.pi*t/(2*T))**2)**2

# Derivatice of lambda w.r.t time
def lam_deriv(t, T):
    return np.pi**2 * np.sin(np.pi*t/T) * np.sin(np.pi*np.sin(np.pi*t/(2*T))**2) / (4*T)

# Function g = time/tau of lambda
def g_lam(lam):
    g = 2/np.pi * np.arcsin(np.sqrt(2/np.pi * np.arcsin(np.sqrt(lam))))
    return g

# Adiabatic gauge potential
def A_lam(alph):
    return alph * sum_y

# Optimal coefficients for AGP that minimize S = Tr(G_lambda^2)
def alpha(lam, f = 0, f_deriv = 0):

    A = lam * h + f
    B = 1 - lam
    C = h + f_deriv

    nom = np.sum(A + 4*B*C)
    denom = 2 * (np.sum(A**2) + N * (B**2)) + 4 * (lam**2) * np.sum(np.tril(J, -1).sum(axis=1))
    alph = nom/denom

    return alph

# Routine to prepare quantum variable
def qarg_prep(qarg):
    hadamard(qarg)

# Apply LCD Hamiltonian by trotterization with N_steps
def apply_lcd_hamiltonian(qarg, N_steps, T):

    qarg_prep(qarg)

    # Precompute timegrid
    dt = T / N_steps
    t_list = np.linspace(dt, T, N_steps)
    lam = np.array([lam_t(t, T) for t in t_list])
    lamdot = np.array([lam_deriv(t, T) for t in t_list])

    # Apply hamiltonian to qarg for each timestep
    for s in range(N_steps):

        # Get alpha for the timestep
        alph = alpha(lam[s])

        # H_0 contribution scaled by dt
        H_step = dt * H_0(lam[s])

        # AGP contribution scaled by dt* lambda_dot(t)
        H_step += dt * lamdot[s] * A_lam(alph)

        # Get unitary from trotterization and apply to qarg
        U = H_step.trotterization()
        U(qarg)

# Precompute f (sine) and f_deriv (cosine) for each timestep
def precompute_opt_pulses(T, t_list, N_steps, N_opt, CRAB: bool):

    if CRAB:
        # Matrices and arrays must be sympy objects
        sin_matrix = sp.MutableDenseMatrix.zeros(N_steps, N_opt)
        cos_matrix = sp.MutableDenseMatrix.zeros(N_steps, N_opt)
        t_list = sp.Array(t_list)

        # Add symbols for random parameters to be called in optimizaton
        r_params = [Symbol("r_"+str(i)) for i in range(N_opt)]
        for k in range(N_opt):
            sin_matrix[:, k] = sp.Matrix(t_list.applyfunc(lambda t: sp.sin(sp.pi * (k+1+r_params[k]) * t / T)))
            cos_matrix[:, k] = sp.Matrix(t_list.applyfunc(lambda t: (sp.pi/T * (k+1+r_params[k])) * sp.cos(sp.pi * (k+1+r_params[k]) * t / T)))

    else:
        sin_matrix = np.zeros((N_steps, N_opt))
        cos_matrix = np.zeros((N_steps, N_opt))

        for k in range(N_opt):
            sin_matrix[:, k] = np.sin(np.pi * (k+1) * t_list/T)
            cos_matrix[:, k] = (np.pi/T * (k+1)) * np.cos(np.pi * (k+1) * t_list/T)

    return sin_matrix, cos_matrix


def apply_cold_hamiltonian(qarg, N_steps, T, beta, CRAB=False):

    # Initialize qarg
    qarg_prep(qarg)

    # Precompute timegrid
    N_opt = len(beta)
    dt = T / N_steps
    time = np.linspace(dt, T, N_steps)
    lam = np.array([lam_t(t, T) for t in time])
    lamdot = np.array([lam_deriv(t, T) for t in time])
    sin_matrix, cos_matrix = precompute_opt_pulses(T, time, N_steps, N_opt, CRAB)

    if CRAB:
        # Transform beta to sympy to work with symbols for random values
        beta = sp.Matrix(beta)

    # Trotterize Hamiltonians with different time-dependent prefactors
    U1 = H_i.trotterization()
    U2 = H_p.trotterization()
    U3 = sum_y.trotterization()
    U4 = sum_z.trotterization()

    # Apply hamiltonian to qarg for each timestep
    for s in range(N_steps):

        # Get alpha, f and f_deriv for the timestep
        f = sin_matrix[s, :] @ beta
        f_deriv = cos_matrix[s, :] @ beta
        alph = alpha(lam[s], f, f_deriv)

        # Prefactor for U1
        a = dt * (1 - lam[s])
        U1(qarg, a)
        # Prefactor for U2
        b = dt * lam[s]
        U2(qarg, b)
        # Prefactor for U3
        c = dt * lamdot[s] * alph
        U3(qarg, c)
        # Prefactor for U4
        if CRAB:
            d = dt * sum(f[i, 0] for i in range(f.rows)) # richtig?
        else:
            d = dt * f
        U4(qarg, d)


def compile_U(qarg, N_opt, N_steps, T, CRAB=False):

    temp = list(qarg.qs.data)

    # Initzialize parameters as symbols
    params = [Symbol("par_"+str(i)) for i in range(N_opt)]

    apply_cold_hamiltonian(qarg, N_steps, T, params, CRAB=CRAB)

    intended_measurements = list(qarg)
    qc = qarg.qs.compile(intended_measurements=intended_measurements)

    qarg.qs.data = temp

    return qc


def LCD_routine(Q, qarg, N_steps, T):

    initialize_qubo_problem(Q)

    # Aapply hamiltonian
    apply_lcd_hamiltonian(qarg, N_steps, T)

    # Measure qarg
    res_dict = qarg.get_measurement()

    return res_dict


def optimization_routine(qarg, N_opt, qc, CRAB): 


    def objective(params):
        # Objective function to be minimized: 
        # Expectation value of the QUBO Hamiltonian

        # Dict to assign the optimization parameters
        subs_dic = {Symbol("par_"+str(i)): params[i] for i in range(len(params))}

        cost = H_p.expectation_value(qarg, compile=False, subs_dic=subs_dic, precompiled_qc=qc)()
        
        return cost
    
    init_point = np.random.rand(N_opt) * np.pi/2

    # Create CRAB objective to make sure randomization is applied at each optimization iteration
    if CRAB:
        objective = CRABObjective(H_p, qarg, qc, N_opt)
    # Otherwise objective from above is used
    else:
        objective = objective

    res = scipy.optimize.minimize(objective,
                                  init_point,
                                  method='Powell'
                                  )
    
    return res.x


def COLD_routine(Q, qarg, N_steps, T, N_opt, CRAB=False):

    initialize_qubo_problem(Q)

    qarg1, qarg2 = qarg.duplicate(), qarg.duplicate()

    # Compile COLD routine into a circuit
    U_circuit = compile_U(qarg1, N_opt, N_steps, T, CRAB)

    # Find optimal params for control pulse
    opt_params = optimization_routine(qarg2, N_opt, U_circuit, CRAB)

    # Apply hamiltonian with optimal parameters
    # Here we do not want the randomized parameters to be included -> CRAB=False
    apply_cold_hamiltonian(qarg, N_steps, T, opt_params, CRAB=False)

    # Measure qarg
    res_dict = qarg.get_measurement()

    return res_dict

def qubo_cost(Q, P):
    expected_cost = 0.0
    for bitstring, prob in P.items():
        # Convert bitstring (e.g., "10110") to numpy array of ints
        x = np.array([int(b) for b in bitstring], dtype=float)
        # Compute quadratic form x^T Q x
        cost = x @ Q @ x
        # Weight by probability
        expected_cost += prob * cost
    return expected_cost

def success_prob(meas, solution):
    sp = 0
    for s in solution.keys():
        try:
            sp += meas[s]
        except KeyError:
            continue
    return sp

def approx_ratio(Q, meas, solution):
    cost = qubo_cost(Q, meas)
    opt_cost = list(solution.values())[0]
    ar = cost/opt_cost
    return ar
