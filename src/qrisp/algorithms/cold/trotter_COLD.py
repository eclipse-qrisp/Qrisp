from qrisp.operators.qubit import X, Y, Z
import numpy as np
from qrisp import QuantumVariable
from qrisp import h as hadamard
from qrisp import x
from sympy import Symbol
import scipy
import sympy as sp
from qrisp.algorithms.cold.crab import CRABObjective
import time


#########################
### System parameters ###
#########################

### Small example
# Q = np.array([[-1, 2, -1],
#               [2, -2, 0],
#               [-1, 0, -1]])

### Solution
# x = (1, 0, 1) minimize to xQx = -4

### Other small example
Q = np.array([[-0.9, 0.5, 0.3], 
              [0.5, -0.7, -0.4], 
              [0.3, -0.4, 0.2]])
### Solution
# x = (0, 1, 1) minimiza to xQx = -1.3

# Q = np.array([[-1.0, 0.5, 0.4, 0.0], 
#               [0.5, -0.9, 0.6, 0.0], 
#               [0.4, 0.6, -0.8, -0.5], 
#               [0.0, 0.0, -0.5, 0.2]])
# ### Solution
# # x = (1, 0, 1, 1) minimize to xQx = -1.8

### Medium example
# Q = np.array([[-1.2,  0.6,  0.6,  0.0,  0.0],
#               [ 0.6, -0.8,  0.6,  0.0,  0.0],
#               [ 0.6,  0.6, -0.9, -0.7,  0.0],
#               [ 0.0,  0.0, -0.7, -0.4,  0.5],
#               [ 0.0,  0.0,  0.0,  0.5,  0.3]])

### Solution
# x = (0, 0, 1, 1, 0) with xQx = -2.7
# x = (1, 0, 1, 1, 0) with xQx = -2.7


### Values that bring Qubo problem into Hamiltonian form
h = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
J = 0.5 * Q
N = Q.shape[0]

### Precompute global Pauli-sums 
sum_z = sum([Z(i) for i in range(N)])
sum_y = sum([Y(i) for i in range(N)])

### Build Hamiltonain terms
H_i = -1 * sum([X(i) for i in range(N)])
H_p = sum([sum([J[i][j]*Z(i)*Z(j) for j in range(i)]) for i in range(N)]) + sum([h[i]*Z(i) for i in range(N)])

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
def apply_lcd_hamiltonian(qarg, N_steps):

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


def apply_cold_hamiltonian(qarg, N_steps, beta, CRAB=False):

    # Initialize qarg
    qarg_prep(qarg)

    # Precompute timegrid
    dt = T / N_steps
    time = np.linspace(dt, T, N_steps)
    lam = np.array([lam_t(t, T) for t in time])
    lamdot = np.array([lam_deriv(t, T) for t in time])
    sin_matrix, cos_matrix = precompute_opt_pulses(T, time, N_steps, N_opt, CRAB)

    if CRAB:
        # Transform beta to sympy to work with symbols for random values
        beta = sp.Matrix(beta)

    # Apply hamiltonian to qarg for each timestep
    for s in range(N_steps):

        # Get alpha, f and f_deriv for the timestep
        f = sin_matrix[s, :] @ beta
        f_deriv = cos_matrix[s, :] @ beta
        alph = alpha(lam[s], f, f_deriv)

        # H_0 contribution scaled by dt
        H_step = dt * H_0(lam[s])

        # AGP contribution scaled by dt* lambda_dot(t)
        H_step = H_step + dt * lamdot[s] * A_lam(alph)

        # Control pulse contribution 
        H_step = H_step + dt * H_control(f, CRAB)

        # Get unitary from trotterization and apply to qarg
        U = H_step.trotterization()
        U(qarg)


def compile_U(qarg, N_opt, N_steps, CRAB=False):

    temp = list(qarg.qs.data)

    # Initzialize parameters as symbols
    params = [Symbol("par_"+str(i)) for i in range(N_opt)]

    apply_cold_hamiltonian(qarg, N_steps, params, CRAB=CRAB)

    intended_measurements = list(qarg)
    qc = qarg.qs.compile(intended_measurements=intended_measurements)

    qarg.qs.data = temp

    return qc


def LCD_routine(qarg, N_steps):

    # Aapply hamiltonian
    apply_lcd_hamiltonian(qarg, N_steps)

    # Measure qarg
    res_dict = qarg.get_measurement()

    # Get statevector
    psi = qarg.qs.statevector(return_type='array')

    return psi, res_dict


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


def COLD_routine(qarg, N_steps, N_opt, CRAB=False):

    qarg1, qarg2 = qarg.duplicate(), qarg.duplicate()

    # Compile COLD routine into a circuit
    U_circuit = compile_U(qarg1, N_opt, N_steps, CRAB)

    # Find optimal params for control pulse
    opt_params = optimization_routine(qarg2, N_opt, U_circuit, CRAB)

    # Apply hamiltonian with optimal parameters
    # Here we do not want the randomized parameters to be included -> CRAB=False
    apply_cold_hamiltonian(qarg, N_steps, opt_params, CRAB=False)

    # Measure qarg and get statevector
    psi = qarg.qs.statevector('array')
    res_dict = qarg.get_measurement()

    return psi, res_dict


# Evolution time
T = 4
# Number of timesteps
N_steps = T*100
# Number of control pulse parameters
N_opt = 1
# Use crab method
CRAB = True

print("\n--- Simulation parameters ---")
print(f'T = {T}')
print(f'Steps = {N_steps}')
print(f'N opt = {N_opt}\n')
# print(f'CRAB: {CRAB}\n')

# Create qarg and run algorithms
qarg_lcd, qarg_cold, qarg_cold_crab = QuantumVariable(N), QuantumVariable(N), QuantumVariable(N)

# psi_lcd, meas_lcd = LCD_routine(qarg_lcd, N_steps)
t0 = time.time()
psi_cold, meas_cold = COLD_routine(qarg_cold, N_steps, N_opt, CRAB=False)
t1 = time.time()
psi_cold_crab, meas_cold_crab = COLD_routine(qarg_cold_crab, N_steps, N_opt, CRAB=True)
t2 = time.time()

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

# print(meas_lcd)
# print(f'Cost LCD: {qubo_cost(Q, meas_lcd)}\n')

print(f'Computational time COLD: {t1-t0}')
print(meas_cold)
print(f'Cost COLD: {qubo_cost(Q, meas_cold)}\n')

print(f'Computational time COLD-CRAB: {t2-t1}')
print(meas_cold_crab)
print(f'Cost COLD-CRAB: {qubo_cost(Q, meas_cold_crab)}\n')