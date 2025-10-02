from qrisp.operators.qubit import X, Y, Z
import numpy as np
from qrisp import QuantumVariable
from qrisp import h as hadamard
from qrisp import x
from sympy import Symbol
import scipy



### Some matrices for evaluating the statevector results ###

# Single-qubit Pauli matrices
pauli_I = np.array([[1, 0], [0, 1]], dtype=complex)
pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Two-qubit operators (site 0 = left, site 1 = right)
def kron2(A, B):
    return np.kron(A, B)

Z0 = kron2(pauli_Z, pauli_I)
Z1 = kron2(pauli_I, pauli_Z)
X0 = kron2(pauli_X, pauli_I)
X1 = kron2(pauli_I, pauli_X)
Y0 = kron2(pauli_Y, pauli_I)
Y1 = kron2(pauli_I, pauli_Y)


#########################
### System parameters ###
#########################

# J/h = 0.5
h = 1
J = 0.5
N = 2 # nr of qubits

# # Hamiltonian with matrices instead of operators for debugging
# def H_0_np(lam):
#     H = (-2*J * (Z0 @ Z1) - h * (Z0 + Z1) + lam*h* (X0 + X1))
#     return H

H0_without_field = -2*J*(Z(0)*Z(1)) - h*(Z(0) + Z(1))

# System Hamiltonian w/o optimizable parameters
def H_0(lam):
    H = -2*J*(Z(0)*Z(1)) - h*(Z(0) + Z(1)) + 2*h*lam*(X(0) + X(1))
    return H

# Control Hamiltonian
def H_control(lam, beta):
    H = f_opt(lam, beta) * sum([Z(i) for i in range(N)])
    return H

# Control pulse weighted by optimizable parameters beta
def f_opt(lam, beta):
    return sum([beta[k] * np.sin(np.pi*(k+1)*g_lam(lam)) for k in range(len(beta))])

# Timve derivative of control pulse
def f_opt_deriv(lam, beta):
    # g_lam = t/T --> df/dt = 1/T df/dg
    return sum([1/T * beta[k] * np.pi*(k+1) * np.cos(np.pi*(k+1)*g_lam(lam)) for k in range(len(beta))])

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
    return alph * sum([Y(j) for j in range(N)])

# Optimal coefficients for AGP that minimize S = Tr(G_lambda^2)
def alpha(lam, t, method: str = 'LCD', beta: list = None):

    if method == 'LCD':
        alph = - h**2 / (4*(h*lam)**2 + h**2 + 4*J**2)

    elif method == 'COLD':

        f = f_opt(lam, beta)
        f_deriv = f_opt_deriv(lam, beta)

        nom = h*(h+f) + h*lam*f_deriv/lam_deriv(t, T)
        denom = 4*(h*lam)**2 + (h + f)**2 + 4*J**2
        alph = nom/denom

    return alph

# Routine to prepare quantum variable
def qarg_prep(qarg):
    x(qarg)

# Apply LCD Hamiltonian by trotterization with N_steps
def apply_lcd_hamiltonian(qarg, N_steps):

    # Apply hamiltonian to qarg for each timestep
    dt = T / N_steps
    for s in range(1, N_steps+1):

        # Get t, lambda, alpha for the timestep
        t = s * dt
        lam = lam_t(t, T)
        alph = alpha(lam, t)

        # H_0 contribution scaled by dt
        H_step = dt * H_0(lam)

        # AGP contribution scaled by dt* lambda_dot(t)
        H_step = dt * lam_deriv(t, T) * A_lam(alph)

        # Get unitary from trotterization and apply to qarg
        U = H_step.trotterization()
        U(qarg)


def apply_cold_hamiltonian(qarg, N_steps, beta):

    # Apply hamiltonian to qarg for each timestep
    dt = T / N_steps
    for s in range(1, N_steps+1):

        # Get t, lambda, alpha for the timestep
        t = s * dt
        lam = lam_t(t, T)
        alph = alpha(lam, t, method='COLD', beta=beta)

        # H_0 contribution scaled by dt
        H_step = dt * H_0(lam)

        # AGP contribution scaled by dt* lambda_dot(t)
        H_step = H_step + dt * lam_deriv(t, T) * A_lam(alph)

        # Control pulse contribution 
        H_step = H_step + dt * H_control(lam, beta)

        # Get unitary from trotterization and apply to qarg
        U = H_step.trotterization()
        U(qarg)


def compile_U(qarg, N_opt, N_steps):

    temp = list(qarg.qs.data)

    # Initzialize parameters as symbols
    params = [Symbol("par_"+str(i)) for i in range(N_opt)]

    apply_cold_hamiltonian(qarg, N_steps, params)

    intended_measurements = list(qarg)
    qc = qarg.qs.compile(intended_measurements=intended_measurements)

    qarg.qs.data = temp # warum?

    return qc


def LCD_routine(qarg, N_steps):

    # Prepare qarg and apply hamiltonian
    # qarg_prep(qarg) #in |11> bringen
    apply_lcd_hamiltonian(qarg, N_steps)

    # Measure qarg
    res_dict = qarg.get_measurement()

    # Get statevector
    psi = qarg.qs.statevector(return_type='array')

    return psi, res_dict


def optimization_routine(qarg, N_opt, qc): 

    def objective(params):
        # Objective function to be minimized: 
        # Expectation value of the QUBO Hamiltonian

        # Dict to assign the optimization parameters
        subs_dic = {Symbol("par_"+str(i)): params[i] for i in range(len(params))}

        cost = H_0(lam=1).expectation_value(qarg, subs_dic=subs_dic, precompiled_qc=qc)()
        
        return cost
    
    init_point = np.random.rand(N_opt) * np.pi/2

    # print(f'Init {init_point} with cost {objective(init_point)}')

    res = scipy.optimize.minimize(objective,
                                  init_point,
                                  method='powell'
                                  )
    
    # print(f'Final cost: {objective(res.x,)}')
    
    return res.x


def COLD_routine(qarg, N_steps, N_opt):

    qarg1, qarg2 = qarg.duplicate(), qarg.duplicate()

    # Compile COLD routine into a circuit
    U_circuit = compile_U(qarg1, N_opt, N_steps)

    # Find optimal params for control pulse
    opt_params = optimization_routine(qarg2, N_opt, U_circuit)

    # Apply hamiltonian with optimal parameters
    apply_cold_hamiltonian(qarg, N_steps, opt_params)

    # Measure qarg and get statevector
    psi = qarg.qs.statevector('array')
    res_dict = qarg.get_measurement()

    return psi, res_dict


# Evolution time
T = 0.01
# Number of timesteps
N_steps = 1
# Number of control pulse parameters
N_opt = 1

# Initialize qarg
qarg_lcd, qarg_cold = QuantumVariable(N), QuantumVariable(N)
psi_lcd, meas = LCD_routine(qarg_lcd, N_steps)
psi_cold, meas = COLD_routine(qarg_cold, N_steps, N_opt)
# print(meas)


########################
### Compute Fidelity ###
########################

# Hamiltonian at t = T
H_gs = (-2*J * (Z0 @ Z1) - h * (Z0 + Z1) + 2*h* (X0 + X1))
evals, evecs = np.linalg.eigh(H_gs)

# Ground state = eigenvector of lowest eigenvalue
ground_index = np.argmin(evals)
psi_target = evecs[:, ground_index]

# print(f'psi_target: {psi_target}')
# print(f'psi_final: {psi_final}')

def fidelity(state_a, state_b):
    overlap = np.vdot(state_a.conjugate(), state_b)
    f = np.abs(overlap)**2
    return f

print(f"Fidelity LCD = {fidelity(psi_target, psi_lcd)}")
print(f"Fidelity COLD = {fidelity(psi_target, psi_cold)}")
