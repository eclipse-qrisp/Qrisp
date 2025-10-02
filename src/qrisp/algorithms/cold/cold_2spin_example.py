from qrisp.operators.qubit import X, Y, Z
import numpy as np
from qrisp import QuantumVariable
from qrisp import h as hadamard
from qrisp import x



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

# System Hamiltonian w/o optimizable parameters
def H_0(lam):
    H = -2*J*(Z(0)*Z(1)) - h*(Z(0) + Z(1)) + 2*h*lam*(X(0) + X(1))
    return H

# Function lambda of time
def lam_t(t, T):
    return np.sin(np.pi/2 * np.sin(np.pi*t/(2*T))**2)**2

# Derivatice of lambda w.r.t time
def lam_deriv(t, T):
    return np.pi**2 * np.sin(np.pi*t/T) * np.sin(np.pi*np.sin(np.pi*t/(2*T))**2) / (4*T)

# Adiabatic gauge potential
def A_lam(alph):
    return alph * sum([Y(j) for j in range(N)])

# Optimal coefficients for AGP that minimize S = Tr(G_lambda^2)
def alpha(lam):
    return - h**2 / (4*(h*lam)**2 + h**2 + 4*J**2)

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
        alph = alpha(lam)

        # H_0 contribution scaled by dt
        H_step = dt * H_0(lam)

        # AGP contribution scaled by dt* lambda_dot(t)
        H_step += dt * lam_deriv(t, T) * A_lam(alph)

        # Get unitary from trotterization and apply to qarg
        U = H_step.trotterization()
        U(qarg)



def LCD_routine(qarg, N_steps):

    # Prepare qarg and apply hamiltonian
    # qarg_prep(qarg) #in |11> bringen
    apply_lcd_hamiltonian(qarg, N_steps)

    # Measure qarg
    res_dict = qarg.get_measurement()

    # Get statevector
    psi = qarg.qs.statevector(return_type='array')

    return psi, res_dict


# Evolution time
T = 1
# Number of timesteps
N_steps = 100

# Initialize qarg
qarg = QuantumVariable(N)
psi, meas = LCD_routine(qarg, N_steps)
print(meas)


########################
### Compute Fidelity ###
########################

# Hamiltonian at t = T
H_gs = (-2*J * (Z0 @ Z1) - h * (Z0 + Z1) + 2*h* (X0 + X1))
evals, evecs = np.linalg.eigh(H_gs)

# Ground state = eigenvector of lowest eigenvalue
ground_index = np.argmin(evals)
psi_target = evecs[:, ground_index]

# State of LCD outcome
psi_final = psi

print(f'psi_target: {psi_target}')
print(f'psi_final: {psi_final}')

overlap = np.vdot(psi_target.conjugate(), psi_final)   # inner product <psi_target|psi_final>
fidelity = np.abs(overlap)**2
print("Fidelity =", fidelity)