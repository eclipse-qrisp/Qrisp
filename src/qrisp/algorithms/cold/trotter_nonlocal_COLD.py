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
""" Q = np.array([[-0.9, 0.5, 0.3], 
              [0.5, -0.7, -0.4], 
              [0.3, -0.4, 0.2]]) """
### Solution
# x = (0, 1, 1) minimiza to xQx = -1.3

# Q = np.array([[-1.0, 0.5, 0.4, 0.0], 
#               [0.5, -0.9, 0.6, 0.0], 
#               [0.4, 0.6, -0.8, -0.5], 
#               [0.0, 0.0, -0.5, 0.2]])
# ### Solution
# # x = (1, 0, 1, 1) minimize to xQx = -1.8

### Medium example
Q = np.array([[-1.2,  0.6,  0.6,  0.0,  0.0],
            [ 0.6, -0.8,  0.6,  0.0,  0.0],
               [ 0.6,  0.6, -0.9, -0.7,  0.0],
               [ 0.0,  0.0, -0.7, -0.4,  0.5],
              [ 0.0,  0.0,  0.0,  0.5,  0.3]])

N = num_qubits = len(Q[0])
### Solution
# x = (0, 0, 1, 1, 0) with xQx = -2.7
# x = (1, 0, 1, 1, 0) with xQx = -2.7


#
Q = np.array([[-3,  2,  1,  0,  0,  3, -3],
       [ 2,  1, -2, -3,  0,  3,  2],
       [ 1, -2,  2,  2,  2,  0, -3],
       [ 0, -3,  2,  2,  0,  0, -1],
       [ 0,  0,  2,  0, -2,  3,  2],
       [ 3,  3,  0,  0,  3,  1, -1],
       [-3,  2, -3, -1,  2, -1,  2]])
# best solutions
# [1, 0, 1, 0, 0, 0, 1] with energy -9
N = num_qubits = len(Q[0])

h_i = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
J_ij = 0.5 * Q


H_i = -1*sum([X(i) for i in range(N)])
H_f_qubo =  sum([h_i[i] *Z(i) for i in range(N)]) + sum([ sum([ J_ij[i][j]*Z(i)*Z(j) for i in range(j) ]) for j in range(N) ])


# full final hamiltonian, also including transverse field
#H_f = H_f_qubo + H_f_cold 

H_control = sum([Z(i) for i in range(N)]) # *f_opt_CRAB(params, lamb) 

A_lamb_nl_1 = -2* ( sum([h_i[i]*Y(i) for i in range(N)])  ) 
A_lamb_nl_2 = -2* sum( [ sum([ J_ij[i][j] * Y(i)*Z(j) + J_ij[i][j]* Z(i)*Y(j) for i in range(j) ]) for j in range(N)])


A_lamb = (A_lamb_nl_1+ A_lamb_nl_2)


 #what is lambda? --> according to the paper https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010312
 # --> lambda is the sin scheduling
def lambda_symbolic(t, T):
    # the scheduling function, used to schedule an aspect of the problem hamiltonian, see p105
    lambda_t = np.sin(np.pi/2 * np.sin( np.pi*t/(2 *T ))**2 )**2
    return lambda_t

def lambda_t_deriv(t, T ): # time derivative of lambda_scheduling
    dtlambda = np.pi**2 * np.sin( np.pi*t/T )* np.sin( np.pi* np.sin( np.pi*t/( 2 *T ))**2 ) /(4*T)  
    return dtlambda


# Then the f_opt_CRAB
def f_opt_CRAB(params, lamb):

    # params == c_k 
    #  lamb == lambda parameter obviously
    #for i in range(len(params)):
    f_opt = sum([ params[k] * np.sin(2*np.pi *(k+1) *lamb) for k in range(len(params))]) 

    return f_opt

def deriv_f_opt_CRAB(params, lamb):

    d_f_opt = sum([ params[k] *  2*np.pi *(k+1) ** np.cos(2*np.pi *(k+1) *lamb) for k in range(len(params))]) 

    return d_f_opt


def alpha_symbolic(t, T, opt_CRAB, d_opt_CRAB):
    # lengthy expression for alpha with non-local AGP
    # from the FraunhoferGPT conversation:
    
    #n = num_qubits
    S_hR = sum([sum([J_ij[i][j]**2 *(h_i[i]+h_i[j]) for i in range(j)]) for j in range(num_qubits)])
    S_hsqR = sum([sum([J_ij[i][j]**2 *(h_i[i]**2+h_i[j]**2) for i in range(j)]) for j in range(num_qubits)])
    S_2 = sum([sum([J_ij[i][j]**2 for i in range(j)]) for j in range(num_qubits)])
    S_4 = sum([sum([J_ij[i][j]**4 for i in range(j)]) for j in range(num_qubits)])
    R_i_list =[ sum([J_ij[i][j]**2 if j!=i else 0 for j in range(num_qubits)])  for i in range(num_qubits)]
    S_Rsq = sum( R_i**2 for R_i in R_i_list)
    S_h = sum(h_i)
    S_hsq = sum(i**2 for i in h_i)
    lamb = lambda_symbolic(t,T)
    c = opt_CRAB
    dc = d_opt_CRAB

    denom = 4 *(
            c**2 *( num_qubits + 4*S_2)
            +c*lamb *(2*S_h + 4*S_hR + 12*S_2)
            +lamb**2 *(S_hsq + 2*S_hsqR + 6*S_hR + 2*S_Rsq + 4*S_2 + 2*S_4)
            + (1-lamb)**2 *num_qubits
            + 8 *(1-lamb)**2 *S_2
    )

    nom = num_qubits *c + S_h + 2*S_2 + num_qubits*(1-lamb)* dc

    alpha = -nom/denom
    return alpha

# Routine to prepare quantum variable
def qarg_prep(qarg):
    hadamard(qarg)






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

    qarg_prep(qarg)
    # Apply hamiltonian to qarg for each timestep
    dt = T / N_steps
    time = np.linspace(dt, T, N_steps)
    lam = np.array([lambda_symbolic(t, T) for t in time])
    lamdot = np.array([lambda_t_deriv(t, T) for t in time])
    sin_matrix, cos_matrix = precompute_opt_pulses(T, time, N_steps, N_opt, CRAB)

    if CRAB:
        # Transform beta to sympy to work with symbols for random values
        beta = sp.Matrix(beta)

    for s in range(1, N_steps+1):
        
        # Get t, lambda, alpha for the timestep
        t = s * dt
        
        f = sin_matrix[s-1, :] @ beta
        f_deriv = cos_matrix[s-1, :] @ beta
        alph = alpha_symbolic(t, T,f,f_deriv)
        #alph = alpha_symbolic(t, T,f[0],f_deriv[0])
        if hasattr(alph, "__len__"):
            alph = alph[0]
            f = f[0]
        """ print(sin_matrix.shape)
        print(sin_matrix[s-1, :].shape)
        print(type(beta))
        print(beta.shape)
        print(f.shape)
        print(type(alph) ) """
        #print(alph.shape)
        
        #print(type(alph))
        #print(alph.shape)
        #ddd = f * A_lamb
        # H_0 contribution scaled by dt
        H_step = dt *(1-lam[s-1])* H_i + dt * lam[s-1]*H_f_qubo
        """ print(type(A_lamb))
        ddd = d_crab_p * A_lamb
        A_lam22 =  alph* A_lamb """
        # AGP contribution scaled by dt* lambda_dot(t)
        
        H_step = dt * lamdot[s-1] * alph* A_lamb

        # Control pulse contribution 
        H_step = H_step + dt * f*  H_control
        #print(H_step)
        # Get unitary from trotterization and apply to qarg
        U = H_step.trotterization()
        U(qarg)

""" def apply_cold_hamiltonian(qarg, N_steps, beta, CRAB=False):

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
        U(qarg) """


def compile_U(qarg, N_opt, N_steps, CRAB=False):

    temp = list(qarg.qs.data)

    # Initzialize parameters as symbols
    params = [Symbol("par_"+str(i)) for i in range(N_opt)]

    apply_cold_hamiltonian(qarg, N_steps, params, CRAB=CRAB)

    intended_measurements = list(qarg)
    qc = qarg.qs.compile(intended_measurements=intended_measurements)

    qarg.qs.data = temp

    return qc




def optimization_routine(qarg, N_opt, qc, CRAB): 


    def objective(params):
        # Objective function to be minimized: 
        # Expectation value of the QUBO Hamiltonian

        # Dict to assign the optimization parameters
        subs_dic = {Symbol("par_"+str(i)): params[i] for i in range(len(params))}

        cost = H_f_qubo.expectation_value(qarg, compile=False, subs_dic=subs_dic, precompiled_qc=qc)()
        print("cost")
        print(cost)
        
        return cost
    
    init_point = np.random.rand(N_opt) * np.pi/2

    # Create CRAB objective to make sure randomization is applied at each optimization iteration
    if CRAB:
        objective = CRABObjective(H_f_qubo, qarg, qc, N_opt)
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
T = 2
# Number of timesteps
N_steps = T*20
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
""" t0 = time.time()
#psi_cold, meas_cold = COLD_routine(qarg_cold, N_steps, N_opt, CRAB=False) """
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

""" print(f'Computational time COLD: {t1-t0}')
print(meas_cold)
print(f'Cost COLD: {qubo_cost(Q, meas_cold)}\n') """

print(f'Computational time COLD-CRAB: {t2-t1}')
print(meas_cold_crab)
print(f'Cost COLD-CRAB: {qubo_cost(Q, meas_cold_crab)}\n')