from qrisp.operators.qubit import X, Y, Z
import numpy as np
from qrisp import QuantumVariable
from qrisp import h as hadamard
from qrisp import x
from sympy import Symbol
import scipy



num_qubits = 5
n = num_qubits
#Q = np.ones((n, n)) # QUBO Matrix
Q = np.array([
    [-1.2,  0.6,  0.6,  0.0,  0.0],
    [ 0.6, -0.8,  0.6,  0.0,  0.0],
    [ 0.6,  0.6, -0.9, -0.7,  0.0],
    [ 0.0,  0.0, -0.7, -0.4,  0.5],
    [ 0.0,  0.0,  0.0,  0.5,  0.3]
], dtype=float)
# best solutions:
# x=(0,0,1,1,0) x=(1,0,1,1,0) with -2.7


num_qubits = 7
n = num_qubits
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


h_i = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
J_ij = 0.5 * Q


H_i = -1*sum([X(i) for i in range(n)])
H_f_qubo =  sum([h_i[i] *Z(i) for i in range(n)]) + sum([ sum([ J_ij[i][j]*Z(i)*Z(j) for i in range(j) ]) for j in range(n) ])


# full final hamiltonian, also including transverse field
#H_f = H_f_qubo + H_f_cold 

H_control = sum([Z(i) for i in range(n)]) # *f_opt_CRAB(params, lamb) 

A_lamb_nl_1 = -2* ( sum([h_i[i]*Y(i) for i in range(n)])  ) 
A_lamb_nl_2 = -2* sum( [ sum([ J_ij[i][j] * Y(i)*Z(j) + J_ij[i][j]* Z(i)*Y(j) for i in range(j) ]) for j in range(n)])


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
    
    n = num_qubits
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
            c**2 *( n + 4*S_2)
            +c*lamb *(2*S_h + 4*S_hR + 12*S_2)
            +lamb**2 *(S_hsq + 2*S_hsqR + 6*S_hR + 2*S_Rsq + 4*S_2 + 2*S_4)
            + (1-lamb)**2 *n 
            + 8 *(1-lamb)**2 *S_2
    )

    nom = n *c + S_h + 2*S_2 + n*(1-lamb)* dc

    alpha = -nom/denom
    return alpha

# Routine to prepare quantum variable
def qarg_prep(qarg):
    hadamard(qarg)


def apply_cold_hamiltonian(qarg, N_steps, beta):

    qarg_prep(qarg)
    # Apply hamiltonian to qarg for each timestep
    dt = T / N_steps
    for s in range(1, N_steps+1):

        # Get t, lambda, alpha for the timestep
        t = s * dt
        lam = lambda_symbolic(t, T)
        lam_deriv = lambda_t_deriv(t, T)
        crab_p = f_opt_CRAB(beta,lam)
        d_crab_p = deriv_f_opt_CRAB(beta,lam)
        alph = alpha_symbolic(t, T,crab_p,d_crab_p)

        # H_0 contribution scaled by dt
        H_step = dt *(1-lam)* H_i+ dt * lam*H_f_qubo

        # AGP contribution scaled by dt* lambda_dot(t)
        H_step = dt * lam_deriv * alph* A_lamb

        # Control pulse contribution 
        H_step = H_step + dt * crab_p*  H_control
        #print(H_step)
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
    #print(qc)
    qarg.qs.data = temp # warum?

    return qc


def optimization_routine(qarg, N_opt, qc): 

    def objective(params):
        # Objective function to be minimized: 
        # Expectation value of the QUBO Hamiltonian
        # Dict to assign the optimization parameters
        subs_dic = {Symbol("par_"+str(i)): params[i] for i in range(len(params))}
        
        cost = H_f_qubo.expectation_value(qarg, subs_dic=subs_dic, precompiled_qc=qc)()
        print(params)
        print(cost)
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
T = 1
# Number of timesteps
N_steps = 20
# Number of control pulse parameters
N_opt = 1
N= num_qubits
# Initialize qarg
qarg_lcd, qarg_cold = QuantumVariable(N), QuantumVariable(N)
psi_cold, meas = COLD_routine(qarg_cold, N_steps, N_opt)
# print(meas)


print("final_res")
print(meas)

#post processing for solution quality
def average_qubo_cost(Q, P):
    expected_cost = 0.0
    for bitstring, prob in P.items():
        # Convert bitstring (e.g., "10110") to numpy array of ints
        x = np.array([int(b) for b in bitstring], dtype=float)
        # Compute quadratic form x^T Q x
        cost = x @ Q @ x
        # Weight by probability
        expected_cost += prob * cost
    return expected_cost

print(average_qubo_cost(Q=Q, P=meas))
