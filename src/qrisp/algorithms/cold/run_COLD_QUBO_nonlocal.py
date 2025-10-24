from qrisp.operators.qubit import X, Y, Z
import numpy as np
from qrisp import QuantumVariable
from qrisp import x
from sympy import Symbol
import scipy
import sympy as sp
from qrisp.algorithms.cold.crab import CRABObjective


# N = 5
Q = np.array([[-1.2,  0.6,  0.6,  0.0,  0.0],
              [ 0.6, -0.8,  0.6,  0.0,  0.0],
              [ 0.6,  0.6, -0.9, -0.7,  0.0],
              [ 0.0,  0.0, -0.7, -0.4,  0.5],
              [ 0.0,  0.0,  0.0,  0.5,  0.3]])
N = len(Q[0])
#solution5 = {'00110': -2.7, '10110': -2.7} 

def create_nl_QUBO(Q):
    N = len(Q[0])
    h_i = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
    J = 0.5 * Q

# Build initial Hamiltonian
    #def H_init():
    H_i = -1 * sum([X(i) for i in range(N)])
        #return H_i

    def qarg_prep(qarg):
        from qrisp import h as hadamard 
        hadamard(qarg)


    # Build problem Hamiltonian
    #def H_prob():
    H_p = sum([sum([J[i][j]*Z(i)*Z(j) for j in range(i)]) for i in range(N)]) + sum([h_i[i]*Z(i) for i in range(N)])
        #return H_p

    #def A_lamb_nonlocal():
    A_lamb_nl_1 = -2* ( sum([h_i[i]*Y(i) for i in range(N)])  ) 
    A_lamb_nl_2 = -2* sum( [ sum([ J[i][j] * Y(i)*Z(j) + J[i][j]* Z(i)*Y(j) for i in range(j) ]) for j in range(N)])
    A_lamb = (A_lamb_nl_1+ A_lamb_nl_2)
        #return A_lamb

    # Control Hamiltonian
    #def H_control():
    H_control = sum([Z(i) for i in range(N)]) # *f_opt_CRAB(params, lamb) 
        #return H_control

    """ # Control pulse with randomized component (CRAB)
    def f_opt_crab(t, T, beta):
        r = np.random.uniform(-0.5, 0.5, len(beta))
        f = sum([beta[k] * np.sin(np.pi*(k+1+r[k])*t/T) for k in range(len(beta))])
        return f, r

    # Time derivative of CRAB control pulse
    def f_opt_crab_deriv(t, T, beta, r):
        # g_lam = t/T --> df/dt = 1/T df/dg
        return sum([1/T * beta[k] * np.pi*(k+1+r[k]) * np.cos(np.pi*(k+1+r[k])*t/T) for k in range(len(beta))])
    """
    # Function lambda of time
    def lam_t():
        #return np.sin(np.pi/2 * np.sin(np.pi*t/(2*T))**2)**2
        t1, T1 = sp.symbols('t T', real=True)
        lam_t_expr = sp.sin(sp.pi/2 * sp.sin(sp.pi*t1/(2*T1))**2)**2
        return lam_t_expr

    def alpha_nonlocal(lam, opt_CRAB, d_opt_CRAB):
        # lengthy expression for alpha with non-local AGP
        # from the FraunhoferGPT conversation:
        
        num_qubits = len(J[0])
        #n = num_qubits
        S_hR = sum([sum([J[i][j]**2 *(h_i[i]+h_i[j]) for i in range(j)]) for j in range(num_qubits)])
        S_hsqR = sum([sum([J[i][j]**2 *(h_i[i]**2+h_i[j]**2) for i in range(j)]) for j in range(num_qubits)])
        S_2 = sum([sum([J[i][j]**2 for i in range(j)]) for j in range(num_qubits)])
        S_4 = sum([sum([J[i][j]**4 for i in range(j)]) for j in range(num_qubits)])
        R_i_list =[ sum([J[i][j]**2 if j!=i else 0 for j in range(num_qubits)])  for i in range(num_qubits)]
        S_Rsq = sum( R_i**2 for R_i in R_i_list)
        S_h = sum(h_i)
        S_hsq = sum(i**2 for i in h_i)
        c = opt_CRAB
        dc = d_opt_CRAB

        denom = 4 *(
                c**2 *( num_qubits + 4*S_2)
                +c*lam *(2*S_h + 4*S_hR + 12*S_2)
                +lam**2 *(S_hsq + 2*S_hsqR + 6*S_hR + 2*S_Rsq + 4*S_2 + 2*S_4)
                + (1-lam)**2 *num_qubits
                + 8 *(1-lam)**2 *S_2
        )

        nom = num_qubits *c + S_h + 2*S_2 + num_qubits*(1-lam)* dc

        alpha = -nom/denom
        return alpha
    
    return qarg_prep, H_i, H_p, A_lamb, H_control,  lam_t, alpha_nonlocal


from qrisp.algorithms.cold.cold_algorithm import COLDproblem

qarg_prep, H_init, H_prob, A_lamb, H_control, lam_t, alpha_nonlocal = create_nl_QUBO(Q)

cold_problem = COLDproblem(qarg_prep, H_init, H_prob, A_lamb, H_control,  lam_t, alpha_nonlocal)
#run(self, qarg, N_steps, T, N_opt, CRAB=False):
qarg_cold = QuantumVariable(N)
# Evolution time
T = 0.5
# Number of timesteps
N_steps = T*20
# Number of control pulse parameters
N_opt = 1
# Use crab method
CRAB = True

meas_cold_crab = cold_problem.run(qarg_cold, int(N_steps), T, N_opt, CRAB)
print(meas_cold_crab)

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


print(f'Cost COLD-CRAB: {qubo_cost(Q, meas_cold_crab)}\n')
print("best 5")
b5 = list(meas_cold_crab.keys())[:5]
for i in b5:
    print(i, qubo_cost(Q, {i:1}))

print("cold_class")