from qrisp.operators.qubit import X,Y,Z, conjugate
import numpy as np
from qrisp import IterationEnvironment, merge, invert, jrange, check_for_tracing_mode, QuantumVariable
import jax.numpy as jnp 

"""
Questions right now: 



Is derivative correct? Should there be a delta_t in there? If not, should there instead be the division by "steps" in the simulate-function?
--> i think the division by "steps" is correct, or no division
"""
# based on KIPU logistics scheduling https://arxiv.org/pdf/2405.15707


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

"""
x=(0,0,1,1,0) 
x=(1,0,1,1,0) 
"""
h_i = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
J_ij = 0.5 * Q

# build the AGP hamiltonian
A_lambda1 = -2* ( sum([h_i[i]*Y(i) for i in range(n)])  ) 
A_lambda2 = -2* sum( [ sum([ J_ij[i][j] * Y(i)*Z(j) + J_ij[i][j]* Z(i)*Y(j) for i in range(j) ]) for j in range(n)])
adiabatic_gp = (A_lambda1+ A_lambda2)

H_i = -1*sum([X(i) for i in range(n)])
H_f =  sum([h_i[i] *Z(i) for i in range(n)]) + sum([ sum([ J_ij[i][j]*Z(i)*Z(j) for i in range(j) ]) for j in range(n) ])
print(H_i)
print(H_f)

def create_CD_unitary( 
        H_i, H_f, adiabatic_gp, qarg_prep,
        order=1, method="commuting_qw", forward_evolution=True):
    
    def lambda_t_deriv(t_steps, t ): # time derivative of lambda 
        dtlambda = np.pi**2 * np.sin( np.pi*t_steps/t )* np.sin( np.pi* np.sin( np.pi*t_steps/( 2 *t ))**2 ) /(4*t)  
        return dtlambda

    def lambda_symbolic(t_steps, t):
        lambda_t = np.sin(np.pi/2 * np.sin( np.pi*t_steps/(2 *t ))**2 )**2
        return lambda_t

    def alpha_symbolic(t_steps, t ):
        lambda_t = lambda_symbolic(t_steps, t)
        alpha_nom = -(sum([h_i[i]**2 for i in range(n)]) + sum([ sum([ J_ij[i][j]**2 for i in range(j) ]) for j in range(n) ])) /4
        alpha_denom = ( 
                        (1-lambda_t)**2 * ( sum([h_i[i]**2 for i in range(n)]) + 4* sum([ sum([ (J_ij[i][j]**2 if i!=j else 0) for i in range(n) ]) for j in range(n) ])) 
                        + lambda_t**2* ( sum([h_i[i]**4 for i in range(n)]) + sum([ sum([ (J_ij[i][j]**4 if i!=j else 0) for i in range(n) ]) for j in range(n) ])) 
                        + lambda_t**2* ( 6* sum([ sum([ h_i[i]**2 * J_ij[i][j]**2 if i!=j else 0 for i in range(n) ]) for j in range(n) ])) 
                        + lambda_t**2* ( 6* sum([ sum([ sum([ J_ij[i][j]**2 * J_ij[i][k]**2 + J_ij[i][j]**2 * J_ij[j][k]**2 + J_ij[i][k]**2 * J_ij[j][k]**2  for i in range(j) ]) for j in range(k) ]) for k in range(n) ])) 
                        )
    
        alpha = alpha_nom/alpha_denom
        return alpha


    def cd_unitary(qarg, t, steps):

        qarg_prep(qarg)
        for ind in range(1,steps+1):
            t_steps=ind*t/steps
            O = (
                (1-lambda_symbolic(t_steps, t))/steps  *H_i 
                    + lambda_symbolic(t_steps, t)/steps  *H_f
                    + lambda_t_deriv(t_steps, t) *alpha_symbolic(t_steps, t)/steps  *adiabatic_gp)
            
            print("params")
            """ print("lambda_sym " + str(lambda_symbolic(t, T)))        
            print("lambda_t_deriv " + str(lambda_t_deriv(t, T)))  """    
            print("alpha_symbolic " + str(alpha_symbolic(t_steps, t)))      
            U = O.trotterization()
            U(qarg)

    return cd_unitary


# good params 0.5/50
T_total =   1 # according to KIPU paper this parameter is better to be chosen small... --> 0.01? 
N_steps = 2 # what is the ideal parameter here?
#delta_t = T_total/N_steps

from qrisp import h, cx
qarg = QuantumVariable(n)
 # which one is correct?? 

def qarg_prep(qarg):
    h(qarg)

U_cd = create_CD_unitary( H_i, H_f, adiabatic_gp, qarg_prep)

U_cd(qarg, t=T_total, steps=N_steps)
#print(qarg.qs)
res_dict = qarg.get_measurement()
#print(res_dict)

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

print(average_qubo_cost(Q=Q, P=res_dict))