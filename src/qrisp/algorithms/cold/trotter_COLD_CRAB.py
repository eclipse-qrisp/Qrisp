from qrisp.operators.qubit import X,Y,Z, conjugate
import numpy as np
from qrisp import QuantumVariable, h 
from sympy import Symbol



"""
# to establish the alpha for optimization i need to:

# G_λ = ∂_λ H_beta + i[A_λ^(1), H_beta] --> derive commutator
# 1/8 Tr [G_λ^2] --> solve square trace of generator for alpha 
# WHERE: 
# H_beta = ( H_0 (p105) + (eq 6.20 p106, ONLY THE sigma_z part) ) 
# --> f^opt_CRAB are from eq 6.21
# AND A_λ^(1) = alpha * (sum sigma_j) --> no  λdt for this derivation
# --> should result in lenghty expression including all the h_i and J_ij

ATTENTION!! This is btw different to the original nested commutator approximation derivation:
A_λ^(1) = ∂_λ H_beta + alpha_1 i[ H_beta, ∂_λ H_beta] 
Which is because we chose a custom AGP here ~ sum(Y[i])
Is this better? What happens if i stick with the normal derivation?
---
ATTENTION!! we have forget the lambda dependency for H_0, i.e. the mulplicative lambda infront of H_i and H_p

ATTENTION!!
Lambda in this work is defined as a redefinition of time, i.e.
--> lambda = t/tau
so any mentions of lambda dependency == t dependendency.
It is further wrapped into the sin^2 scheduling at some parts through-out the thesis,
where we end up with the KIPU scheduling again
"""

# --> this should result in 6.18, p105 
# BUT REPLACE Z^2 -> (Z+f^opt_CRAB )^2   and   Z ∂_λ X -> ((Z+f^opt_CRAB ) ∂_λ X - (∂_λ f^opt_CRAB ) X )   
# (??? NOT CORRECT, see point above, need to consider problem matrix coefficients)

#THEN THE FULL HAMILTONIAN LOOKS LIKE:

# H_COLD = ( H_0 (p105) + (eq 6.20 p106, ONLY THE sigma_z part) +  λdt * alpha * (sum sigma_j) ),
# see p106
# where the last part is the AGP

# Q: How do we optimize? 
# --> ATTENTION!! The suggested optimization schemes (Nealder-Mead and Powell are local optimizers, not suited for the problem at hand really (or maybe yes?) )
#   --> maybe use dual annealing?
# --> we have params = c_k of len(N_k) which are optimizeable
# --> but with respect to which cost function --> fidelity 
# --> can also minimize the size of the control pulse, see eq. 5.2 p94
#Quote from p86 on the optimization procedure
#All that is left now is the third COLD step, which is the optimisation of the coefficients βk(λ) 
#using QOCT methods presented in Sec. 3.2. A natural, though not
#exclusive, cost function for this process would be the final state fidelity from Eq. (3.15)
#with respect to the desired eigenstate of H(λf ). We will provide a more detailed discussion of the optimal control component of COLD in the 
# --> ⟨ψ(τ,u)|HT |ψ(τ,u)⟩ ??
# !!!! How can we improve the results??




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



""" num_qubits = 3        
n = num_qubits
#Q = np.ones((n, n)) # QUBO Matrix
Q = np.array([
    [-1.2,  0.6,  0.6],
    [ 0.6, -0.8,  0.6],
    [ 0.6,  0.6, -0.9],
], dtype=float)  """

h_i = -0.5 * np.diag(Q) - 0.5 * np.sum(Q, axis=1)
J_ij = 0.5 * Q


H_i = -1*sum([X(i) for i in range(n)])
H_f_qubo =  sum([h_i[i] *Z(i) for i in range(n)]) + sum([ sum([ J_ij[i][j]*Z(i)*Z(j) for i in range(j) ]) for j in range(n) ])


# additional transverse field hamiltonian, which is added to final hamiltionian, and taken times lambda.
# lambda multiplication does not appear 
a = h_i# or n*[10] # some scaling parameter chosen 10*J_0 in the orignial work? 
# set a = h_i ? 
H_f_cold =  sum([a[i] *X(i) for i in range(n)]) # *lamb_scheduling

# full final hamiltonian, also including transverse field
#H_f = H_f_qubo + H_f_cold 

H_control = sum([Z(i) for i in range(n)]) # *f_opt_CRAB(params, lamb) 

A_lamb = -1*sum([Y(i) for i in range(n)]) # *lamb_dt *alpha

""" A_lambda1 = -2* ( sum([h_i[i]*Y(i) for i in range(n)])  ) 
A_lambda2 = -2* sum( [ sum([ J_ij[i][j] * Y(i)*Z(j) + J_ij[i][j]* Z(i)*Y(j) for i in range(j) ]) for j in range(n)])
A_lamb = (A_lambda1+ A_lambda2) 
 """


 #what is lambda?
def lambda_symbolic(t, T):
    # on p28 of the thesis it only talks about redefinition of time, s.t.
    # lambda = t/tau --> here =t/T
    # no further mention of a scheduling function??
    lambda_t = t/T 
    return lambda_t

def lambda_t_deriv(t, T ): # time derivative of lambda 
    return 1/T

def lambda_scheduling(t, T):
    # the scheduling function, used to schedule an aspect of the problem hamiltonian, see p105
    lambda_t = np.sin(np.pi/2 * np.sin( np.pi*t/(2 *T ))**2 )**2
    return lambda_t

def lambda_scheduling_deriv(t, T ): # time derivative of lambda_scheduling
    dtlambda = np.pi**2 * np.sin( np.pi*t/T )* np.sin( np.pi* np.sin( np.pi*t/( 2 *T ))**2 ) /(4*T)  
    return dtlambda


# Then the f_opt_CRAB
def f_opt_CRAB(params, lamb, r_uniform):
    
    #!!!!!!!!!!!!!
    # r_uniform needs to be rerandomized on every iteration...

    # params == c_k 
    #  lamb == lambda parameter obviously
    #for i in range(len(params)):
    f_opt = sum([ params[k] * np.sin(2*np.pi *(k+1) *(1+ r_uniform[k])*lamb) for k in range(len(params))]) 

    return f_opt

def deriv_f_opt_CRAB(params, lamb, r_uniform):

    d_f_opt = sum([ params[k] *  2*np.pi *(k+1) *(1+ r_uniform[k]) * np.cos(2*np.pi *(k+1) *(1+ r_uniform[k]) *lamb) for k in range(len(params))]) 

    return d_f_opt


# DONT FORGET!! THERE IS AN i HIDDEN SOMEWHERE!!
def alpha_symbolic_old(t, T, opt_CRAB #J_ij, h_i, f_opt_CRAB
                   ):
    
    # some random names taken from my FHGenie derivation lol
    d = lambda_scheduling_deriv(t,T)
    g = lambda_scheduling(t,T)
    v = opt_CRAB
    b_a_i =[v +g +i for i in h_i]

    denom = 2*( 
            sum([ 
                sum([ 
                    J_ij[i][j] 
                    for i in range(len(J_ij[0]))]) 
                for j in range(len(J_ij[0]))] )
            + sum([i**2 for i in b_a_i])
            )       
    alpha = d* sum(b_a_i) / (denom) # *i-> imaginary !!!
    
    return alpha 

# DONT FORGET!! THERE IS AN i HIDDEN SOMEWHERE!!
def alpha_symbolic(t, T, opt_CRAB, d_opt_CRAB #J_ij, h_i, f_opt_CRAB
                   ):
    
    # Carlottas derivation
    
    v = opt_CRAB
    w = d_opt_CRAB
    lamb = lambda_symbolic(t,T)
    A = [(lamb * h_i[i] + v) for i in range(len(h_i))]
    B = 1 - lamb
    C = [( h_i[i] + w) for i in range(len(h_i))]


    a_nom = sum ([A[i] + 4*B* C[i] for i in range(len(A))])
    a_denom = 2* sum([A[i]**2 + B**2 for i in range(len(A))]) + 4 * sum([sum([J_ij[i][j]for i in range(j)]) for j in range(len(J_ij[0]))])
    alpha = a_nom/ a_denom # *i-> imaginary !!!
    
    return alpha 





def trotterization_COLD(steps,  T,
        #params, 
        #H_i, H_f_qubo, H_f_cold , H_control, A_lamb, 
        #lambda_t_deriv, lambda_symbolic, lambda_scheduling, lambda_scheduling_deriv, 
        #alpha_symbolic, 
        order=1, method="commuting_qw", forward_evolution=True
        ):
    
    # set seeded randomly drawn r_k for f_opt_CRAB
    

    def cold_hamiltonian(qarg, qarg_prep, params):
        #r_uniform = np.random.uniform(-0.5,0.5,len(params))
        r_uniform = np.ones(1,)*0.2
        #qarg_prep(qarg)
        h(qarg)
        for ind in range(1,int(steps)+1):
            t=ind*T/steps
            opt_CRAB = f_opt_CRAB(params, lambda_symbolic(t, T), r_uniform)
            d_opt_CRAB = deriv_f_opt_CRAB(params, lambda_symbolic(t, T), r_uniform)

            try:       
                control_term =  opt_CRAB/steps *H_control
            except TypeError:
                control_term = 0 

            O = (
                (1-lambda_symbolic(t, T)) /steps *H_i 
                + lambda_symbolic(t, T) /steps *H_f_qubo 
                #+ lambda_scheduling(t, T) /steps *H_f_cold # this term was added in the PhD thesis, we can omit it for our purposes
                + control_term 
                + lambda_t_deriv(t, T) *alpha_symbolic(t, T, opt_CRAB,d_opt_CRAB)/steps *A_lamb 
            ) #*i-> imaginary DONT FORGET!! THERE IS AN i HIDDEN SOMEWHERE!!
            # *lambda_symbolic(t, T) ??
            """ print("params")
            print(params)
            print("lambda_sym " + str(lambda_symbolic(t, T)))        
            print("lambda_t_deriv " + str(lambda_t_deriv(t, T)))     
            print("lambda_scheduling " + str(lambda_scheduling(t, T)))     
            print("alpha_symbolic " + str(alpha_symbolic(params, t, T, opt_CRAB)))         
            print("f_opt_CRAB " + str(opt_CRAB) ) """
            
            #print(O)
            U = O.trotterization()
            U(qarg)
        #print(qarg.qs)
    return cold_hamiltonian





def optimization_routine(qarg, cold_hamiltonian, qarg_prep, N_params,
                        optimizer="Nelder-Mead",
                        options={},
                        ):

    
    """ MINIMIZATION FROM THE QAOA implementation"""
    # --> need to extract qc from trotter circuit
    # --> use compile_circuit method for that. 

    # need deltat, T ??
    
    #print(compiled_qc)
    
    import scipy
    #https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
    

    def optimization_wrapper(params, qarg, qarg_prep , mes_kwargs={}):
        #params = np.random.rand(N_params) * np.pi / 2
        #print("wrapper")
        temp = list(qarg.qs.data)
        params = [0.57489003]
        # Define params for COLD control terms
        symbols = [Symbol("params" + str(i)) for i in range(N_params)]


        
        cold_hamiltonian(qarg,qarg_prep, params)
        #print(qarg.qs)
        # Compile quantum circuit with intended measurements
        intended_measurement_qubits = list(qarg)
        compiled_qc = qarg.qs.compile(intended_measurements=intended_measurement_qubits)
        print(qarg.qs.depth())
        qarg.qs.data = temp
        subs_dic = {symbols[i]: params[i] for i in range(len(symbols))}
        # instead of this state_prep it should somehow be a U(subs_dic)
        cl_cost = -H_f_qubo.expectation_value(qarg, subs_dic=subs_dic, precompiled_qc=compiled_qc)()
        print(cl_cost)
        if cl_cost < -2:
            print(qarg.get_measurement(subs_dic=subs_dic, precompiled_qc=compiled_qc))
            print(compiled_qc)
        #print(qarg.qs.depth())
        print(params)


        return cl_cost

    init_point = np.random.rand(N_params) * np.pi / 2

    res_sample = scipy.optimize.minimize(
                optimization_wrapper,
                init_point,
                #method=optimizer,
                options={"maxiter":20},
                args=(qarg, qarg_prep
                      #, mes_kwargs
                      ),
                
            )
    
    return res_sample.x, res_sample.fun


def cold_routine(qarg, qarg_prep, cold_hamiltonian, N_params):

    #deltat=T/steps
     # this is actually incorrect, this should be reseeded on every optimization loop
    qarg_dupl= qarg.duplicate()
    opt_theta, opt_res = optimization_routine(qarg_dupl, cold_hamiltonian,  qarg_prep, N_params)
    opt_theta = [0.57489003]
    cold_hamiltonian(qarg, qarg_prep, opt_theta)
    
    #print(qarg.qs) 
    res_sample = qarg.get_measurement()

    return res_sample 


def qarg_init(qarg):
    h(qarg) 


qarg_outer = QuantumVariable(len(Q[0]))

########################
#PROBLEM -- F_OPT_CRAB is being reseeded on every function call, this should not be the case!!!! --> make it seed once per run!!
#######################

T = 1
N_Steps =2
N_params=1
res_dict = cold_routine(qarg_outer, qarg_init, trotterization_COLD(steps=N_Steps,T = T), N_params)

print(res_dict)

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

# for tomorrow: 
# further alpha investigations - we have forgot the lambda dependency for H_0, i.e. the mulplicative lambda infront of H_i and H_p
# F_OPT_CRAB seed once per run?? --> move this into the optimization_wrapper!!
# The suggested optimization schemes (Nealder-Mead and Powell are local optimizers, not suited for the problem at hand really (or maybe yes?) 