# %%
import jax
import jax.numpy as jnp
import networkx as nx
from qrisp.my_jasp_test.jax_coby import cobyla
from qrisp.jasp import jaspify
from qrisp import QuantumFloat, QuantumVariable, rz, rzz, h, rx, expectation_value,x
from itertools import combinations
import math

"""
This has not really been justified, due to unclarities with the maxclique formulation
--> same problems are bound to arise here

"""


@jax.jit
def extract_boolean_digit(integer, digit):
    return jnp.bool((integer>>digit & 1))

# %%

def maxsat_post_process(problem):
    clauses = jnp.array(problem[1])
    n_nodes = problem[0]
    def maxsat_process_inner(result):
       
        bool_arr= jnp.array([extract_boolean_digit(jnp.uint8(result),i) for i in range(n_nodes)]) 
        jax.debug.print("{result} result", result=result)
        def process_index(index):
            
            return_val = jnp.where(index > 0, 1 - bool_arr[index - 1], bool_arr[-index - 1])
            jax.debug.print("{return_val} return_val", return_val=return_val)
            return return_val
        clause_values = jax.lax.map(lambda clause: jnp.prod(jax.lax.map(process_index,clause)),clauses)
        #clause_values = jax.vmap(lambda clause: jnp.prod(jax.vmap(process_index)(clause)))(clauses)
        cost = -jnp.sum(1 - clause_values)

        return -cost
    
    return maxsat_process_inner

from qrisp.algorithms.qaoa.problems.maxSat import create_maxsat_cost_polynomials
from qrisp import app_sb_phase_polynomial


# this one needs jaspification
def create_maxsat_jax_cost_op(problem, depth):
    n_vars = problem[0]
    def maxsat_jax_cost_op(gamma_beta):
        qf = QuantumFloat(n_vars)

        #init_State
        for pa in range(n_vars):
            h(qf[pa])

        for pa in range(n_vars):
            rx(2 * gamma_beta[0], qf[pa])

        cost_polynomials, symbols = create_maxsat_cost_polynomials(problem)

        def cost_operator(qv, gamma):
            for P in cost_polynomials:
                app_sb_phase_polynomial([qv], -P, symbols, t=gamma)

        cost_operator(qf,gamma_beta[1])


        return qf
    
    return maxsat_jax_cost_op

def massat_obj(G,depth):
    def massat_obj_inner(x_angles):
        res = expectation_value(
            #state_prep = apply_rx, 
            state_prep = create_maxsat_jax_cost_op(G,depth),
            #state_prep = mco,
            shots = 1000, 
            post_processor=maxsat_post_process(G))(x_angles)
        return res

    return massat_obj_inner

@jaspify(terminal_sampling=True)
def main():
    Depth = 1

    clauses = [[1,2,-3],[1,4,-6],[4,5,6],[1,3,-4],[2,4,5],[1,3,5],[-2,-3,6]]
    num_vars = 6
    problem = (num_vars, clauses)
    key = jax.random.key(11)
    x0 = jax.random.uniform(key= key, shape=(2*Depth,))
    obj_func = massat_obj(problem, depth=Depth)
    result, value = cobyla(obj_func, x0, cons=[],maxfun=100) 
    return result,value
result, value = main()
print("Optimal solution:", result)
print("Optimal value:", value)

# %% 




""" import jax

import jax.numpy as jnp

def compute_cost(state, clauses):
    def process_index(index):
        return jnp.where(index > 0, 1 - state[index - 1], state[-index - 1])

    clause_values = jax.vmap(lambda clause: jnp.prod(jax.vmap(process_index)(clause)))(clauses)
    cost = -jnp.sum(1 - clause_values)
    return cost

jit_compute_cost = jax.jit(compute_cost) """

""" def create_maxsat_cl_cost_function(problem):

    clauses = problem[1]
    def cl_cost_function(res_dic):
        cost = 0
        for state, prob in res_dic.items():
            for clause in clauses:
                cost += -(1-math.prod((1-int(state[index-1])) if index>0 else int(state[-index-1]) for index in clause))*prob

        return cost

    return cl_cost_function  """
# %%
