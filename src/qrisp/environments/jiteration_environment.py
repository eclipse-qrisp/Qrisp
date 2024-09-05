"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

from jax.lax import while_loop
from jax.core import Literal

from qrisp.environments import QuantumEnvironment


# The idea behind this equation compiler is the following:
# The jrange feature executes 2 iterations of the loop to capture
# the loop body in an QuantumEnvironment. We need 2 iterations to
# determine which inputs are updated each iteration and which inputs
# stay the same.
# To compile this structure into a while primitive, we need to analyze both
# environments.

# For instance consider the following code:
    
# from qrisp import *

# def test_function(i):
    
#     qv = QuantumVariable(i)
#     h(qv[0])
    
#     base_qb = qv[0]
#     for i in jrange(qv.size-1):
#         cx(base_qb, qv[i+1])
#     return measure(qv)

# jispr = make_jispr(test_function)(100)

# print(jispr)

# It gives us 

# { lambda ; a:QuantumCircuit b:i32[]. let
#     c:QuantumCircuit d:QubitArray = create_qubits a b
#     e:Qubit = get_qubit d 0
#     f:QuantumCircuit = h c e
#     g:i32[] = get_size d
#     h:i32[] = sub g 1
#     i:QuantumCircuit j:i32[] = q_env[
#       jispr={ lambda ; k:QuantumCircuit e:Qubit h:i32[] d:QubitArray. let
#           l:i32[] = add h 1
#           m:Qubit = get_qubit d l
#           n:QuantumCircuit = cx k e m
#           j:i32[] = add h 1
#         in (n, j) }
#       type=JIterationEnvironment
#     ] f e h d
#     o:QuantumCircuit = q_env[
#       jispr={ lambda ; p:QuantumCircuit e:Qubit j:i32[] d:QubitArray. let
#           q:i32[] = add j 1
#           r:Qubit = get_qubit d q
#           s:QuantumCircuit = cx p e r
#           _:i32[] = add j 1
#         in (s,) }
#       type=JIterationEnvironment
#     ] i e j d
#     t:QuantumCircuit u:i32[] = measure o d
#   in (t, u) }

# For both iterations we see that the q_env primitive receives four inputs
# However for the second iteration, input 0 and input 2 are updated.

# These inputs represent the loop index and the QuantumCircuit. The other
# two inputs (QubitArray and the base_qb) stay constant.

# From this information we now need to construct the information to build the
# while primitive.

def iteration_env_evaluator(eqn, context_dic):

    # This represents the case that we are faced with the q_env primitive of the
    # first iteration. We set it as an attribute to use it in the other case.
    if not hasattr(eqn.primitive, "iteration_1_eqn"):
        eqn.primitive.iteration_1_eqn = eqn
        return None

    # We can now retrieve the equations for both iterations.
    
    # Set the aliases for the equations and the jisprs
    iteration_1_eqn = eqn.primitive.iteration_1_eqn
    iteration_2_eqn = eqn
    iter_1_jispr = iteration_1_eqn.params["jispr"]
    iter_2_jispr = iteration_2_eqn.params["jispr"]
    
    
    # Move the loop index to the last argument
    
    # The loop index is increased in the last equation of the jispr.
    # We can therefore identify the variable by finding the fist argument
    # of the incrementation equation.
    increment_eq = iter_1_jispr.eqns[-1]
    arg_pos = iter_1_jispr.invars.index(increment_eq.invars[0])
    
    # Move the index to the last position in both the jispr and the equation
    iter_1_jispr.invars.append(iter_1_jispr.invars.pop(arg_pos))
    iteration_1_eqn.invars.append(iteration_1_eqn.invars.pop(arg_pos))
    
    # The way the environment jispr is collected allows for permuted arguments,
    # ie. the first argument of the first iteration could be the the last argument
    # of the second iteration. We outsource this task to this function.
    permutation = find_signature_permutation(iter_1_jispr, iter_2_jispr)
    
    # Find the permuted variables and update the list for the jispr
    permuted_invars = [iter_2_jispr.invars[i] for i in permutation]
    iter_2_jispr.invars.clear()
    iter_2_jispr.invars.extend(permuted_invars)
    
    # Find the permuted variables and update the list for the equation
    permuted_invars = [iteration_2_eqn.invars[i] for i in permutation]
    iteration_2_eqn.invars.clear()
    iteration_2_eqn.invars.extend(permuted_invars)
    
    # Now, we figure out which of the return values need to be updated after 
    # each iteration. The update needs to happen if the input values of 
    # both iterations are not the same. In this case we find the 
    # updated value in the output of the first iteration.
    
    iter_1_invar_hashes = [hash(x) for x in iteration_1_eqn.invars]
    iter_2_invar_hashes = [hash(x) for x in iteration_2_eqn.invars]
    iter_1_outvar_hashes = [hash(x) for x in iteration_1_eqn.outvars]
    
    # This list will contain the information about which variables need to be
    # updated. If the entry is None, no update is required. Otherwise the entry
    # is an integer indicating which return value of iteration 1 is used for
    # the update.
    update_rules = []
    
    # Iterate through the input variables of the equation of iteration 1
    for i in range(len(iteration_1_eqn.invars)):
        
        # If the input variable is not in the input variables of iteration 2,
        # it needs to be updated.
        if iter_1_invar_hashes[i] not in iter_2_invar_hashes:
            
            # Find the index in the outvars of iteration 1
            res_index = iter_1_outvar_hashes.index(iter_2_invar_hashes[i])
            update_rules.append(res_index)
        else:
            # Otherwise append None
            update_rules.append(None)
    
    
    # We can now construct the body function of the loop
    # The body function will receive the tuple val which has the signature
    # of the body_jisprs of the iteration environments PLUS the loop cancelation
    # threshold at the last position.
    
    def body_fun(val):
        
        # We evaluate the body (without the loop cancelation treshold).
        res = iter_1_jispr.eval(*val[:-1])
        
        # Convert the result into a tuple if it isn't one already
        if not isinstance(res, tuple):
            res = (res,)
        
        # Collect the return values.
        # For the values that are updated after each iteration, we need to return
        # those.
        return_values = []
        
        for i in range(len(update_rules)):
            if update_rules[i] is None:
                return_values.append(val[i])
            else:
                return_values.append(res[update_rules[i]])
        
        # Return the appropriate values (with the cancelation threshold).
        return tuple(return_values + [val[-1]])
    
    # The condition function should compare whether the loop index (second last position)
    # is smaller than the loop cancelation threshold (last position)
    def cond_fun(val):
        return val[-2] < val[-1]
    
    # We now prepare the "init_val" keyword of the loop.
    
    # For that we extract the invalues for the first iteration
    # Note that the treshold is given as the last argument
    init_val = [context_dic[x] for x in iteration_1_eqn.invars]
    
    # We insert the looping index (starts at 0)
    init_val.insert(-1, 0)
    
    # And evaluate the loop primitive.
    res = while_loop(cond_fun, body_fun, init_val = tuple(init_val))
    
    # Finally, we insert the result values that receive and update 
    # into the context dic
    for i in range(len(update_rules)-1):
        if update_rules[i] is not None:
            iteration_2_eqn.outvars[update_rules[i]]
            context_dic[iteration_2_eqn.outvars[update_rules[i]]] = res[i]


# This function takes two jaxpr objects that perform the same jax semantics
# but have their input signature somehow permuted.
# The function returns the permutation of the arguments
def find_signature_permutation(jaxpr_0, jaxpr_1):
    
    # This dictionary will contain the translation between the variables
    translation_dic = {}
    
    # Iterate through the equations
    for i in range(len(jaxpr_0.eqns)):
        eqn_0 = jaxpr_0.eqns[i]
        eqn_1 = jaxpr_1.eqns[i]
        
        # If the equations don't perform the same primitive, the iterations have
        # differing semantics
        if eqn_0.primitive.name != eqn_1.primitive.name:
            raise Exception("Jax semantics changed during jrange iteration")
        
        # Check the in-variables of both equations
        for j in range(len(eqn_0.invars)):
            var = eqn_0.invars[j]
            if isinstance(var, Literal):
                if not isinstance(eqn_1.invars[j], Literal) or eqn_1.invars[j].val != var.val:
                    raise Exception("Jax semantics changed during jrange iteration")
            elif var in translation_dic:
                if translation_dic[var] != eqn_1.invars[j]:
                    raise Exception("Jax semantics changed during jrange iteration")
            else:
                translation_dic[var] = eqn_1.invars[j]
    
    # Compute the permutation of the invars
    return [jaxpr_1.invars.index(translation_dic[var]) for var in jaxpr_0.invars]

# Create the environment class, which performs the above logic for environment flattening.
class JIterationEnvironment(QuantumEnvironment):
    def jcompile(self, eqn, context_dic):
        iteration_env_evaluator(eqn, context_dic)