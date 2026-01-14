"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
********************************************************************************
"""

from jax.lax import while_loop
from jax.extend.core import Literal
from jax.core import ShapedArray

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

# jaspr = make_jaspr(test_function)(100)

# print(jaspr)

# It gives us

# { lambda ; a:QuantumCircuit b:i32[]. let
#     c:QuantumCircuit d:QubitArray = jasp.create_qubits a b
#     e:Qubit = jasp.get_qubit d 0
#     f:QuantumCircuit = jasp.h c e
#     g:i32[] = jasp.get_size d
#     h:i32[] = sub g 1
#     i:i32[] = sub h h
#     j:QuantumCircuit k:i32[] = jasp.q_env[
#       jaspr={ lambda ; l:QuantumCircuit h:i32[] i:i32[] d:QubitArray e:Qubit. let
#           _:i32[] = add h 0
#           m:i32[] = add i 1
#           n:Qubit = jasp.get_qubit d m
#           o:QuantumCircuit = jasp.cx l e n
#           k:i32[] = add i 1
#         in (o, k) }
#       type=JIterationEnvironment
#     ] f h i d e
#     p:QuantumCircuit = jasp.q_env[
#       jaspr={ lambda ; q:QuantumCircuit h:i32[] k:i32[] d:QubitArray e:Qubit. let
#           _:i32[] = add h 0
#           r:i32[] = add k 1
#           s:Qubit = jasp.get_qubit d r
#           t:QuantumCircuit = jasp.cx q e s
#           _:i32[] = add k 1
#         in (t,) }
#       type=JIterationEnvironment
#     ] j h k d e
#     u:QuantumCircuit v:i32[] = jasp.measure p d
#     w:f32[] = integer_pow[y=0] 2.0
#     x:f32[] = convert_element_type[new_dtype=float64 weak_type=False] v
#     y:f32[] = mul x w
#     z:i32[] = convert_element_type[new_dtype=int64 weak_type=False] y
#   in (u, z) }

# For both iterations we see that the q_env primitive receives five inputs
# However for the second iteration, input 0 and input 2 are updated.

# These inputs represent the QuantumCircuit and the loop index. The other
# inputs (QubitArray, loop threshold and the base_qb) stay constant.

# From this information we now need to construct the information to build the
# while primitive.


def iteration_env_evaluator(eqn, context_dic):

    # This represents the case that we are faced with the q_env primitive of the
    # first iteration. We set it as an attribute to use it in the other case.
    if not hasattr(eqn.primitive, "iteration_1_eqn"):
        eqn.primitive.iteration_1_eqn = eqn
        return None

    # We can now retrieve the equations for both iterations.

    # Set the aliases for the equations and the jasprs
    iteration_1_eqn = eqn.primitive.iteration_1_eqn
    iteration_2_eqn = eqn
    iter_1_jaspr = iteration_1_eqn.params["jaspr"].flatten_environments()
    iter_2_jaspr = iteration_2_eqn.params["jaspr"].flatten_environments()

    if len(iter_2_jaspr.outvars) > 1:
        raise Exception("Found jrange with external carry value")

    # Move the loop index to the last argument

    # The loop index is increased in the last equation of the jaspr.
    # We can therefore identify the variable by finding the fist argument
    # of the incrementation equation.
    increment_eq = iter_1_jaspr.eqns[-1]
    arg_pos = iter_1_jaspr.invars.index(increment_eq.invars[0])
    # Move the loop index to the second position in both the jaspr and the equation
    iter_1_jaspr.invars.insert(0, iter_1_jaspr.invars.pop(arg_pos))
    iteration_1_eqn.invars.insert(0, iteration_1_eqn.invars.pop(arg_pos))

    # The same for the jaspr of the second iteration
    increment_eq = iter_2_jaspr.eqns[-1]
    arg_pos = iter_2_jaspr.invars.index(increment_eq.invars[0])
    # Move the loop index to the second position in both the jaspr and the equation
    iter_2_jaspr.invars.insert(0, iter_2_jaspr.invars.pop(arg_pos))
    iteration_2_eqn.invars.insert(0, iteration_2_eqn.invars.pop(arg_pos))

    # Move the loop threshold variable to the last position
    # The loop threshold is demarked as the variable that gets incremented
    # by 0 in the first equation

    demark_eq = iter_1_jaspr.eqns[0]
    arg_pos = iter_1_jaspr.invars.index(demark_eq.invars[0])
    iter_1_jaspr.invars.insert(0, iter_1_jaspr.invars.pop(arg_pos))
    iteration_1_eqn.invars.insert(0, iteration_1_eqn.invars.pop(arg_pos))

    # Same for second iteration
    demark_eq = iter_2_jaspr.eqns[0]
    arg_pos = iter_2_jaspr.invars.index(demark_eq.invars[0])
    iter_2_jaspr.invars.insert(0, iter_2_jaspr.invars.pop(arg_pos))
    iteration_2_eqn.invars.insert(0, iteration_2_eqn.invars.pop(arg_pos))

    # For the jaspr of both iterations we now have the situation that
    # the loop threshold is the argument at the last position and the loop
    # index the argument at the second last position.

    # The way the environment jaspr is collected allows for permuted arguments,
    # ie. the first argument of the first iteration could be the the last argument
    # of the second iteration. We outsource this task to this function.
    verify_semantic_equivalence(iter_1_jaspr, iter_2_jaspr)

    # Now, we figure out which of the return values need to be updated after
    # each iteration. The update needs to happen if the input values of
    # both iterations are not the same. In this case we find the
    # updated value in the output of the first iteration.

    iter_1_invar_hashes = []
    for var in iteration_1_eqn.invars:
        if isinstance(var, Literal):
            iter_1_invar_hashes.append(hash(var.val))
        else:
            iter_1_invar_hashes.append(hash(var))
            
    iter_2_invar_hashes = []
    for var in iteration_2_eqn.invars:
        if isinstance(var, Literal):
            iter_2_invar_hashes.append(hash(var.val))
        else:
            iter_2_invar_hashes.append(hash(var))

    iter_1_outvar_hashes = []
    for var in iteration_1_eqn.outvars:
        if isinstance(var, Literal):
            iter_1_outvar_hashes.append(hash(var.val))
        else:
            iter_1_outvar_hashes.append(hash(var))
            
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
            try:
                res_index = iter_1_outvar_hashes.index(iter_2_invar_hashes[i])
            except ValueError:
                # If the iter 2 invar is not part of the iter 1 outvars
                # and also not part of the iter 1 invars, it is most likely
                # because the user loaded an array from a static list.
                # Loading arrays from static lists is realized via adding
                # a constant to the Jaxpr, which contains the list.
                # Since the loop body is traced twice, this means to constants
                # are added to the Jaxpr and therefore also two variables
                # (even if they represent the same constant).
                # In this case no update is required.
                if isinstance(iteration_2_eqn.invars[i].aval, ShapedArray):
                    update_rules.append(None)
                    continue
                
                raise
            update_rules.append(res_index)
        else:
            # Otherwise append None
            update_rules.append(None)

    # We can now construct the body function of the loop
    # The body function will receive the tuple val which has the signature
    # of the body_jasprs of the iteration environments PLUS the loop cancelation
    # threshold at the last position.

    def body_fun(val):

        # We evaluate the body
        res = iter_1_jaspr.eval(*val)

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
        return tuple(return_values)

    # The condition function should compare whether the loop index (first position)
    # is smaller than the loop cancelation threshold (second position)
    def cond_fun(val):
        return val[1] <= val[0]

    # We now prepare the "init_val" keyword of the loop.

    # For that we extract the invalues for the first iteration
    # Note that the threshold is given as the last argument
    init_val = [context_dic[x] for x in iteration_1_eqn.invars]

    # And evaluate the loop primitive.
    res = while_loop(cond_fun, body_fun, init_val=tuple(init_val))

    # Finally, we insert the result values that receive and update
    # into the context dic
    context_dic[iteration_2_eqn.outvars[-1]] = res[-1]
    return
    # print(iteration_2_eqn)
    # print(update_rules)
    for i in range(len(update_rules) - 1):
        if update_rules[i] is not None:
            iteration_2_eqn.outvars[update_rules[i]]
            context_dic[iteration_2_eqn.outvars[update_rules[i]]] = res[i]


# This function takes two jaxpr objects that perform the same jax semantics
# but have their input signature somehow permuted.
# The function returns the permutation of the arguments
def verify_semantic_equivalence(jaxpr_0, jaxpr_1):

    # This dictionary will contain the translation between the variables
    translation_dic = {}

    # Iterate through the equations

    eqn_list_0 = list(jaxpr_0.eqns)

    # Filter out the += 0 in the second iteration (see jrange.py for more details
    # about this)
    eqn_list_1 = list(jaxpr_1.eqns)  # [1:]
    # translation_dic[jaxpr_0.invars[-1]] = jaxpr_1.eqns[0].outvars[0]

    while eqn_list_0:

        eqn_0 = eqn_list_0.pop(0)
        eqn_1 = eqn_list_1.pop(0)

        # If the equations don't perform the same primitive, the iterations have
        # differing semantics

        if eqn_0.primitive.name != eqn_1.primitive.name:
            if (
                eqn_0.primitive.name == "convert_element_type"
                and eqn_0.invars[0] in jaxpr_0.invars
            ):
                eqn_list_1.insert(0, eqn_1)
                continue
            raise Exception("Jax semantics changed during jrange iteration")

        invars_0 = list(eqn_0.invars)
        invars_1 = list(eqn_1.invars)

        # Check the in-variables of both equations
        for j in range(len(invars_1)):
            var_0 = invars_0[j]
            var_1 = invars_1[j]

            if isinstance(var_0, Literal):
                if not isinstance(var_1, Literal) or var_0.val != var_1.val:
                    raise Exception("Jax semantics changed during jrange iteration")
            elif var_0 in translation_dic:
                if translation_dic[var_0] != var_1:

                    raise Exception("Jax semantics changed during jrange iteration")
            else:
                translation_dic[var_0] = var_1


# Create the environment class, which performs the above logic for environment flattening.
class JIterationEnvironment(QuantumEnvironment):
    def jcompile(self, eqn, context_dic):
        iteration_env_evaluator(eqn, context_dic)