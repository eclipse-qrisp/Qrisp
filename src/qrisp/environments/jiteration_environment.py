"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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


# The jrange feature executes 2 iterations of the loop to capture
# the loop body in a QuantumEnvironment. We need 2 iterations to
# determine which inputs are updated each iteration and which inputs
# stay the same.
# To compile this structure into a while primitive, we need to analyze both
# environments.
#
# Each environment body contains a named marker jit equation inserted
# by the jrange iterator right before __exit__:
#
#   jit[name=_jrange_marker jaxpr=_jrange_marker] updated_idx threshold
#
#   invars[0] = updated loop index (result of add idx 1)
#   invars[1] = threshold (stop value)
#
# This marker serves as a stable identification anchor for the
# threshold (invars[1]) and the increment (found via the add equation
# that feeds invars[0]).
#
#
# For instance consider the following code:
#
# from qrisp import *
#
# def test_function(i):
#
#     qv = QuantumVariable(i)
#     h(qv[0])
#
#     base_qb = qv[0]
#     for i in jrange(qv.size-1):
#         cx(base_qb, qv[i+1])
#     return measure(qv)
#
# jaspr = make_jaspr(test_function, flatten_envs=False)(100)
#
# print(jaspr)
#
# It gives us (abbreviated):
#
# let _jrange_marker = { lambda ; a:i64[] b:i64[]. let  in (a,) } in
# { lambda ; c:QuantumState. let
#     ...
#     j:QuantumCircuit k:i32[] = jasp.q_env[
#       jaspr={ lambda ; l:QuantumCircuit h:i32[] i:i32[] d:QubitArray e:Qubit. let
#           m:i32[] = add i 1
#           n:Qubit = jasp.get_qubit d m
#           o:QuantumCircuit = jasp.cx l e n
#           k:i32[] = jit[name=_jrange_marker jaxpr=_jrange_marker] m h
#         in (o, k) }
#       type=JIterationEnvironment
#     ] f h i d e
#     p:QuantumCircuit = jasp.q_env[
#       jaspr={ lambda ; q:QuantumCircuit h:i32[] k:i32[] d:QubitArray e:Qubit. let
#           r:i32[] = add k 1
#           s:Qubit = jasp.get_qubit d r
#           t:QuantumCircuit = jasp.cx q e s
#           _:i32[] = jit[name=_jrange_marker jaxpr=_jrange_marker] r h
#         in (t,) }
#       type=JIterationEnvironment
#     ] j h k d e
#     ...
#   in (...) }
#
# For both iterations we see that the q_env primitive receives five inputs.
# The marker equation lets us robustly identify:
#   - The threshold (invars[1] of the marker)
#   - The loop index before increment (invars[0] of the add that feeds
#     the marker)
#   - The updated loop index position in the body outvars (marker's outvar)
#
# From this information the while primitive is constructed.


def iteration_env_evaluator(eqn, context_dic):
    """Process a JIterationEnvironment equation during environment flattening.

    Each jrange loop emits two consecutive ``q_env`` equations (iteration 1
    and iteration 2).  This function is called once per equation.  The first
    call stores the equation; the second call retrieves it, pairs the two
    iterations, and compiles them into a JAX ``while_loop`` primitive.
    """

    # First call: store the equation keyed by the primitive's identity.
    # The primitive is the JIterationEnvironment instance, which is unique
    # to this jrange loop, so the key distinguishes different loops.
    if id(eqn.primitive) not in context_dic:
        context_dic[id(eqn.primitive)] = eqn
        return None

    # Second call: retrieve the first equation and remove the entry.
    iteration_1_eqn = context_dic[id(eqn.primitive)]
    del context_dic[id(eqn.primitive)]
    iteration_2_eqn = eqn

    # Flatten any nested environments inside the loop bodies.
    iter_1_jaspr = iteration_1_eqn.params["jaspr"].flatten_environments()
    iter_2_jaspr = iteration_2_eqn.params["jaspr"].flatten_environments()

    # The second iteration's body must return exactly one value (the
    # QuantumCircuit).  Multiple outputs would indicate an unsupported
    # external carry value escaping the loop.
    if len(iter_2_jaspr.outvars) > 1:
        raise Exception("Found jrange with external carry value")

    # ------------------------------------------------------------------
    # Identify the loop index and threshold variables via the named
    # marker jit equation inserted right before __exit__ by jrange.
    #
    # IMPORTANT: we find the *original* Vars from the unflattened body
    # Jaspr because those are the same objects that appear in
    # iteration_1_eqn.invars / iteration_2_eqn.invars.  The flattened
    # body (iter_1_jaspr / iter_2_jaspr) has *new* Vars from the
    # reinterpret pass and cannot be used for invar rearrangement.
    #
    # The marker has signature  marker(updated_loop_index, threshold)
    #   invars[0] = updated loop index (output of the add increment)
    #   invars[1] = threshold (stop value)
    # ------------------------------------------------------------------
    from qrisp.jasp.program_control.jrange_iterator import JRANGE_MARKER_NAME

    def _find_marker_eqn(eqns):
        # The marker is the last equation in the body, so scan backward.
        for eqn in reversed(eqns):
            if eqn.primitive.name == "jit":
                if eqn.params.get("name") == JRANGE_MARKER_NAME:
                    return eqn
        raise Exception(
            f"JIterationEnvironment: could not find marker equation "
            f"'{JRANGE_MARKER_NAME}' in body Jaxpr."
        )

    # --- Use ORIGINAL (unflattened) bodies for Var identification ---
    # These Vars match iteration_*_eqn.invars.
    orig_body_1 = iteration_1_eqn.params["jaspr"]
    orig_body_2 = iteration_2_eqn.params["jaspr"]

    def _find_vars_from_marker(body, eqn_invars):
        """Return (threshold_var, loop_index_var) from a body Jaspr.
        The returned Vars are looked up in *eqn_invars* so they match
        the equation's invar list (original or flattened)."""
        marker = _find_marker_eqn(body.eqns)
        thresh = marker.invars[1]
        updated = marker.invars[0]
        # Find the add that feeds the marker (immediately preceding it,
        # so scan backward).
        loop = None
        for eqn in reversed(body.eqns):
            if eqn.outvars[0] is updated:
                loop = eqn.invars[0]
                break
        if loop is None:
            raise Exception(
                "Could not find increment equation feeding the jrange marker."
            )
        # Verify the vars are in the target invars list
        if thresh not in eqn_invars:
            raise Exception("Threshold var not found in equation invars.")
        if loop not in eqn_invars:
            raise Exception("Loop index var not found in equation invars.")
        return thresh, loop

    # Original Vars → for iteration_*_eqn.invars
    threshold_var, loop_index_var = _find_vars_from_marker(
        orig_body_1, iteration_1_eqn.invars
    )
    threshold_var_2, loop_index_var_2 = _find_vars_from_marker(
        orig_body_2, iteration_2_eqn.invars
    )

    # Flattened Vars → for iter_*_jaspr.invars
    thresh_flat_1, loop_flat_1 = _find_vars_from_marker(
        iter_1_jaspr, iter_1_jaspr.invars
    )
    thresh_flat_2, loop_flat_2 = _find_vars_from_marker(
        iter_2_jaspr, iter_2_jaspr.invars
    )

    # --- inc_res_index: from the flattened body's marker outvar ---
    marker_flat_1 = _find_marker_eqn(iter_1_jaspr.eqns)
    inc_outvar = marker_flat_1.outvars[0]
    inc_res_index = None
    for i, ov in enumerate(iter_1_jaspr.jaxpr.outvars):
        if ov is inc_outvar:
            inc_res_index = i
            break
    if inc_res_index is None:
        raise Exception("Could not find marker output in iteration 1 outvars.")

    # --- Rearrange: threshold at position 0, loop index at position 1 ---
    def _move_var_to_front(invars_list, target_var):
        """Move *target_var* to index 0 in *invars_list* (mutates in place)."""
        arg_pos = invars_list.index(target_var)
        invars_list.insert(0, invars_list.pop(arg_pos))

    # Iteration 1: flattened Jaspr invars + original equation invars
    _move_var_to_front(iter_1_jaspr.invars, loop_flat_1)
    _move_var_to_front(iteration_1_eqn.invars, loop_index_var)

    _move_var_to_front(iter_1_jaspr.invars, thresh_flat_1)
    _move_var_to_front(iteration_1_eqn.invars, threshold_var)

    # Iteration 2
    _move_var_to_front(iter_2_jaspr.invars, loop_flat_2)
    _move_var_to_front(iteration_2_eqn.invars, loop_index_var_2)

    _move_var_to_front(iter_2_jaspr.invars, thresh_flat_2)
    _move_var_to_front(iteration_2_eqn.invars, threshold_var_2)

    # After rearrangement: position 0 = threshold, position 1 = loop index.
    # Verify that both iterations have the same semantics despite possible
    # argument permutation.
    verify_semantic_equivalence(iter_1_jaspr, iter_2_jaspr)

    # Now, we figure out which of the return values need to be updated after
    # each iteration. We use structural information and Var identity:
    # - Position 0 (threshold): never updated (pass-through)
    # - Position 1 (loop index): always updated from the increment marker eqn
    # - Other positions: use identity comparison (Var is Var) instead of
    #   hash(var), because hash(var) depends on id(var) which can vary
    #   non-deterministically across JAX trace cache lifetimes, leading to
    #   false "unchanged" classifications and ultimately infinite loops.

    # inc_res_index was already computed above from the marker equation.

    # This list will contain the information about which variables need to be
    # updated. If the entry is None, no update is required. Otherwise the entry
    # is an integer indicating which return value of iteration 1 is used for
    # the update.
    update_rules = []

    # Iterate through the input variables of the equation of iteration 1
    for i in range(len(iteration_1_eqn.invars)):
        # Position 0: threshold — never updated (pass-through)
        if i == 0:
            update_rules.append(None)
        # Position 1: loop index — always updated from increment eqn
        elif i == 1:
            update_rules.append(inc_res_index)
        # Other positions: use identity comparison (Var is Var)
        # JAX reuses the same Var object for unchanged variables within a
        # single Jaxpr, so identity comparison is reliable and deterministic.
        elif iteration_1_eqn.invars[i] is not iteration_2_eqn.invars[i]:
            # Variable was updated — find which output of iteration 1
            # provides the new value for iteration 2's input.
            try:
                res_index = iteration_1_eqn.outvars.index(iteration_2_eqn.invars[i])
            except ValueError:
                # If the iter 2 invar is not part of the iter 1 outvars
                # and also not part of the iter 1 invars, it is most likely
                # because the user loaded an array from a static list.
                # Loading arrays from static lists is realized via adding
                # a constant to the Jaxpr, which contains the list.
                # Since the loop body is traced twice, this means two constants
                # are added to the Jaxpr and therefore also two variables
                # (even if they represent the same constant).
                # In this case no update is required.
                if isinstance(iteration_2_eqn.invars[i].aval, ShapedArray):
                    update_rules.append(None)
                    continue

                raise
            update_rules.append(res_index)
        else:
            # Variable unchanged (same Var object in both iterations)
            update_rules.append(None)

    # Construct the body function of the while loop.
    # val has the same structure as init_val: position 0 = threshold,
    # position 1 = loop index, remaining positions = other carried values.

    def body_fun(val):

        # We evaluate the body
        res = iter_1_jaspr.eval(*val)

        # Convert the result into a tuple if it isn't one already
        if not isinstance(res, tuple):
            res = (res,)

        # Collect the return values.
        return_values = []

        for i in range(len(update_rules)):
            if update_rules[i] is None:
                return_values.append(val[i])
            else:
                return_values.append(res[update_rules[i]])

        # Return the appropriate values (with the cancelation threshold).
        return tuple(return_values)

    # The condition function: continue while loop_index <= threshold.
    # Position 0 = threshold, position 1 = loop index.
    def cond_fun(val):
        return val[1] <= val[0]

    # Build the initial loop state from the first iteration's inputs.
    # After rearrangement: position 0 = threshold, position 1 = loop index.
    init_val = [context_dic[x] for x in iteration_1_eqn.invars]

    # And evaluate the loop primitive.
    res = while_loop(cond_fun, body_fun, init_val=tuple(init_val))

    # Finally, we insert the result values that receive and update
    # into the context dic.
    # Since we already verified that iter_2_eqn only has a single
    # output value, we can be sure that there are no "external carry values"
    # Therefore, only the QuantumState variable will be used in
    # the remainder of the program.
    context_dic[iteration_2_eqn.outvars[-1]] = res[-1]
    return


# This function takes two jaxpr objects that perform the same jax semantics
# but have their input signature somehow permuted.
# The function returns the permutation of the arguments
def verify_semantic_equivalence(jaxpr_0, jaxpr_1):

    # This dictionary will contain the translation between the variables
    translation_dic = {}

    # Iterate through the equations

    eqn_list_0 = list(jaxpr_0.eqns)

    eqn_list_1 = list(jaxpr_1.eqns)

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
