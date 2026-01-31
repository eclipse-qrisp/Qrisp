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

import jax.numpy as jnp

from qrisp.jasp.interpreter_tools import (
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
    exec_eqn,
)




def evaluate_cond_eqn(cond_eqn, context_dic, eqn_evaluator=exec_eqn):
    """
    Evaluates a JAX condition equation within the context of the JASP interpreter.
    
    This function handles the branching logic of jax.lax.cond or similar primitives. 
    It determines which branch to execute based on the condition variable.
    
    Args:
        cond_eqn (jax.core.JaxprEqn): The equation representing the condition.
        context_dic (dict): Dictionary mapping variables to their values in the current context.
        eqn_evaluator (function, optional): The function used to evaluate individual equations 
                                            within the branches. Defaults to exec_eqn.

    Raises:
        Exception: If the condition variable depends on a Qrisp ProcessedMeasurement (real-time feedback), 
                   which cannot be resolved during circuit generation/interpretation.
    """
    
    # Extract the invalues from the context dic
    invalues = extract_invalues(cond_eqn, context_dic)
    
    from qrisp.jasp.jasp_expression import ProcessedMeasurement
    
    if isinstance(invalues[0], ProcessedMeasurement):
        raise Exception("Tried to convert real-time feedback into QuantumCircuit")

    # Iterate through branches to find the one matching the condition index (invalues[0])
    for i in range(len(cond_eqn.params["branches"])):
        if int(invalues[0]) == i:
            res = eval_jaxpr(
                cond_eqn.params["branches"][i], eqn_evaluator=eqn_evaluator
            )(*invalues[1:])
            break

    if not isinstance(res, tuple):
        res = (res,)

    insert_outvalues(cond_eqn, context_dic, res)


def evaluate_while_loop(
    while_loop_eqn, context_dic, eqn_evaluator=exec_eqn, break_after_first_iter=False
):
    """
    Evaluates a JAX while loop equation within the context of the JASP interpreter.

    This handles `jax.lax.while_loop`, performing iterations as long as the condition function
    returns True.

    Args:
        while_loop_eqn (jax.core.JaxprEqn): The equation representing the while loop.
        context_dic (dict): Dictionary mapping variables to their values.
        eqn_evaluator (function, optional): Function to evaluate equations within the loop body.
        break_after_first_iter (bool, optional): Debugging flag to force exit after one iteration. Defaults to False.

    Raises:
        Exception: If the loop condition depends on a Qrisp ProcessedMeasurement.
    """
    
    from qrisp.jasp.jasp_expression import ProcessedMeasurement

    # Parse parameter structure for constants and carry variables
    num_const_cond_args = while_loop_eqn.params["cond_nconsts"]
    num_const_body_args = while_loop_eqn.params["body_nconsts"]
    overall_constant_amount = num_const_cond_args + num_const_body_args
    
    def break_condition(invalues):
        """Helper to evaluate the loop condition jaxpr."""
        constants = invalues[:num_const_cond_args]
        carries = invalues[overall_constant_amount:]
        
        new_invalues = constants + carries
        
        res = eval_jaxpr(
            while_loop_eqn.params["cond_jaxpr"], eqn_evaluator=eqn_evaluator
        )(*new_invalues)
        
        if isinstance(res, ProcessedMeasurement):
            raise Exception("Tried to convert real-time feedback into QuantumCircuit")
        
        return res

    # Extract the invalues from the context dic
    invalues = extract_invalues(while_loop_eqn, context_dic)
    outvalues = invalues[overall_constant_amount:]

    while break_condition(invalues):
        
        constants = invalues[num_const_cond_args:overall_constant_amount]
        carries = invalues[overall_constant_amount:]
        
        new_invalues = constants + carries

        outvalues = eval_jaxpr(
            while_loop_eqn.params["body_jaxpr"], eqn_evaluator=eqn_evaluator
        )(*new_invalues)

        # Update the non-const invalues for the next iteration
        
        if len(while_loop_eqn.params["body_jaxpr"].jaxpr.outvars) == 1:
            outvalues = (outvalues,)
        
        invalues = invalues[:overall_constant_amount] + list(outvalues)

        if break_after_first_iter:
            break

    insert_outvalues(while_loop_eqn, context_dic, outvalues)


def evaluate_scan(scan_eq, context_dic, eqn_evaluator=exec_eqn):
    """
    Evaluates a JAX scan equation within the context of the JASP interpreter.

    This handles `jax.lax.scan` (and `jax.lax.map` which lowers to scan). It iterates
    over input arrays, applying a function that carries state, and stacks the outputs.

    Args:
        scan_eq (jax.core.JaxprEqn): The equation representing the scan operation.
        context_dic (dict): Dictionary mapping variables to their values.
        eqn_evaluator (function, optional): Function to evaluate the scanned body equation.
    """

    invalues = extract_invalues(scan_eq, context_dic)

    f = eval_jaxpr(scan_eq.params["jaxpr"], eqn_evaluator=eqn_evaluator)

    length = scan_eq.params["length"]
    reverse = scan_eq.params.get("reverse", False)

    carry_amount = scan_eq.params["num_carry"]
    const_amount = scan_eq.params["num_consts"]

    # Separating inputs: constants (pushed into body but not iterated), 
    # initial carry, and scanned inputs (arrays to be sliced).
    init = invalues[const_amount : carry_amount + const_amount]
    scan_invalues = invalues[const_amount + carry_amount :]

    # Setup the iterator over the length dimension.
    # Note: JAX scan behavior with reverse=True means we consume inputs from the end
    # and produce outputs that correspond to those inputs (so effectively reversed output),
    # but the 'carry' evolves in the reverse direction.
    iterator = range(length)
    if reverse:
        iterator = reversed(iterator)

    # Slice all scanned inputs along the leading dimension (0).
    xs = []
    for i in iterator:
        xs.append([val[i] for val in scan_invalues])

    carry = init
    consts = invalues[:const_amount]
    
    # Store outputs for each iteration to be stacked later.
    ys_collection = None

    for x in xs:
        # Construct arguments: constants definitions, current carry, and slice of scanned inputs
        args = consts + list(carry) + x

        res = f(*args)

        if not isinstance(res, tuple):
            res = (res,)

        # Scan body returns (new_carry, output_slice)
        carry = res[:carry_amount]
        y = res[carry_amount:]

        # Initialize collections if first iteration
        if ys_collection is None:
            ys_collection = [[] for _ in range(len(y))]

        for i, val in enumerate(y):
            ys_collection[i].append(val)

    if ys_collection is None:
        # Handle length=0 case or no-output case
        # We need to look at output variables to determine correct shapes and dtypes for empty results.
        jaxpr = scan_eq.params["jaxpr"].jaxpr
        outvars = jaxpr.outvars
        y_vars = outvars[carry_amount:]
        ys = []
        for v in y_vars:
            # Result is empty along scanned dimension (length=0) plus element shape
            shape = (length,) + v.aval.shape
            ys.append(jnp.zeros(shape, dtype=v.aval.dtype))
    else:
        # Stack the results into arrays.
        # If reverse=True, we iterated backwards, so ys_collection contains: [y[N-1], y[N-2], ... y[0]]
        # To match JAX scan semantics (output array index matches input array index), 
        # we need to reverse the collection before stacking -> [y[0], ... y[N-1]]
        if reverse:
            ys = [jnp.stack(col[::-1]) for col in ys_collection]
        else:
            ys = [jnp.stack(col) for col in ys_collection]

    outvalues = list(carry) + ys

    insert_outvalues(scan_eq, context_dic, outvalues)
