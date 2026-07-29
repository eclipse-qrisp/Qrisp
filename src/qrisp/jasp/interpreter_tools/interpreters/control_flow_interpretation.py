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

from collections.abc import Callable
from typing import Any

import jax.lax
import jax.numpy as jnp
from jax.extend.core import JaxprEqn

from qrisp.jasp.interpreter_tools import (
    ContextDict,
    eval_jaxpr,
    exec_eqn,
    extract_invalues,
    insert_outvalues,
)


def evaluate_cond_eqn(cond_eqn: JaxprEqn, context_dic: ContextDict, eqn_evaluator: Callable = exec_eqn) -> None:
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

    # Deferred import: qc_extraction_interpreter (where ProcessedMeasurement is
    # defined) is loaded after control_flow_interpretation within
    # interpreter_tools.interpreters, so this can't be a top-level import.
    from qrisp.jasp.interpreter_tools.interpreters import ProcessedMeasurement

    if isinstance(invalues[0], ProcessedMeasurement):
        raise Exception("Tried to convert real-time feedback into QuantumCircuit")

    # invalues[0] is the branch index to execute
    branch_jaxpr = cond_eqn.params["branches"][int(invalues[0])]
    res = eval_jaxpr(branch_jaxpr, eqn_evaluator=eqn_evaluator)(*invalues[1:])

    if not isinstance(res, tuple):
        res = (res,)

    insert_outvalues(cond_eqn, context_dic, res)


def evaluate_while_loop(
    while_loop_eqn: JaxprEqn,
    context_dic: ContextDict,
    eqn_evaluator: Callable = exec_eqn,
    break_after_first_iter: bool = False,
) -> None:
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

    # Deferred import: qc_extraction_interpreter (where ProcessedMeasurement is
    # defined) is loaded after control_flow_interpretation within
    # interpreter_tools.interpreters, so this can't be a top-level import.
    from qrisp.jasp.interpreter_tools.interpreters import ProcessedMeasurement

    # Parse parameter structure for constants and carry variables
    num_const_cond_args = while_loop_eqn.params["cond_nconsts"]
    num_const_body_args = while_loop_eqn.params["body_nconsts"]
    overall_constant_amount = num_const_cond_args + num_const_body_args

    def break_condition(invalues: list) -> Any:
        """Helper to evaluate the loop condition jaxpr."""
        constants = invalues[:num_const_cond_args]
        carries = invalues[overall_constant_amount:]

        new_invalues = constants + carries

        res = eval_jaxpr(while_loop_eqn.params["cond_jaxpr"], eqn_evaluator=eqn_evaluator)(*new_invalues)

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

        outvalues = eval_jaxpr(while_loop_eqn.params["body_jaxpr"], eqn_evaluator=eqn_evaluator)(*new_invalues)

        # Update the non-const invalues for the next iteration

        if len(while_loop_eqn.params["body_jaxpr"].jaxpr.outvars) == 1:
            outvalues = (outvalues,)

        invalues = invalues[:overall_constant_amount] + list(outvalues)

        if break_after_first_iter:
            break

    insert_outvalues(while_loop_eqn, context_dic, outvalues)


def evaluate_scan(scan_eq: JaxprEqn, context_dic: ContextDict, eqn_evaluator: Callable = exec_eqn) -> None:
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
    # Stack the results into arrays.
    # If reverse=True, we iterated backwards, so ys_collection contains: [y[N-1], y[N-2], ... y[0]]
    # To match JAX scan semantics (output array index matches input array index),
    # we need to reverse the collection before stacking -> [y[0], ... y[N-1]]
    elif reverse:
        ys = [jnp.stack(col[::-1]) for col in ys_collection]
    else:
        ys = [jnp.stack(col) for col in ys_collection]

    outvalues = list(carry) + ys

    insert_outvalues(scan_eq, context_dic, outvalues)


def evaluate_while_loop_under_trace(
    while_loop_eqn: JaxprEqn, context_dic: ContextDict, eqn_evaluator: Callable = exec_eqn
) -> None:
    """
    Evaluates a JAX while loop equation by reinterpreting the body/condition Jaxprs
    with the given eqn_evaluator and replaying them via ``jax.lax.while_loop``,
    preserving the loop structure under an active trace.

    Unlike :func:`evaluate_while_loop`, which unrolls the loop eagerly using concrete
    Python truthiness (for use outside of tracing mode), this function is meant to be
    called *while* the interpreter itself is being traced (e.g. to compile a profiling
    metric, or to build a post-processing Jaxpr) -- the resulting while_loop primitive
    is a real traced JAX operation.

    Args:
        while_loop_eqn (jax.core.JaxprEqn): The equation representing the while loop.
        context_dic (dict): Dictionary mapping variables to their values.
        eqn_evaluator (function, optional): Function to evaluate equations within the
                                            body/condition Jaxprs. Defaults to exec_eqn.
    """

    invalues = extract_invalues(while_loop_eqn, context_dic)

    body_jaxpr = while_loop_eqn.params["body_jaxpr"]
    cond_jaxpr = while_loop_eqn.params["cond_jaxpr"]
    body_nconsts = while_loop_eqn.params["body_nconsts"]
    cond_nconsts = while_loop_eqn.params["cond_nconsts"]
    overall_constant_amount = body_nconsts + cond_nconsts

    body_eval = eval_jaxpr(body_jaxpr, eqn_evaluator=eqn_evaluator)
    cond_eval = eval_jaxpr(cond_jaxpr, eqn_evaluator=eqn_evaluator)

    def body_fun(val: tuple) -> tuple:
        constants = val[cond_nconsts:overall_constant_amount]
        carries = val[overall_constant_amount:]
        body_res = body_eval(*(constants + carries))
        body_res = body_res if isinstance(body_res, tuple) else (body_res,)
        return val[:overall_constant_amount] + body_res

    def cond_fun(val: tuple) -> Any:
        constants = val[:cond_nconsts]
        carries = val[overall_constant_amount:]
        return cond_eval(*(constants + carries))

    outvalues = jax.lax.while_loop(cond_fun, body_fun, tuple(invalues))[overall_constant_amount:]

    insert_outvalues(while_loop_eqn, context_dic, outvalues)


def evaluate_scan_under_trace(scan_eq: JaxprEqn, context_dic: ContextDict, eqn_evaluator: Callable = exec_eqn) -> None:
    """
    Evaluates a JAX scan equation by reinterpreting the body Jaxpr with the given
    eqn_evaluator and replaying it via ``jax.lax.scan``, preserving the loop
    structure under an active trace.

    Unlike :func:`evaluate_scan`, which unrolls the scan eagerly with a plain Python
    for-loop (for use outside of tracing mode), this function is meant to be called
    *while* the interpreter itself is being traced -- the resulting scan primitive is
    a real traced JAX operation.

    Args:
        scan_eq (jax.core.JaxprEqn): The equation representing the scan operation.
        context_dic (dict): Dictionary mapping variables to their values.
        eqn_evaluator (function, optional): Function to evaluate the scanned body
                                            equation. Defaults to exec_eqn.
    """

    invalues = extract_invalues(scan_eq, context_dic)

    num_consts = scan_eq.params["num_consts"]
    num_carry = scan_eq.params["num_carry"]
    length = scan_eq.params["length"]
    reverse = scan_eq.params.get("reverse", False)
    unroll = scan_eq.params.get("unroll", 1)

    consts = invalues[:num_consts]
    init = invalues[num_consts : num_consts + num_carry]
    xs = invalues[num_consts + num_carry :]

    scan_body = eval_jaxpr(scan_eq.params["jaxpr"], eqn_evaluator=eqn_evaluator)

    if num_consts > 0:

        def wrapped_body(carry: Any, x: Any) -> tuple[Any, tuple]:
            carry_args = list(carry) if isinstance(carry, tuple) else [carry]
            x_args = list(x) if isinstance(x, tuple) else [x]
            args = consts + carry_args + x_args
            result = scan_body(*args)
            if not isinstance(result, tuple):
                result = (result,)
            new_carry = result[0] if num_carry == 1 else result[:num_carry]
            return new_carry, result[num_carry:]

    else:

        def wrapped_body(carry: Any, x: Any) -> tuple[Any, tuple]:
            carry_args = list(carry) if isinstance(carry, tuple) else [carry]
            x_args = list(x) if isinstance(x, tuple) else [x]
            args = carry_args + x_args
            result = scan_body(*args)
            if not isinstance(result, tuple):
                result = (result,)
            new_carry = result[0] if num_carry == 1 else result[:num_carry]
            return new_carry, result[num_carry:]

    xs_arg = xs[0] if len(xs) == 1 else tuple(xs)
    init_arg = init[0] if len(init) == 1 else tuple(init)

    final_carry, ys = jax.lax.scan(wrapped_body, init_arg, xs_arg, length=length, reverse=reverse, unroll=unroll)

    if not isinstance(final_carry, tuple):
        final_carry = (final_carry,)
    if not isinstance(ys, tuple):
        ys = (ys,)

    outvalues = final_carry + ys

    insert_outvalues(scan_eq, context_dic, outvalues)
