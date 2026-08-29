# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Evaluates cond/while/scan as real traced JAX ops, with flatten/unflatten helpers for quantum types."""

from collections.abc import Callable, Sequence
from typing import Any

import jax.lax
from jax import Array
from jax.extend.core import JaxprEqn, Literal, Var

from qrisp.jasp.interpreter_tools import (
    ContextDict,
    Jlist,
    eval_jaxpr,
    exec_eqn,
    extract_invalues,
    insert_outvalues,
)
from qrisp.jasp.primitives import AbstractQuantumState, AbstractQubitArray


def evaluate_cond_under_trace(cond_eqn: JaxprEqn, context_dic: ContextDict, eqn_evaluator: Callable = exec_eqn) -> None:
    """Evaluate a JAX cond equation while preserving its branch structure under an active trace.

    Reinterprets each branch Jaxpr with the given eqn_evaluator and replays it via
    ``jax.lax.switch``. Unlike :func:`~qrisp.jasp.interpreter_tools.interpreters.control_flow_interpretation.evaluate_cond_eqn`,
    which picks and evaluates a single branch eagerly using a concrete Python index
    (for use outside of tracing mode), this function is meant to be called *while*
    the interpreter itself is being traced -- the resulting switch primitive is a
    real traced JAX operation.

    Parameters
    ----------
    cond_eqn : jax.extend.core.JaxprEqn
        The equation representing the conditional.
    context_dic : ContextDict
        Dictionary mapping variables to their values.
    eqn_evaluator : Callable, optional
        Function to evaluate equations within each branch Jaxpr. The default
        is exec_eqn.

    """
    invalues = extract_invalues(cond_eqn, context_dic)

    branch_fns = [eval_jaxpr(branch_jaxpr, eqn_evaluator=eqn_evaluator) for branch_jaxpr in cond_eqn.params["branches"]]

    # invalues[0] is the branch index/predicate encoding; the remaining invalues
    # are the operands/carries passed to whichever branch ends up selected.
    outvalues = jax.lax.switch(invalues[0], branch_fns, *invalues[1:])
    outvalues = (outvalues,) if len(cond_eqn.outvars) == 1 else outvalues

    insert_outvalues(cond_eqn, context_dic, outvalues)


def evaluate_while_loop_under_trace(
    while_loop_eqn: JaxprEqn, context_dic: ContextDict, eqn_evaluator: Callable = exec_eqn
) -> None:
    """Evaluate a JAX while-loop equation while preserving its loop structure under an active trace.

    Reinterprets the body/condition Jaxprs with the given eqn_evaluator and replays
    them via ``jax.lax.while_loop``. Unlike :func:`~qrisp.jasp.interpreter_tools.interpreters.control_flow_interpretation.evaluate_while_loop`,
    which unrolls the loop eagerly using concrete Python truthiness (for use outside
    of tracing mode), this function is meant to be called *while* the interpreter
    itself is being traced (e.g. to compile a profiling metric, or to build a
    post-processing Jaxpr) -- the resulting while_loop primitive is a real traced
    JAX operation.

    Parameters
    ----------
    while_loop_eqn : jax.extend.core.JaxprEqn
        The equation representing the while loop.
    context_dic : ContextDict
        Dictionary mapping variables to their values.
    eqn_evaluator : Callable, optional
        Function to evaluate equations within the body/condition Jaxprs. The
        default is exec_eqn.

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
    """Evaluate a JAX scan equation while preserving its loop structure under an active trace.

    Reinterprets the body Jaxpr with the given eqn_evaluator and replays it via
    ``jax.lax.scan``. Unlike :func:`~qrisp.jasp.interpreter_tools.interpreters.control_flow_interpretation.evaluate_scan`,
    which unrolls the scan eagerly with a plain Python for-loop (for use outside of
    tracing mode), this function is meant to be called *while* the interpreter
    itself is being traced -- the resulting scan primitive is a real traced JAX
    operation.

    Parameters
    ----------
    scan_eq : jax.extend.core.JaxprEqn
        The equation representing the scan operation.
    context_dic : ContextDict
        Dictionary mapping variables to their values.
    eqn_evaluator : Callable, optional
        Function to evaluate the scanned body equation. The default is exec_eqn.

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
    num_xs = len(xs)

    scan_body = eval_jaxpr(scan_eq.params["jaxpr"], eqn_evaluator=eqn_evaluator)

    # NOTE: whether a carry/x is expanded into multiple positional args must be
    # decided by the Jaxpr arity (num_carry/num_xs), not by ``isinstance(...,
    # tuple)``. A *single* logical carry can itself be represented at runtime
    # as a plain Python tuple (e.g. the (pre_alloc_array, free_indices,
    # abs_qst) state used by the static-register interpreter, or catalyst's
    # (qreg, Jlist)); such a value must be passed to the body as one argument,
    # not unpacked into several.
    if num_consts > 0:

        def wrapped_body(carry: Any, x: Any) -> tuple[Any, tuple]:
            carry_args = [carry] if num_carry == 1 else list(carry)
            x_args = [x] if num_xs == 1 else list(x)
            args = consts + carry_args + x_args
            result = scan_body(*args)
            if not isinstance(result, tuple):
                result = (result,)
            new_carry = result[0] if num_carry == 1 else result[:num_carry]
            return new_carry, result[num_carry:]

    else:

        def wrapped_body(carry: Any, x: Any) -> tuple[Any, tuple]:
            carry_args = [carry] if num_carry == 1 else list(carry)
            x_args = [x] if num_xs == 1 else list(x)
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


# ---------------------------------------------------------------------------
# Signature flatten/unflatten helpers
#
# Shared by catalyst_interpreter.py and cl_func_interpreter.py, which both
# represent AbstractQuantumState as (state, Jlist) and AbstractQubitArray as a
# bare Jlist, but need a flat list of plain JAX values wherever they hand
# operands to a native jax.lax control-flow primitive (switch/while_loop) or
# jax.jit. The two representations only differ in what "state" is (a packed
# classical bit array for cl_func, a Catalyst quantum-register tracer for
# catalyst) -- something this pair of functions never inspects, so the exact
# same logic serves both.
# ---------------------------------------------------------------------------


def unflatten_signature(values: Sequence[Any], variables: Sequence[Var | Literal]) -> list[Any]:
    """Convert flattened JAX values back to structured quantum types.

    During JAX tracing, quantum types (AbstractQuantumState, AbstractQubitArray)
    are flattened into multiple array values. This function reconstructs the
    original structure.

    Parameters
    ----------
    values : Sequence[Any]
        Flattened sequence of JAX array values (cl_func_interpreter.py's callers
        pass a tuple, catalyst_interpreter.py's pass a list -- this function
        converts to a list itself either way, so it doesn't care which).
    variables : Sequence[Var | Literal]
        List of JAX variables describing the expected types.

    Returns
    -------
    list[Any]
        List of values with quantum types reconstructed:
        - AbstractQuantumState -> (state, Jlist)
        - AbstractQubitArray -> Jlist
        - Other types -> unchanged

    """
    # Rebound under a new name: `values` is declared as tuple (its incoming type),
    # so reassigning it to a list here would conflict with that declared type.
    remaining_values: list[Any] = list(values)
    unflattened_values: list[Any] = []

    for var in variables:
        if isinstance(var.aval, AbstractQuantumState):
            # Reconstruct (state, free_qubits_jlist) tuple
            state = remaining_values.pop(0)
            jlist_tuple = (remaining_values.pop(0), remaining_values.pop(0))
            unflattened_values.append((state, Jlist.unflatten([], jlist_tuple)))
        elif isinstance(var.aval, AbstractQubitArray):
            # Reconstruct Jlist from (array, counter) tuple
            jlist_tuple = (remaining_values.pop(0), remaining_values.pop(0))
            unflattened_values.append(Jlist.unflatten([], jlist_tuple))
        else:
            # Classical values pass through unchanged
            unflattened_values.append(remaining_values.pop(0))

    return unflattened_values


def flatten_signature(values: list[Any], variables: Sequence[Var | Literal]) -> list[Array]:
    """Flatten structured quantum types into plain JAX arrays.

    Quantum types need to be flattened for JAX operations like switch and
    while_loop that expect flat argument lists.

    Parameters
    ----------
    values : list[Any]
        List of values that may include quantum types.
    variables : Sequence[Var | Literal]
        List of JAX variables describing the types.

    Returns
    -------
    list[Array]
        Flattened list of JAX arrays:
        - AbstractQuantumState (state, Jlist) -> [state, array, counter]
        - AbstractQubitArray Jlist -> [array, counter]
        - Other types -> unchanged

    """
    remaining_values = list(values)
    flattened_values: list[Array] = []

    for var in variables:
        value = remaining_values.pop(0)
        if isinstance(var.aval, AbstractQuantumState):
            # Flatten (state, Jlist) -> [state, jlist.array, jlist.counter]
            flattened_values.extend((value[0], *value[1].flatten()[0]))
        elif isinstance(var.aval, AbstractQubitArray):
            # Flatten Jlist -> [array, counter]
            flattened_values.extend(value.flatten()[0])
        else:
            # Classical values pass through unchanged
            flattened_values.append(value)

    return flattened_values
