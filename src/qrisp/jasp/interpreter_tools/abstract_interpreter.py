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

from typing import Callable

import jax.numpy as jnp
from jax import make_jaxpr
from jax.extend.core import ClosedJaxpr, Literal

from qrisp.jasp import check_for_tracing_mode


class ContextDict(dict):
    """
    A dictionary-like data structure to hold variable bindings during Jaxpr evaluation.
    """

    def __getitem__(self, key):
        if isinstance(key, Literal):
            res = key.val
        else:
            res = dict.__getitem__(self, key)

        if type(res) == int:
            return jnp.asarray(res, dtype=jnp.dtype("int64"))
        if type(res) == float:
            return jnp.asarray(res, dtype=jnp.dtype("float64"))
        return res


def exec_eqn(eqn, context_dic):
    invalues = extract_invalues(eqn, context_dic)
    res = eqn.primitive.bind(*invalues, **eqn.params)
    insert_outvalues(eqn, context_dic, res)


def eval_jaxpr(jaxpr, return_context_dic=False, eqn_evaluator=exec_eqn) -> Callable:
    """
    Evaluates a Jaxpr using the provided equation evaluator.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr | ClosedJaxpr | Jaspr
        The JAX expression to evaluate.

    return_context_dic : bool, optional
        Whether to return the context dictionary along with the output values,
        by default False.

    eqn_evaluator : Callable, optional
        The function to evaluate each equation in the JAX expression,
        by default exec_eqn.

    Returns
    -------
    Callable
        A function that evaluates the jaxpr.

    """

    # Import here to avoid circular imports
    from qrisp.jasp.jasp_expression import Jaspr

    consts = []
    if isinstance(jaxpr, ClosedJaxpr):
        consts = list(jaxpr.consts)
        jaxpr_core = jaxpr.jaxpr
    elif isinstance(jaxpr, Jaspr):
        consts = list(jaxpr.consts)
        jaxpr_core = jaxpr
    else:
        jaxpr_core = jaxpr

    invars = list(jaxpr_core.constvars) + list(jaxpr_core.invars)
    outvars = list(jaxpr_core.outvars)

    def jaxpr_evaluator(*args):

        args_and_consts = consts + list(args)

        if len(args_and_consts) != len(invars):
            raise ValueError(
                "Tried to evaluate jaxpr with insufficient arguments: "
                f"expected {len(invars)} (including {len(consts)} consts), got {len(args_and_consts)}."
            )

        context_dic = ContextDict(dict(zip(invars, args_and_consts)))

        eval_jaxpr_with_context_dic(jaxpr_core, context_dic, eqn_evaluator)

        if return_context_dic:
            outputs = [context_dic] + [context_dic[v] for v in outvars]
        else:
            outputs = [context_dic[v] for v in outvars]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return jaxpr_evaluator


def reinterpret(jaxpr, eqn_evaluator=exec_eqn):

    if isinstance(jaxpr, ClosedJaxpr):
        inter_jaxpr = jaxpr.jaxpr
    else:
        inter_jaxpr = jaxpr

    res = make_jaxpr(eval_jaxpr(inter_jaxpr, eqn_evaluator=eqn_evaluator))(
        *[var.aval for var in inter_jaxpr.constvars + inter_jaxpr.invars]
    ).jaxpr

    res.constvars.extend(res.invars[: len(inter_jaxpr.constvars)])
    temp = list(res.invars[len(inter_jaxpr.constvars) :])
    res.invars.clear()
    res.invars.extend(temp)

    if isinstance(jaxpr, ClosedJaxpr):
        res = ClosedJaxpr(res, jaxpr.consts)

    return res


def eval_jaxpr_with_context_dic(jaxpr, context_dic, eqn_evaluator=exec_eqn):

    # Iterate through the equations
    for eqn in jaxpr.eqns:
        # Evaluate the primitive

        default_eval = eqn_evaluator(eqn, context_dic)

        if default_eval:
            if (
                eqn.primitive.name in ["while", "cond", "scan"]
                and not check_for_tracing_mode()
            ):

                from qrisp.jasp import (
                    evaluate_cond_eqn,
                    evaluate_scan,
                    evaluate_while_loop,
                )

                if eqn.primitive.name == "while":
                    evaluate_while_loop(eqn, context_dic, eqn_evaluator)
                elif eqn.primitive.name == "cond":
                    evaluate_cond_eqn(eqn, context_dic, eqn_evaluator)
                else:
                    evaluate_scan(eqn, context_dic, eqn_evaluator)

                continue

            exec_eqn(eqn, context_dic)


def extract_invalues(eqn, context_dic):
    invalues = []
    for i in range(len(eqn.invars)):
        invar = eqn.invars[i]
        invalues.append(context_dic[invar])
    return invalues


def extract_constvalues(eqn, context_dic):
    constvalues = []
    for i in range(len(eqn.constvars)):
        constvar = eqn.constvars[i]
        constvalues.append(context_dic[constvar])

    return constvalues


def insert_outvalues(eqn, context_dic, outvalues):

    if eqn.primitive.multiple_results:
        if len(outvalues) != len(eqn.outvars):
            raise Exception(
                "Tried to insert invalid amount of values into the Context Dictionary"
            )

        for i in range(len(eqn.outvars)):
            context_dic[eqn.outvars[i]] = outvalues[i]
    else:
        context_dic[eqn.outvars[0]] = outvalues
