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

from jax.extend.core import ClosedJaxpr, Literal
from jax import make_jaxpr
import jax.numpy as jnp
from qrisp.jasp import check_for_tracing_mode


class ContextDict(dict):

    def __getitem__(self, key):
        if isinstance(key, Literal):
            res = key.val
        else:
            res = dict.__getitem__(self, key)

        if type(res) == int:
            return jnp.array(res, dtype=jnp.dtype("int64"))
        elif type(res) == float:
            return jnp.array(res, dtype=jnp.dtype("float64"))
        else:
            return res


def exec_eqn(eqn, context_dic):
    invalues = extract_invalues(eqn, context_dic)
    res = eqn.primitive.bind(*invalues, **eqn.params)
    insert_outvalues(eqn, context_dic, res)


def eval_jaxpr(jaxpr, return_context_dic=False, eqn_evaluator=exec_eqn):
    """
    Evaluates a Jaxpr using the given context dic to replace variables

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The jaxpr to evaluate.

    Returns
    -------
    None.

    """

    from qrisp.jasp import Jaspr

    if isinstance(jaxpr, ClosedJaxpr):
        consts = list(jaxpr.consts)
        jaxpr = jaxpr.jaxpr
    elif isinstance(jaxpr, Jaspr):
        consts = list(jaxpr.consts)
    else:
        consts = []

    def jaxpr_evaluator(*args):

        args = consts + list(args)
        temp_var_list = jaxpr.constvars + jaxpr.invars

        if len(temp_var_list) != len(args):
            raise Exception("Tried to evaluate jaxpr with insufficient arguments")

        context_dic = ContextDict({temp_var_list[i]: args[i] for i in range(len(args))})
        eval_jaxpr_with_context_dic(jaxpr, context_dic, eqn_evaluator)

        if return_context_dic:
            outvals = [context_dic]
        else:
            outvals = []

        for i in range(len(jaxpr.outvars)):
            outvals.append(context_dic[jaxpr.outvars[i]])

        if len(outvals) == 1:
            return outvals[0]
        else:
            return tuple(outvals)

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

        print(f"Evaluating eqn: {eqn}")

        default_eval = eqn_evaluator(eqn, context_dic)

        if default_eval:
            if (
                eqn.primitive.name in ["while", "cond", "scan"]
                and not check_for_tracing_mode()
            ):

                from qrisp.jasp import (
                    evaluate_cond_eqn,
                    evaluate_while_loop,
                    evaluate_scan,
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
