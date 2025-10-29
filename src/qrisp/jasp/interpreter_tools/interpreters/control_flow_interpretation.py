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
    exec_eqn,
    extract_invalues,
    insert_outvalues,
)


def evaluate_cond_eqn(cond_eqn, context_dic, eqn_evaluator=exec_eqn):

    # Extract the invalues from the context dic
    invalues = extract_invalues(cond_eqn, context_dic)

    from qrisp.jasp.jasp_expression import ProcessedMeasurement

    if isinstance(invalues[0], ProcessedMeasurement):
        raise Exception("Tried to convert real-time feedback into QuantumCircuit")

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

    from qrisp.jasp.jasp_expression import ProcessedMeasurement

    num_const_cond_args = while_loop_eqn.params["cond_nconsts"]
    num_const_body_args = while_loop_eqn.params["body_nconsts"]
    overall_constant_amount = max(num_const_cond_args, num_const_body_args)

    def break_condition(invalues):

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

        constants = invalues[:num_const_body_args]
        carries = invalues[overall_constant_amount:]

        new_invalues = constants + carries

        outvalues = eval_jaxpr(
            while_loop_eqn.params["body_jaxpr"], eqn_evaluator=eqn_evaluator
        )(*new_invalues)

        # Update the non-const invalues

        if len(while_loop_eqn.params["body_jaxpr"].jaxpr.outvars) == 1:
            outvalues = (outvalues,)

        invalues = invalues[:overall_constant_amount] + list(outvalues)

        if break_after_first_iter:
            break

    insert_outvalues(while_loop_eqn, context_dic, outvalues)


def evaluate_scan(scan_eq, context_dic, eqn_evaluator=exec_eqn):

    invalues = extract_invalues(scan_eq, context_dic)

    f = eval_jaxpr(scan_eq.params["jaxpr"], eqn_evaluator=eqn_evaluator)

    length = scan_eq.params["length"]

    carry_amount = scan_eq.params["num_carry"]
    const_amount = scan_eq.params["num_consts"]

    init = invalues[const_amount : carry_amount + const_amount]

    if len(invalues) == carry_amount + const_amount:
        xs = [[]] * length
    else:
        xs = [[invalues[-1][i]] for i in range(length)]

    carry = init
    consts = invalues[:const_amount]
    ys = []
    for x in xs:
        args = consts + list(carry) + x

        res = f(*args)

        if not isinstance(res, tuple):
            res = (res,)

        carry = res[:carry_amount]
        y = res[carry_amount:]

        if len(y):
            ys.append(y[0])

    if len(ys):
        ys = [jnp.stack(ys)]

    outvalues = list(carry) + ys

    insert_outvalues(scan_eq, context_dic, outvalues)
