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

from functools import lru_cache

import numpy as np
from sympy import lambdify

from jax import make_jaxpr, jit
from jax.extend.core import JaxprEqn, Var, ClosedJaxpr, Jaxpr
from jax.lax import add_p, sub_p, while_loop

from qrisp.jasp.primitives import AbstractQuantumState, quantum_gate_p, greek_letters
from qrisp.jasp.interpreter_tools import eval_jaxpr, extract_invalues, insert_outvalues


def copy_jaxpr_eqn(eqn):
    return JaxprEqn(
        primitive=eqn.primitive,
        invars=list(eqn.invars),
        outvars=list(eqn.outvars),
        params=dict(eqn.params),
        source_info=eqn.source_info,
        effects=eqn.effects,
        ctx=eqn.ctx,
    )


qc_var_count = np.zeros(1, dtype=np.int64)


def invert_eqn(eqn):
    """
    Receives and equation that describes either an operation or a pjit primitive
    and returns an equation that describes the inverse.

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        The equation to be inverted.

    Returns
    -------
    inverted_eqn
        The equation with inverted operation.

    """

    if eqn.primitive.name == "jit":
        params = dict(eqn.params)
        params["jaxpr"] = eqn.params["jaxpr"].inverse()

        name = params["name"]
        if name[-3:] == "_dg":
            params["name"] = params["name"][:-3]
        elif name not in ["gidney_mcx", "gidney_mcx_inv"]:
            params["name"] += "_dg"
        elif name == "gidney_mcx":
            params["name"] = "gidney_mcx_inv"
        elif name == "gidney_mcx_inv":
            params["name"] = "gidney_mcx"

        primitive = eqn.primitive
    elif eqn.primitive.name == "while":
        return invert_loop_eqn(eqn)
    elif eqn.primitive.name == "cond":
        params = dict(eqn.params)

        branches = params["branches"]

        new_branch_list = []

        for i in range(len(branches)):
            new_branch_list.append(branches[i].inverse())

        params["branches"] = tuple(new_branch_list)

        primitive = eqn.primitive

    else:
        params = dict(eqn.params)
        params["gate"] = params["gate"].inverse()
        primitive = eqn.primitive

    return JaxprEqn(
        primitive=primitive,
        invars=list(eqn.invars),
        outvars=list(eqn.outvars),
        params=params,
        source_info=eqn.source_info,
        effects=eqn.effects,
        ctx=eqn.ctx,
    )


@lru_cache(int(1e5))
def invert_jaspr(jaspr):
    """
    Takes a Jaspr and returns a Jaspr, which performs the inverted quantum operation

    Parameters
    ----------
    jaspr : qrisp.jasp.Jaspr
        The Jaspr to be inverted.

    Returns
    -------
    inverted_jaspr : qrisp.jasp.Jaspr
        The inverted/daggered Jaspr.

    """

    # Flatten all environments in the jaspr
    jaspr = jaspr.flatten_environments()
    # We separate the equations into classes where one executes Operations and
    # the one that doesn't execute Operations
    if jaspr.inv_jaspr is not None:
        return jaspr.inv_jaspr

    op_eqs = []
    non_op_eqs = []
    deletions = []
    current_abs_qst = jaspr.invars[-1]

    # Since the Operation equations require as inputs only qubit object and a QuantumState
    # we achieve our goal by pulling all the non-Operation equations to the front
    # and the Operation equations to the back.

    for eqn in jaspr.eqns:
        if eqn.primitive == quantum_gate_p or (
            (eqn.primitive.name in ["jit", "while", "cond"])
            and len(eqn.invars)
            and isinstance(eqn.invars[-1].aval, AbstractQuantumState)
        ):
            # Insert the inverted equation at the front
            op_eqs.insert(0, invert_eqn(eqn))
        elif eqn.primitive.name == "jasp.measure":
            raise Exception("Tried to invert a jaspr containing a measurement")
        elif eqn.primitive.name == "jasp.delete_qubits":
            eqn = copy_jaxpr_eqn(eqn)
            deletions.append(eqn)
        elif len(eqn.invars) and isinstance(eqn.invars[-1].aval, AbstractQuantumState):
            eqn = copy_jaxpr_eqn(eqn)
            eqn.invars[-1] = current_abs_qst
            current_abs_qst = eqn.outvars[-1]
            non_op_eqs.append(copy_jaxpr_eqn(eqn))
        else:
            non_op_eqs.append(copy_jaxpr_eqn(eqn))

    # Finally, we need to make sure the Order of QuantumState I/O is also reversed.
    n = len(op_eqs)
    if n == 0:
        return jaspr

    op_eqs = op_eqs + deletions

    for i in range(len(op_eqs)):
        op_eqs[i].invars[-1] = current_abs_qst
        current_abs_qst = Var(aval=AbstractQuantumState())
        qc_var_count[0] += 1
        op_eqs[i].outvars[-1] = current_abs_qst

    def eqn_evaluator(eqn, context_dic):

        if eqn.primitive == quantum_gate_p:
            invals = extract_invalues(eqn, context_dic)
            num_qubits = eqn.params["gate"].num_qubits

            params = eqn.params["gate"].params
            tracers = invals[num_qubits:-1]
            symbols = greek_letters[: len(params)]

            processed_tracers = []

            for expr in params:
                processed_tracers.append(
                    lambdify(symbols, expr, modules="jax")(*tracers)
                )

            new_gate = eqn.params["gate"].bind_parameters(
                {symbols[i]: params[i] for i in range(len(params))}
            )

            outvalues = quantum_gate_p.bind(
                *(invals[:num_qubits] + processed_tracers + [invals[-1]]), gate=new_gate
            )

            insert_outvalues(eqn, context_dic, outvalues)

        else:
            return True

    temp_jaxpr = ClosedJaxpr(
        Jaxpr(
            invars=list(jaspr.invars),
            outvars=jaspr.outvars[:-1] + [current_abs_qst],
            constvars=jaspr.constvars,
            eqns=non_op_eqs + op_eqs,
            debug_info=jaspr.debug_info,
        ),
        jaspr.consts,
    )

    processed_jaxpr = make_jaxpr(eval_jaxpr(temp_jaxpr, eqn_evaluator=eqn_evaluator))(
        *[invar.aval for invar in jaspr.invars]
    )

    from qrisp.jasp import Jaspr

    res = Jaspr(processed_jaxpr)

    # res = Jaspr(
    #     constvars=jaspr.constvars,
    #     invars=list(jaspr.invars),
    #     outvars=jaspr.outvars[:-1] + [current_abs_qst],
    #     eqns=non_op_eqs + op_eqs,
    # )

    if jaspr.ctrl_jaspr:
        res.ctrl_jaspr = jaspr.ctrl_jaspr.inverse()

    return res


# The following two functions manage the logic behind inverting a loop.
# For this we have to consider two aspects.

# 1. The loop body contains the loop index incrementation. This incrementation
#    needs to be reversed to a decrementation. Furthermore the quantum part of
#    the loop body needs to be inverted.

# 2. The loop index initialization and the threshold need to be switched and
#    the comparison to determine the break condition needs to be adjusted.


def invert_loop_body(jaxpr):
    # This function treats the loop body

    # This list will contain the equations with the loop index decrementation
    new_eqn_list = list(jaxpr.eqns)

    # Find the incrementation equation via the named marker jit equation
    # inserted by jrange right before __exit__.  Only jrange-based loops
    # carry this marker, so loops created by other means cannot be
    # automatically inverted.
    from qrisp.jasp.program_control.jrange_iterator import JRANGE_MARKER_NAME

    marker_eqn = None
    for eqn in new_eqn_list:
        if eqn.primitive.name == "jit":
            if eqn.params.get("name") == JRANGE_MARKER_NAME:
                marker_eqn = eqn
                break

    if marker_eqn is None:
        raise Exception(
            "Automatic loop inversion is only supported for jrange-based "
            "loops. The current loop body does not contain a jrange marker "
            f"('{JRANGE_MARKER_NAME}')."
        )

    # marker(updated_loop_index, threshold)
    # Find the add/sub whose outvar is the marker's invars[0].
    updated_var = marker_eqn.invars[0]
    increment_eqn_index = None
    for i in range(len(new_eqn_list))[::-1]:
        if new_eqn_list[i].outvars[0] is updated_var:
            if new_eqn_list[i].primitive in (add_p, sub_p):
                increment_eqn_index = i
                break

    if increment_eqn_index is None:
        raise Exception(
            "Could not find the increment equation feeding the jrange "
            "marker. The loop body may be malformed."
        )

    increment_eqn = new_eqn_list[increment_eqn_index]

    # Change the primitive
    if increment_eqn.primitive is add_p:
        new_primitive = sub_p
    else:
        new_primitive = add_p

    # Create the decrement equation
    decrement_eqn = JaxprEqn(
        primitive=new_primitive,
        invars=list(increment_eqn.invars),
        outvars=list(increment_eqn.outvars),
        params=increment_eqn.params,
        source_info=increment_eqn.source_info,
        effects=increment_eqn.effects,
        ctx=increment_eqn.ctx,
    )
    new_eqn_list[increment_eqn_index] = decrement_eqn

    # Create the new Jaspr
    from qrisp.jasp import Jaspr

    new_jaspr = Jaspr(jaxpr).update_eqns(new_eqn_list)

    # Quantum inversion
    res_jaspr = new_jaspr.inverse()

    return res_jaspr


# This function performs the above mentioned step 2 to treat the loop primitive.
def invert_loop_eqn(eqn):

    # Process the loop body
    body_jaxpr = eqn.params["body_jaxpr"]
    inv_loop_body = invert_loop_body(body_jaxpr)

    # Assert that inversion preserved the body's input/output structure
    orig_in_avals = [v.aval for v in body_jaxpr.jaxpr.invars]
    inv_in_avals = [v.aval for v in inv_loop_body.jaxpr.invars]
    orig_out_avals = [v.aval for v in body_jaxpr.jaxpr.outvars]
    inv_out_avals = [v.aval for v in inv_loop_body.jaxpr.outvars]
    assert len(orig_in_avals) == len(inv_in_avals), \
        f"Body inversion changed invar count: {len(orig_in_avals)} → {len(inv_in_avals)}"
    assert len(orig_out_avals) == len(inv_out_avals), \
        f"Body inversion changed outvar count: {len(orig_out_avals)} → {len(inv_out_avals)}"
    for i, (a, b) in enumerate(zip(orig_in_avals, inv_in_avals)):
        assert type(a) is type(b), \
            f"Body inversion changed invar[{i}] type: {type(a).__name__} → {type(b).__name__}"
    for i, (a, b) in enumerate(zip(orig_out_avals, inv_out_avals)):
        assert type(a) is type(b), \
            f"Body inversion changed outvar[{i}] type: {type(a).__name__} → {type(b).__name__}"

    # Extract the compared carry positions from the original cond.
    # Access everything via .jaxpr to ensure consistent Var objects.
    cond_jaxpr = eqn.params["cond_jaxpr"]
    cond_core = cond_jaxpr.jaxpr
    for eq in cond_core.eqns:
        if id(eq.outvars[0]) == id(cond_jaxpr.jaxpr.outvars[0]):
            cond_eq = eq
            break

    cond_ins = cond_core.invars

    # Helper: backtrace a Var through convert_element_type chains
    # to find the original invar it ultimately derives from.
    def backtrace(var):
        for _ in range(10):  # safety limit
            found = False
            for eq in cond_core.eqns:
                if eq.primitive.name == "convert_element_type" and eq.outvars[0] is var:
                    var = eq.invars[0]
                    found = True
                    break
            if not found:
                break
        return var

    # Find positions, backtracing through any type conversions
    pos_a = next(i for i, v in enumerate(cond_ins) if v is backtrace(cond_eq.invars[0]))
    pos_b = next(i for i, v in enumerate(cond_ins) if v is backtrace(cond_eq.invars[1]))

    # Trace a cond with swapped operands. Swapping the operands of
    # le/ge inverts the comparison. Combined with the invar swap
    # below, this gives the correct inverted loop condition.
    carries_count = len(eqn.outvars)
    carry_avals = [v.aval for v in eqn.invars[-carries_count:]]

    if cond_eq.primitive.name == "ge":
        def swapped_cond(*carries):
            return carries[pos_b] >= carries[pos_a]
    else:
        def swapped_cond(*carries):
            return carries[pos_b] <= carries[pos_a]

    inv_cond_jaxpr = make_jaxpr(swapped_cond)(*carry_avals)

    # Swap the equation invars at positions corresponding to the
    # compared carries (init ↔ threshold).
    invars = eqn.invars
    new_invars = list(invars)
    eqn_pos_a = eqn.params["cond_nconsts"] + eqn.params["body_nconsts"] + pos_a
    eqn_pos_b = eqn.params["cond_nconsts"] + eqn.params["body_nconsts"] + pos_b
    new_invars[eqn_pos_a], new_invars[eqn_pos_b] = \
        new_invars[eqn_pos_b], new_invars[eqn_pos_a]

    # Assemble the inverted while equation directly — no while_loop
    # re-trace, avoiding JAX's non-deterministic body_nconsts.
    res = JaxprEqn(
        primitive=eqn.primitive,
        invars=new_invars,
        outvars=list(eqn.outvars),
        params={
            "body_jaxpr": inv_loop_body,
            "cond_jaxpr": inv_cond_jaxpr,
            "body_nconsts": eqn.params["body_nconsts"],
            "cond_nconsts": eqn.params["cond_nconsts"],
        },
        source_info=eqn.source_info,
        effects=eqn.effects,
        ctx=eqn.ctx,
    )

    return res
