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

from functools import lru_cache

from jax import make_jaxpr
from jax.extend.core import JaxprEqn, ClosedJaxpr
from jax.lax import add_p, sub_p, while_loop

from qrisp.jasp.primitives import AbstractQuantumCircuit, OperationPrimitive
from qrisp.jasp.interpreter_tools import eval_jaxpr

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

    if eqn.primitive.name == "pjit":
        params = dict(eqn.params)
        params["jaxpr"] = ClosedJaxpr(
            (eqn.params["jaxpr"].jaxpr).inverse(), eqn.params["jaxpr"].consts
        )

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
            new_branch_list.append(
                ClosedJaxpr((branches[i].jaxpr).inverse(), branches[i].consts)
            )

        params["branches"] = tuple(new_branch_list)

        primitive = eqn.primitive

    else:
        params = dict(eqn.params)
        primitive = eqn.primitive.inverse()

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

    # Since the Operation equations require as inputs only qubit object and a QuantumCircuit
    # we achieve our goal by pulling all the non-Operation equations to the front
    # and the Operation equations to the back.

    for eqn in jaspr.eqns:
        if isinstance(eqn.primitive, OperationPrimitive) or (
            (eqn.primitive.name in ["pjit", "while", "cond"])
            and len(eqn.invars)
            and isinstance(eqn.invars[-1].aval, AbstractQuantumCircuit)
        ):
            # Insert the inverted equation at the front
            op_eqs.insert(0, invert_eqn(eqn))
        elif eqn.primitive.name == "jasp.measure":
            raise Exception("Tried to invert a jaspr containing a measurement")
        elif eqn.primitive.name == "jasp.delete_qubits":
            deletions.append(eqn)
        else:
            non_op_eqs.append(eqn)

    # Finally, we need to make sure the Order of QuantumCircuit I/O is also reversed.
    n = len(op_eqs)
    if n == 0:
        return jaspr

    for i in range(n // 2):

        for j in range(len(op_eqs[i].invars)):
            if isinstance(op_eqs[i].invars[j].aval, AbstractQuantumCircuit):
                break
        else:
            raise

        for k in range(len(op_eqs[n - i - 1].invars)):
            if isinstance(op_eqs[n - i - 1].invars[k].aval, AbstractQuantumCircuit):
                break
        else:
            raise

        op_eqs[i].invars[j], op_eqs[n - i - 1].invars[k] = (
            op_eqs[n - i - 1].invars[k],
            op_eqs[i].invars[j],
        )

        for j in range(len(op_eqs[i].outvars)):
            if isinstance(op_eqs[i].outvars[j].aval, AbstractQuantumCircuit):
                break
        else:
            raise

        for k in range(len(op_eqs[n - i - 1].outvars)):
            if isinstance(op_eqs[n - i - 1].outvars[k].aval, AbstractQuantumCircuit):
                break
        else:
            raise

        op_eqs[i].outvars[j], op_eqs[n - i - 1].outvars[k] = (
            op_eqs[n - i - 1].outvars[k],
            op_eqs[i].outvars[j],
        )

    from qrisp.jasp import Jaspr

    # Find the last AbstractQuantumCircuit variable

    for j in range(len(op_eqs[-1].outvars)):
        if isinstance(op_eqs[-1].outvars[j].aval, AbstractQuantumCircuit):
            last_abs_qc = op_eqs[-1].outvars[j]
            break

    if len(deletions):
        first_deletion = copy_jaxpr_eqn(deletions[0])
        first_deletion.invars[-1] = last_abs_qc
        last_abs_qc = deletions[-1].outvars[-1]

    res = Jaspr(
        constvars=jaspr.constvars,
        invars=list(jaspr.invars),
        outvars=jaspr.outvars[:-1] + [last_abs_qc],
        eqns=non_op_eqs + op_eqs + deletions,
    )

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

    # Find the incrementation equation. For this we identify the equation,
    # which updates the loop index

    loop_index = jaxpr.jaxpr.outvars[-3]
    for i in range(len(new_eqn_list))[::-1]:
        if loop_index == new_eqn_list[i].outvars[0]:
            break
    increment_eqn_index = int(i)
    increment_eqn = new_eqn_list[increment_eqn_index]

    # Change the primitive
    if increment_eqn.primitive is add_p:
        new_primitive = sub_p
    else:
        new_primitive = add_p

    try:
        if increment_eqn.invars[1].val != 1:
            raise
    except:
        raise Exception(
            "Dynamic loop inversion is only supported for loops the have step size 1."
        )

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

    new_jaspr = Jaspr(jaxpr.jaxpr).update_eqns(new_eqn_list)

    # Quantum inversion
    res_jaspr = new_jaspr.inverse()

    return res_jaspr


# This function performs the above mentioned step 2 to treat the loop primitive
def invert_loop_eqn(eqn):

    overall_constant_amount= max(eqn.params["body_nconsts"], eqn.params["cond_nconsts"])
    
    # Process the loop body
    body_jaxpr = eqn.params["body_jaxpr"]
    inv_loop_body = invert_loop_body(body_jaxpr)

    def body_fun(val):
        
        constants = val[:eqn.params["body_nconsts"]]
        carries = val[overall_constant_amount:]
        
        body_res = eval_jaxpr(inv_loop_body)(*(constants + carries))
        
        return val[:overall_constant_amount] + tuple(body_res)
        
    # Process the loop cancelation
    cond_jaxpr = eqn.params["cond_jaxpr"]

    def cond_fun(val):
        
        if cond_jaxpr.eqns[0].primitive.name == "ge":
            return val[-3] <= val[-2]
        else:
            return val[-3] >= val[-2]

    # Create the new equation by tracing the while loop
    def tracing_function(*args):
        return while_loop(cond_fun, body_fun, tuple(args))

    jaxpr = make_jaxpr(tracing_function)(*[var.aval for var in eqn.invars])
    new_eqn = jaxpr.eqns[0]

    # The new invars should have initial loop index at loop threshold switched.
    # The loop initialization is located at invars[-3] and the threshold at invars[-2]
    invars = eqn.invars
    new_invars = list(invars)
    new_invars[-2], new_invars[-3] = new_invars[-3], new_invars[-2]

    # Create the Equation
    res = JaxprEqn(
        primitive=new_eqn.primitive,
        invars=new_invars,
        outvars=list(eqn.outvars),
        params=new_eqn.params,
        source_info=eqn.source_info,
        effects=eqn.effects,
        ctx=new_eqn.ctx
    )

    return res
