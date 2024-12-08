"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

from jax import make_jaxpr
from jax.core import JaxprEqn, ClosedJaxpr
from jax.lax import add_p, sub_p, while_loop

from qrisp.jasp.primitives import AbstractQuantumCircuit, OperationPrimitive

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
        params["jaxpr"] = ClosedJaxpr(invert_jaspr(eqn.params["jaxpr"].jaxpr),
                                      eqn.params["jaxpr"].consts)
        primitive = eqn.primitive
    elif eqn.primitive.name == "while":
        return invert_loop_eqn(eqn)
    else:
        params = dict(eqn.params)
        primitive = eqn.primitive.inverse()
        
    return JaxprEqn(primitive = primitive,
                    invars = list(eqn.invars),
                    outvars = list(eqn.outvars),
                    params = params,
                    source_info = eqn.source_info,
                    effects = eqn.effects)
    
        


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
    op_eqs = []
    non_op_eqs = []
    deletions = []
    
    # Since the Operation equations require as inputs only qubit object and a QuantumCircuit
    # we achieve our goal by pulling all the non-Operation equations to the front
    # and the Operation equations to the back.
    
    for eqn in jaspr.eqns:
        if isinstance(eqn.primitive, OperationPrimitive) or ((eqn.primitive.name == "pjit" or eqn.primitive.name == "while") and isinstance(eqn.outvars[0].aval, AbstractQuantumCircuit)):
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
    
    for i in range(n//2):
        
        op_eqs[i].invars[0], op_eqs[n-i-1].invars[0] = op_eqs[n-i-1].invars[0], op_eqs[i].invars[0]
        op_eqs[i].outvars[0], op_eqs[n-i-1].outvars[0] = op_eqs[n-i-1].outvars[0], op_eqs[i].outvars[0]
    
    from qrisp.jasp import Jaspr
    
    return Jaspr(constvars = jaspr.constvars, 
                 invars = jaspr.invars, 
                 outvars = op_eqs[-1].outvars[:1] + jaspr.outvars[1:], 
                 eqns = non_op_eqs + op_eqs + deletions)
        


def invert_loop_body(jaspr):
    from qrisp.jasp import Jaspr
    
    new_eqn_list = list(jaspr.eqns)
    
    increment_eqn = new_eqn_list[-1]
    
    if increment_eqn.primitive is add_p:
        new_primitive = sub_p
    else:
        new_primitive = add_p
    
    if increment_eqn.invars[1].val != 1:
        raise Exception
    
    decrement_eqn = JaxprEqn(primitive = new_primitive,
                            invars = list(increment_eqn.invars),
                            outvars = list(increment_eqn.outvars),
                            params = increment_eqn.params,
                            source_info = increment_eqn.source_info,
                            effects = increment_eqn.effects,)
    
    new_eqn_list[-1] = decrement_eqn
    
    return Jaspr(jaspr.jaxpr).update_eqns(new_eqn_list).inverse()


def invert_loop_eqn(eqn):
    
    body_jaxpr = eqn.params["body_jaxpr"]
    inv_loop_body = invert_loop_body(body_jaxpr)
    
    def body_fun(val):
        return inv_loop_body.eval(*val)
    
    # The condition function should compare whether the loop index (second last position)
    # is smaller than the loop cancelation threshold (last position)
    def cond_fun(val):
        return val[-2] >= val[-1]

    def tracing_function(*args):
        return while_loop(cond_fun, body_fun, tuple(args))

    jaxpr = make_jaxpr(tracing_function)(*[var.aval for var in eqn.invars])
    
    new_eqn = jaxpr.eqns[0]
    
    invars = eqn.invars
    new_invars = list(invars)
    new_invars[-1], new_invars[-2] = new_invars[-2], new_invars[-1]
    
    res = JaxprEqn(primitive = new_eqn.primitive,
                   invars = new_invars,
                   outvars = eqn.outvars,
                   params = new_eqn.params,
                   source_info = eqn.source_info,
                   effects = eqn.effects)
    
    return res