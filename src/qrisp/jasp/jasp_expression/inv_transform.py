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

from jax.core import JaxprEqn, ClosedJaxpr

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
    
    # Since the Operation equations require as inputs only qubit object and a QuantumCircuit
    # we achieve our goal by pulling all the non-Operation equations to the front
    # and the Operation equations to the back.
    
    for eqn in jaspr.eqns:
        if isinstance(eqn.primitive, OperationPrimitive) or (eqn.primitive.name == "pjit" and isinstance(eqn.outvars[0].aval, AbstractQuantumCircuit)):
            # Insert the inverted equation at the front
            op_eqs.insert(0, invert_eqn(eqn))
        elif eqn.primitive.name == "jasp.measure":
            raise Exception("Tried to invert a jaspr containing a measurement")
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
                 eqns = non_op_eqs + op_eqs)
        
