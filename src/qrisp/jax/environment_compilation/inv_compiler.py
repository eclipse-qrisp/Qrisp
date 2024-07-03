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

from jax import jit
from jax.core import Jaxpr, JaxprEqn
from qrisp.jax import get_tracing_qs, check_for_tracing_mode

def invert_eqn(eqn):
    """
    Returns an equation that describes the inverse operation.

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        The equation to be inverted.

    Returns
    -------
    inverted_eqn
        The equation with inverted operation.

    """
    
    return JaxprEqn(primitive = eqn.primitive.inverse(),
                    invars = eqn.invars,
                    outvars = eqn.outvars,
                    params = eqn.params,
                    source_info = eqn.source_info,
                    effects = eqn.effects)


def inv_transform(jaxpr):
    """
    Takes a jaxpr returning a quantum circuit and returns a jaxpr, which returns
    the inverse quantum circuit

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        A jaxpr returning a QuantumCircuit.

    Returns
    -------
    inverted_jaxpr : jaxpr.core.Jaxpr
        A jaxpr returning the inverse QuantumCircuit.

    """
    
    # We separate the equations into classes where one executes Operations and
    # the one that doesn't execute Operations
    
    op_eqs = []
    non_op_eqs = []
    
    # Since the Operation equations require as inputs only qubit object and a QuantumCircuit
    # we achieve our goal by pulling all the non-Operation equations to the front
    # and the Operation equations to the back.
    
    from qrisp.circuit import Operation
    for eqn in jaxpr.eqns:
        if isinstance(eqn.primitive, Operation):
            # Insert the inverted equation at the front
            op_eqs.insert(0, invert_eqn(eqn))
        else:
            non_op_eqs.append(eqn)

    # Finally, we need to make sure the Order of QuantumCircuit I/O is also reversed.        
    n = len(op_eqs)
    for i in range(n//2):
        
        op_eqs[i].invars[0], op_eqs[n-i-1].invars[0] = op_eqs[n-i-1].invars[0], op_eqs[i].invars[0]
        op_eqs[i].outvars[0], op_eqs[n-i-1].outvars[0] = op_eqs[n-i-1].outvars[0], op_eqs[i].outvars[0]
    
    
    return Jaxpr(constvars = jaxpr.constvars, 
                 invars = jaxpr.invars, 
                 outvars = jaxpr.outvars, 
                 eqns = non_op_eqs + op_eqs)
        
