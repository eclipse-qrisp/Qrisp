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

from jax import tree_util
from qrisp.core import QuantumVariable, QuantumSession
from builtins import id


def flatten_qv(qv):
    # return the tracers and auxiliary data (structure of the object)
    children = (qv.reg, qv.size)
    aux_data = (id(qv), qv.name)  # No auxiliary data in this simple example
    return children, aux_data

def unflatten_qv(aux_data, children):
    # reconstruct the object from children and auxiliary data
    
    res = QuantumVariable.__new__(QuantumVariable)
    
    res.reg = children[0]
    res.size = children[1]
    res.name = aux_data[1]
    res.qs = QuantumSession()
    
    return res

# Register as a PyTree with JAX
tree_util.register_pytree_node(QuantumVariable, flatten_qv, unflatten_qv)