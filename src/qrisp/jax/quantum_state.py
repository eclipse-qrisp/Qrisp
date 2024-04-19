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

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings
from qrisp.jax import QuantumPrimitive

class AbstractQuantumState(AbstractValue):
    
    def __init__(self):
        AbstractValue.__init__(self)
        self.burned = False

def create_register(size, state):
    return create_register_p.bind(size, state)
        
raise_to_shaped_mappings[AbstractQuantumState] = lambda aval, _: aval

create_quantum_state_p = QuantumPrimitive("create_quantum_state")
def create_quantum_state_abstract_eval():
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    
    return AbstractQuantumState()

create_quantum_state_p.def_abstract_eval(create_quantum_state_abstract_eval)


# Register Creation

create_register_p = QuantumPrimitive("create_reg")
create_register_p.multiple_results = True

from qrisp.jax import AbstractQuantumRegister

def create_register_abstract_eval(size, state):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    state.burned = True
    
    return AbstractQuantumState(), AbstractQuantumRegister()

create_register_p.def_abstract_eval(create_register_abstract_eval)

# State entangling
entangle_p = QuantumPrimitive("entangle")

def entangle_abstract_eval(*states):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    for state in states:
        state.burned = True
    return AbstractQuantumState()

entangle_p.def_abstract_eval(entangle_abstract_eval)
