"""
\********************************************************************************
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
********************************************************************************/
"""

import copy

import jax
import jax.numpy as jnp

from qrisp.jasp.interpreter_tools.dynamic_list import Jlist

@jax.tree_util.register_pytree_node_class
class DynamicQuantumMemory:
    
    def __init__(self, max_size):
        self.free_qubit_list = Jlist(jnp.arange(max_size, 0, -1), max_size = max_size)
        self.max_size = max_size
        
    def request_qubits(self, amount):
        self.free_qubit_list, res = request_qubit_helper(self.free_qubit_list, amount)
        
    def receive_qubits(self, qubit_list):
        qubit_list = qubit_list.copy()
        self.free_qubit_list = qubit_list.extend(self.free_qubit_list)

    def flatten(self):
        """
        Flatten the DynamicJaxArray into a tuple of arrays and auxiliary data.
        This is useful for JAX transformations and serialization.
        """
        return (self.free_qubit_list,), (self.max_size, self.mm_style)

    @classmethod
    def unflatten(cls, aux_data, children):
        """
        Recreate a DynamicJaxArray from flattened data.
        """
        obj = cls()
        obj.free_qubit_list = children[0]
        
        obj.max_size = aux_data[0]
        obj.mm_style = aux_data[1]
        
        return obj

    # Add this method to make the class compatible with jax.tree_util
    def tree_flatten(self):
        return self.flatten()

    # Add this class method to make the class compatible with jax.tree_util
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.unflatten(aux_data, children)


@jax.jit
def request_qubit_helper(free_qubits, amount):
    
    res_qubits = Jlist()
    
    def loop_body(i, val_tuple):
        free_qubits, reg_qubits = val_tuple
        reg_qubits.append(free_qubits.pop())
        return free_qubits, reg_qubits
    
    free_qubits, res_qubits = jax.lax.fori_loop(0, amount, loop_body, (free_qubits, res_qubits))
    
    return free_qubits, res_qubits

    