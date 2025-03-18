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

from jax.core import AbstractValue, raise_to_shaped_mappings
from qrisp.jasp.primitives import QuantumPrimitive, AbstractQubitArray

class AbstractQuantumCircuit(AbstractValue):

    def __repr__(self):
        return "QuantumCircuit"
    
    def __hash__(self):
        return hash(AbstractQuantumCircuit)
    
    def __eq__(self, other):
        return isinstance(other, AbstractQuantumCircuit)

def create_qubits(size, state):
    return create_qubits_p.bind(size, state)
        
raise_to_shaped_mappings[AbstractQuantumCircuit] = lambda aval, _: aval

# Register Creation
create_qubits_p = QuantumPrimitive("create_qubits")
create_qubits_p.multiple_results = True

@create_qubits_p.def_abstract_eval
def create_qubits_abstract_eval(size, qc):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    
    return AbstractQubitArray(), AbstractQuantumCircuit()

@create_qubits_p.def_impl
def create_qubit_impl(size, qc):
    from qrisp.circuit import QubitAlloc
    qubit_list = []
    
    for i in range(int(size)):
        qubit_list.append(qc.add_qubit())
        qc.append(QubitAlloc(), [qubit_list[-1]])
    
    return qc, qubit_list

# Register Deletion
delete_qubits_p = QuantumPrimitive("delete_qubits")

@delete_qubits_p.def_abstract_eval
def delete_qubits_abstract_eval(qarr, qc):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    
    return AbstractQuantumCircuit()

@delete_qubits_p.def_impl
def delete_qubits_impl(qarr, qc):
    from qrisp.circuit import QubitDealloc
    
    for i in range(len(qarr)):
        qc.append(QubitDealloc(), [qarr[i]])
    
    return qc

quantum_kernel_p = QuantumPrimitive("quantum_kernel")

@quantum_kernel_p.def_abstract_eval
def quantum_kernel_abstract_eval():
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    
    return AbstractQuantumCircuit()