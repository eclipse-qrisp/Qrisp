# """
# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************
# """

"""Defines :class:`AbstractQuantumState` and its create_qubits/delete_qubits/quantum_kernel primitives."""

from jax.core import AbstractValue

from qrisp.circuit import QubitAlloc, QubitDealloc
from qrisp.jasp.primitives.abstract_quantum_register import AbstractQubitArray
from qrisp.jasp.primitives.quantum_primitive import QuantumPrimitive


class AbstractQuantumState(AbstractValue):
    """JAX abstract value representing the quantum state threaded through tracing."""

    def __init__(self):
        self.vma = None
        AbstractValue.__init__(self)

    def __repr__(self):
        return "QuantumState"

    def __hash__(self):
        return hash(AbstractQuantumState)

    def __eq__(self, other):
        return isinstance(other, AbstractQuantumState)


def create_qubits(size, state):
    """Bind the create_qubits primitive."""
    return create_qubits_p.bind(size, state)


# Register Creation
create_qubits_p = QuantumPrimitive("create_qubits")
create_qubits_p.multiple_results = True


@create_qubits_p.def_abstract_eval
def create_qubits_abstract_eval(_size, qc):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.

    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.

    """
    assert isinstance(qc, AbstractQuantumState)

    return AbstractQubitArray(), AbstractQuantumState()


@create_qubits_p.def_impl
def create_qubit_impl(size, qc):
    """Concrete evaluation of the create_qubits primitive."""
    qubit_list = []

    for _ in range(int(size)):
        qubit_list.append(qc.add_qubit())
        qc.append(QubitAlloc(), [qubit_list[-1]])

    return qubit_list, qc


# Register Deletion
delete_qubits_p = QuantumPrimitive("delete_qubits")


@delete_qubits_p.def_abstract_eval
def delete_qubits_abstract_eval(_qarr, _qc):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.

    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.

    """
    return AbstractQuantumState()


@delete_qubits_p.def_impl
def delete_qubits_impl(qarr, qc):
    """Concrete evaluation of the delete_qubits primitive."""
    for qubit in qarr:
        qc.append(QubitDealloc(), [qubit])

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
    return AbstractQuantumState()
