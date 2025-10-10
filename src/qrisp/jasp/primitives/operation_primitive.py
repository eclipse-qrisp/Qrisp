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

from sympy import symbols
import jax.numpy as jnp
import jax

from qrisp.jasp.primitives import (
    QuantumPrimitive,
    AbstractQuantumCircuit,
    AbstractQubit,
)

greek_letters = symbols(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
)

    
quantum_gate_p = QuantumPrimitive("quantum_gate")

@quantum_gate_p.def_impl
def append_impl(*args, **kwargs):

    gate = kwargs["gate"]
    
    qc = args[-1]
    args = args[:-1]
    """Concrete evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    actual instances. 
    """
    qubit_args = args[: gate.num_qubits]
    parameter_args = args[gate.num_qubits :]

    temp_op = gate.bind_parameters(
        {
            greek_letters[i]: float(parameter_args[i])
            for i in range(len(parameter_args))
        }
    )
    qc.append(temp_op, list(qubit_args))
    return qc

@quantum_gate_p.def_abstract_eval
def abstract_eval(*args, **kwargs):
    
    gate = kwargs["gate"]
    qc = args[-1]
    qubit_args = [args[i] for i in range(gate.num_qubits)]
    parameter_args = [args[i] for i in range(gate.num_qubits, len(args) - 1)]

    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    """
    if not isinstance(qc, AbstractQuantumCircuit):
        raise Exception(
            f"Tried to execute OperationPrimitive.bind with the last argument of type {type(qc)} instead of AbstractQuantumCircuit"
        )

    if not all([isinstance(qb, AbstractQubit) for qb in qubit_args]):
        raise Exception(
            f"Tried to execute {gate.name} with incompatible qubit tracers {[type(qb) for qb in qubit_args]}"
        )

    if not all(
        [
            isinstance(param, jnp.number)
            or (
                isinstance(param, jax.core.ShapedArray)
                and len(param.shape) == 0
            )
            for param in parameter_args
        ]
    ):
        raise Exception(
            f"Tried to execute Operation {gate.name} with incompatible parameter types {[type(param) for param in parameter_args]} (required are number types)"
        )

    return AbstractQuantumCircuit()
