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

from jax.core import ShapedArray

from qrisp.jisp.primitives import AbstractQuantumCircuit, AbstractQubit, QuantumPrimitive, AbstractQubitArray

# Create the primitive
Measurement_p = QuantumPrimitive("measure")  

@Measurement_p.def_abstract_eval
def measure_abstract_eval(qc, meas_object):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    
    if isinstance(meas_object, AbstractQubit):
        return AbstractQuantumCircuit(), ShapedArray((), bool)
    elif isinstance(meas_object, AbstractQubitArray):
        return AbstractQuantumCircuit(), ShapedArray((), int)
    else:
        raise Exception(f"Tried to call measurement primitive with type {type(meas_object)}")

# Measurement_p.num_qubits = 1
Measurement_p.multiple_results = True


@Measurement_p.def_impl
def measure_abstract_eval(qc, meas_object):
    from qrisp import get_measurement_from_qc, Qubit, default_backend
    
    if isinstance(meas_object, Qubit):
        res = get_measurement_from_qc(qc, [meas_object], shots = 1, backend = default_backend)
        return qc, bool(list(res.keys())[0])
    else:
        res = get_measurement_from_qc(qc, meas_object, shots = 1, backend = default_backend)
        return qc, int(list(res.keys())[0])
