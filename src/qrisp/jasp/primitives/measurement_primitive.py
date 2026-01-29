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

from jax.core import ShapedArray
from qrisp.circuit import Reset, Qubit

from qrisp.jasp.primitives import (
    AbstractQuantumCircuit,
    AbstractQubit,
    QuantumPrimitive,
    AbstractQubitArray,
)

# Create the primitive
Measurement_p = QuantumPrimitive("measure")


@Measurement_p.def_abstract_eval
def measure_abstract_eval(meas_object, qc):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """

    if isinstance(meas_object, AbstractQubit):
        return ShapedArray((), bool), AbstractQuantumCircuit()
    elif isinstance(meas_object, AbstractQubitArray):
        return ShapedArray((), dtype="int64"), AbstractQuantumCircuit()
    else:
        raise Exception(
            f"Tried to call measurement primitive with type {type(meas_object)}"
        )


Measurement_p.multiple_results = True


@Measurement_p.def_impl
def measure_implementation(meas_object, qc):
    from qrisp import Qubit, QuantumCircuit, Clbit

    return_bool = False
    if isinstance(meas_object, Qubit):
        meas_object = [meas_object]
        return_bool = True

    if isinstance(qc, QuantumCircuit):
        if return_bool:
            meas_res = Clbit("cb_" + str(len(qc.clbits)))
            qc.clbits.insert(0, meas_res)
            qc.measure(meas_object, meas_res)
            return meas_res, qc
        else:
            clbit_list = []
            for i in range(len(meas_object)):
                meas_res = Clbit("cb_" + str(len(qc.clbits)))
                qc.clbits.insert(0, meas_res)
                qc.measure(meas_object[i], meas_res)
                clbit_list.append(meas_res)
            return clbit_list, qc
    else:
        res = 0
        for i in range(len(meas_object)):
            res += 2**i * qc.measure([meas_object[i]])

        if return_bool:
            return bool(res), qc
        return res, qc


reset_p = QuantumPrimitive("reset")


@reset_p.def_abstract_eval
def reset_abstract_eval(reset_object, qc):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return AbstractQuantumCircuit()


@reset_p.def_impl
def reset_implementation(reset_object, qc):
    if isinstance(reset_object, Qubit):
        reset_object = [reset_object]
    for i in range(len(reset_object)):
        qc.reset([reset_object[i]])
    return qc
