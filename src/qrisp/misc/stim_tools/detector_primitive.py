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

import numpy as np

from qrisp.circuit import Operation, QuantumCircuit
from qrisp.misc.stim_tools.stim_primitive import StimPrimitive

detector_p = StimPrimitive("detector")

from jax.core import ShapedArray

@detector_p.def_abstract_eval
def detector_abstract_eval(*measurements_and_abs_qc):
    measurements = measurements_and_abs_qc[:-1]
    
    for b in measurements:
        if not isinstance(b, ShapedArray) or not isinstance(b.dtype, np.dtypes.BoolDType):
            raise Exception(f"Tried to trace detector with value {b} (permitted is boolean)")
    
    return ShapedArray((), bool)

def detector(*measurements):
    from qrisp.jasp import TracingQuantumSession
    qs = TracingQuantumSession.get_instance()
    return detector_p.bind(*(list(measurements) + [qs.abs_qc]))

@detector_p.def_impl
def detector_implementation(*measurements_and_qc):
    measurements = measurements_and_qc[:-1]
    qc = measurements_and_qc[-1]
    res = qc.add_clbit()
    qc.append(StimDetector(len(measurements)), clbits = list(measurements) + [res])
    return res


class StimDetector(Operation):
    def __init__(self, num_inputs):
        
        definition = QuantumCircuit(0, num_inputs + 1)
        Operation.__init__(self, "stim.detector", num_clbits = num_inputs + 1, definition = definition)