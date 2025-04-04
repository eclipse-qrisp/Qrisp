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

from qrisp.jasp.primitives import get_qubit, slice_qb_array, get_size, fuse_qb_array


class DynamicQubitArray:
    
    def __init__(self, tracer):
        self.tracer = tracer
    
    def __getitem__(self, key):
        tracer = self.tracer
        if isinstance(key, slice):
            start = key.start
            if key.start is None:
                start = 0
            stop = key.stop
            if key.stop is None:
                stop = get_size(tracer)
            
            return DynamicQubitArray(slice_qb_array(tracer, start, stop))
        else:
            from qrisp.jasp.tracing_logic.tracing_quantum_session import TracingQuantumSession
            qs = TracingQuantumSession.get_instance()
            id_tuple = (id(tracer), id(key))
            if not id_tuple in qs.qubit_cache:
                qs.qubit_cache[id_tuple] = get_qubit(tracer, key)
            return qs.qubit_cache[id_tuple]
    
    @property
    def size(self):
        return get_size(self.tracer)
    
    def __add__(self, other):
        return DynamicQubitArray(fuse_qb_array(self.tracer, other.tracer))
    
    @property
    def reg(self):
        return self
    
    def measure(self):
        from qrisp.jasp import TracingQuantumSession, Measurement_p
        qs = TracingQuantumSession.get_instance()
        res, qs.abs_qc = Measurement_p.bind(self.tracer, qs.abs_qc)
        return res
        
from jax import tree_util
from builtins import id


def flatten_dqa(dqa):
    return (dqa.tracer,), None

def unflatten_dqa(aux_data, children):
    return DynamicQubitArray(children[0])

tree_util.register_pytree_node(DynamicQubitArray, flatten_dqa, unflatten_dqa)