# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:58:12 2024

@author: sea
"""

from qrisp.jasp.primitives import get_qubit, slice_qb_array, get_size


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
                stop = get_size(tracer) - 1
            
            return DynamicQubitArray(slice_qb_array(tracer, start, stop))
        else:
            from qrisp.jasp.tracing_quantum_session import TracingQuantumSession
            id_tuple = (id(tracer), id(key))
            qs = TracingQuantumSession.get_instance()
            if not id_tuple in qs.qubit_cache:
                qs.qubit_cache[id_tuple] = get_qubit(tracer, key)
            return qs.qubit_cache[id_tuple]
    
    @property
    def size(self):
        return get_size(self.tracer)
        
from jax import tree_util
from builtins import id


def flatten_dqa(dqa):
    return (dqa.tracer,), None

def unflatten_dqa(aux_data, children):
    return DynamicQubitArray(children[0])

tree_util.register_pytree_node(DynamicQubitArray, flatten_dqa, unflatten_dqa)