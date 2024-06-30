# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:58:12 2024

@author: sea
"""

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings
from qrisp.jax.primitives import QuantumPrimitive, AbstractQubit

get_qubit_p = QuantumPrimitive("get_qubit")

class AbstractQubitArray(AbstractValue):
    
    def __repr__(self):
        return "QubitArray"
    
def get_qubit(qb_array, index):
    return get_qubit_p.bind(qb_array, index)

raise_to_shaped_mappings[AbstractQubitArray] = lambda aval, _: aval

@get_qubit_p.def_abstract_eval
def get_qubit_abstract_eval(qb_array, index):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return AbstractQubit()
