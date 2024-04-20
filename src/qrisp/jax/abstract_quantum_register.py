# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:58:12 2024

@author: sea
"""

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings
from qrisp.jax import QuantumPrimitive

get_qubit_p = QuantumPrimitive("get_qubit")
put_qubit_p = QuantumPrimitive("put_qubit")

class AbstractQubitArray(AbstractValue):
    
    def __repr__(self):
        return "QubitArray"

    
def get_qubit(qb_array, index):
    return get_qubit_p.bind(qb_array, index)

def put_qubit(qb_array, index):
    return put_qubit_p.bind(qb_array, index)




raise_to_shaped_mappings[AbstractQubitArray] = lambda aval, _: aval


from qrisp.jax import AbstractQubit

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

@put_qubit_p.def_abstract_eval
def put_qubit_abstract_eval(qb_array):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return qb_array
