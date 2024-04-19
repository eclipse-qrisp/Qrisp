# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:58:12 2024

@author: sea
"""

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings
from qrisp.jax import QuantumPrimitive

get_qubit_p = QuantumPrimitive("get_qubit")

class AbstractQuantumRegister(AbstractValue):
    pass
    
def get_qubit(reg, index):
    return get_qubit_p.bind(reg, index)



raise_to_shaped_mappings[AbstractQuantumRegister] = lambda aval, _: aval


from qrisp.jax import AbstractQubit

def get_qubit_abstract_eval(register, index):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return AbstractQubit()

get_qubit_p.def_abstract_eval(get_qubit_abstract_eval)