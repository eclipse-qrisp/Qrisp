# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:06:04 2024

@author: sea
"""

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings

class AbstractQubit(AbstractValue):
    
    def __repr__(self):
        return "Qubit"
    
    def __hash__(self):
        return hash(type(self))
    
    def __eq__(self, other):
        if not isinstance(other, AbstractQubit):
            return False
        return isinstance(other, AbstractQubit)

raise_to_shaped_mappings[AbstractQubit] = lambda aval, _: aval


