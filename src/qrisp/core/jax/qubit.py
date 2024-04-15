# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:06:04 2024

@author: sea
"""

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings

class AbstractQubit(AbstractValue):
    pass

raise_to_shaped_mappings[AbstractQubit] = lambda aval, _: aval


