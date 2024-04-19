# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:36:11 2024

@author: sea
"""

from jax.core import Primitive
# Wrapper to identify Qrisp primitives
class QuantumPrimitive(Primitive):
    pass