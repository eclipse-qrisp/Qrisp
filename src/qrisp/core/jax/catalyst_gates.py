# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:55:09 2024

@author: sea
"""

import qrisp.circuit.standard_operations as std_ops
def append_operation(operation, qubits=[], clbits=[]):
    from qrisp import find_qs
    
    qs = find_qs(qubits)
    
    qs.append(operation, qubits, clbits)

from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings, ShapedArray

from qrisp.core.jax import AbstractQuantumState, AbstractQubit, QrispPrimitive

##########
# X-Gate #
##########

XGate_p = QrispPrimitive("XGate")  # Create the primitive

def x_prim(qb, state):
    """The JAX-traceable way to use the JAX primitive.
    
    Note that the traced arguments must be passed as positional arguments
    to `bind`. 
    """
    return XGate_p.bind(qb, state)

def x_abstract_eval(state, qb):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    if state.burned:
        raise Exception("Tried to apply gate onto burned state")
    state.burned = True
    
    return AbstractQuantumState()

XGate_p.def_abstract_eval(x_abstract_eval)

##########
# H-Gate #
##########

HGate_p = QrispPrimitive("HGate")  # Create the primitive

def h_prim(qb, state):
    """The JAX-traceable way to use the JAX primitive.
    
    Note that the traced arguments must be passed as positional arguments
    to `bind`. 
    """
    return HGate_p.bind(qb, state)

def h_abstract_eval(state, qb):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    if state.burned:
        raise Exception("Tried to apply gate onto burned state")
    state.burned = True
    
    return AbstractQuantumState()

HGate_p.def_abstract_eval(h_abstract_eval)


###########
# CX-Gate #
###########

CXGate_p = QrispPrimitive("CXGate")  # Create the primitive

def cx_prim(state, qb_0, qb_1):
    """The JAX-traceable way to use the JAX primitive.
    
    Note that the traced arguments must be passed as positional arguments
    to `bind`. 
    """
    return CXGate_p.bind(state, qb_0, qb_1)

def cx_abstract_eval(state, qb_0, qb_1):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    if state.burned:
        raise Exception("Tried to apply gate onto burned state")
    state.burned = True
    
    return AbstractQuantumState()

CXGate_p.def_abstract_eval(cx_abstract_eval)


Measurement_p = QrispPrimitive("measure")  # Create the primitive

def measure_prim(state, qb):
    """The JAX-traceable way to use the JAX primitive.
    
    Note that the traced arguments must be passed as positional arguments
    to `bind`. 
    """
    return Measurement_p.bind(state, qb)

def measure_abstract_eval(state, qb):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    if state.burned:
        raise Exception("Tried to apply gate onto burned state")
    state.burned = True
    
    assert isinstance(qb, AbstractQubit)
    return AbstractQuantumState(), ShapedArray((), bool)

Measurement_p.def_abstract_eval(measure_abstract_eval)

Measurement_p.multiple_results = True

import qrisp.circuit.standard_operations as std_ops
translation_dic = {"x" : XGate_p, "cx" : CXGate_p, "measure" : Measurement_p, "h" : HGate_p}