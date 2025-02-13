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

import jax.numpy as jnp
from jax.core import AbstractValue, Primitive, raise_to_shaped_mappings, ShapedArray
from qrisp.jasp.primitives import QuantumPrimitive, AbstractQubit

get_qubit_p = QuantumPrimitive("get_qubit")
get_size_p = QuantumPrimitive("get_size")
slice_p = QuantumPrimitive("slice")
fuse_p = QuantumPrimitive("fuse")

class AbstractQubitArray(AbstractValue):
    
    def __repr__(self):
        return "QubitArray"
    
    def __eq__(self, other):
        return type(self) == type(other)
    
    def __hash__(self):
        return hash(type(self))
    
    def _getitem(self, tracer, key):
        if isinstance(key, slice):
            start = key.start
            if key.start is None:
                start = 0
            stop = key.stop
            if key.stop is None:
                stop = get_size(tracer) - 1
            
            return slice_qb_array(tracer, start, stop)
        else:
            id_tuple = (id(tracer), id(key))
            from qrisp.jasp import TracingQuantumSession
            qs = TracingQuantumSession.get_instance()
            if not id_tuple in qs.qubit_cache:
                from qrisp.jasp import get_qubit
                qs.qubit_cache[id_tuple] = get_qubit(tracer, key)
            return qs.qubit_cache[id_tuple]
        
    
def get_qubit(qb_array, index):
    return get_qubit_p.bind(qb_array, index)

def get_size(qb_array):
    return get_size_p.bind(qb_array)
    
def slice_qb_array(qb_array, start, stop):
    return slice_p.bind(qb_array, start, stop)

def fuse_qb_array(qb_array_0, qb_array_1):
    return fuse_p.bind(qb_array_0, qb_array_1)


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


@get_size_p.def_abstract_eval
def get_size_abstract_eval(qb_array):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return ShapedArray((), dtype = "int64")


@slice_p.def_abstract_eval
def get_slice_abstract_eval(qb_array, start, stop):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return AbstractQubitArray()

@fuse_p.def_abstract_eval
def fuse_abstract_eval(qb_array_0, qb_array_1):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return AbstractQubitArray()


@get_qubit_p.def_impl
def get_qubit_impl(qb_array, index):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return qb_array[index]

@get_size_p.def_impl
def get_size_impl(qb_array):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return len(qb_array)

@slice_p.def_impl
def get_slice_impl(qb_array, start, stop):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return qb_array[start:stop]

@fuse_p.def_impl
def fuse_impl(qb_array_0, qb_array_1):
    """Abstract evaluation of the primitive.
    
    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments. 
    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.
    """
    return qb_array_0 + qb_array_1