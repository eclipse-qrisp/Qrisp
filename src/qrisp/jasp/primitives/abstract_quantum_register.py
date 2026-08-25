"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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
********************************************************************************
"""

from jax.core import AbstractValue, ShapedArray, Tracer

from qrisp.jasp.primitives.abstract_qubit import AbstractQubit
from qrisp.jasp.primitives.quantum_primitive import QuantumPrimitive

get_qubit_p = QuantumPrimitive("get_qubit")
get_size_p = QuantumPrimitive("get_size")
slice_p = QuantumPrimitive("slice")
fuse_p = QuantumPrimitive("fuse")


class AbstractQubitArray(AbstractValue):
    """JAX abstract value representing a traced array of qubits."""

    def __init__(self):
        self.vma = None
        AbstractValue.__init__(self)

    def __repr__(self):
        return "QubitArray"

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def _getitem(self, tracer, key):
        if isinstance(key, slice):
            start, stop = _normalize_slice_bounds(tracer, key)
            return slice_qb_array(tracer, start, stop)

        return _get_or_cache_qubit(tracer, key)


def get_qubit(qb_array, index):
    """Bind the get_qubit primitive."""
    if not isinstance(index, (Tracer, int)):
        index = int(index)
    return get_qubit_p.bind(qb_array, index)


def get_size(qb_array):
    """Bind the get_size primitive."""
    return get_size_p.bind(qb_array)


def slice_qb_array(qb_array, start, stop):
    """Bind the slice primitive."""
    return slice_p.bind(qb_array, start, stop)


def _normalize_slice_bounds(tracer, key: slice) -> tuple:
    """Normalize a slice key's start/stop into concrete bounds for a QubitArray.

    ``stop`` defaults to the array's full size, matching Python's usual exclusive-stop
    slicing convention (not ``size - 1``). Only step=1 (or unspecified) is supported.
    """
    if key.step is not None and key.step != 1:
        raise NotImplementedError("Slicing with DynamicQubitArray only supports step=1")
    start = key.start if key.start is not None else 0
    stop = key.stop if key.stop is not None else get_size(tracer)
    return start, stop


def _get_or_cache_qubit(tracer, key):
    """Look up (or bind and cache) the Qubit at index/key ``key`` of ``tracer``."""
    # Deferred import: qrisp.jasp.tracing_logic is loaded after
    # qrisp.jasp.primitives, so this can't be a top-level import.
    from qrisp.jasp import TracingQuantumSession

    qs = TracingQuantumSession.get_instance()
    id_tuple = (id(tracer), id(key))
    if id_tuple not in qs.qubit_cache:
        qs.qubit_cache[id_tuple] = get_qubit(tracer, key)
    return qs.qubit_cache[id_tuple]


def fuse_qb_array(qb_array_0, qb_array_1):
    """Bind the fuse primitive."""
    return fuse_p.bind(qb_array_0, qb_array_1)


@get_qubit_p.def_abstract_eval
def get_qubit_abstract_eval(_qb_array, _index):
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
def get_size_abstract_eval(_qb_array):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.

    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.

    """
    return ShapedArray((), dtype="int64")


@slice_p.def_abstract_eval
def get_slice_abstract_eval(_qb_array, _start, _stop):
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
def fuse_abstract_eval(arg_0, arg_1):
    """Abstract evaluation of the fuse primitive."""
    if not isinstance(arg_0, (AbstractQubit, AbstractQubitArray)):
        raise Exception(f"Tried to fuse type {type(arg_0)}")
    if not isinstance(arg_1, (AbstractQubit, AbstractQubitArray)):
        raise Exception(f"Tried to fuse type {type(arg_1)}")

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
def fuse_impl(arg_0, arg_1):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.

    Args:
      xs, ys, zs: abstractions of the arguments.
    Result:
      a ShapedArray for the result of the primitive.

    """
    if not isinstance(arg_0, list):
        arg_0 = [arg_0]

    if not isinstance(arg_1, list):
        arg_1 = [arg_1]

    return arg_0 + arg_1
