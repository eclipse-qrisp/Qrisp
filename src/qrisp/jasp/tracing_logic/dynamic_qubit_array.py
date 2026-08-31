# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Defines DynamicQubitArray, a Jasp-compatible tracer-backed dynamic array of qubits."""

from jax import tree_util

from qrisp.jasp.primitives import AbstractQubit, fuse_qb_array, get_size, slice_qb_array
from qrisp.jasp.primitives.abstract_quantum_register import (
    _get_or_cache_qubit,
    _normalize_slice_bounds,
)


class DynamicQubitArray:
    """A Jasp-compatible dynamic array of qubits."""

    def __init__(self, tracer):
        self.tracer = tracer

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = _normalize_slice_bounds(self.tracer, key)
            return DynamicQubitArray(slice_qb_array(self.tracer, start, stop))

        return _get_or_cache_qubit(self.tracer, key)

    @property
    def size(self):
        return get_size(self.tracer)

    def __add__(self, other):
        if isinstance(other, DynamicQubitArray):
            other = other.tracer
        if isinstance(other, list):
            temp = self
            for x in other:
                if not isinstance(getattr(x, "aval", x), AbstractQubit):
                    raise ValueError(
                        "Can only concatenate type AbstractQubit or list[AbstractQubit] to DynamicQubitArray"
                    )
                temp += x
            return temp
        return DynamicQubitArray(fuse_qb_array(self.tracer, other))

    def __radd__(self, other):
        if isinstance(other, DynamicQubitArray):
            other = other.tracer
        if isinstance(other, list):
            for x in other:
                if not isinstance(getattr(x, "aval", x), AbstractQubit):
                    raise ValueError(
                        "Can only concatenate type AbstractQubit or list[AbstractQubit] to DynamicQubitArray"
                    )
            for x in reversed(other):
                self = DynamicQubitArray(fuse_qb_array(x, self.tracer))
            return self

        return DynamicQubitArray(fuse_qb_array(other, self.tracer))

    @property
    def reg(self):
        return self

    def measure(self):
        from qrisp.jasp import Measurement_p, TracingQuantumSession

        qs = TracingQuantumSession.get_instance()
        res, qs.abs_qst = Measurement_p.bind(self.tracer, qs.abs_qst)
        return res


def flatten_dqa(dqa):
    return (dqa.tracer,), None


def unflatten_dqa(_, children):
    return DynamicQubitArray(children[0])


tree_util.register_pytree_node(DynamicQubitArray, flatten_dqa, unflatten_dqa)
