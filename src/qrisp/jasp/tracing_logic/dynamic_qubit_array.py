"""
********************************************************************************
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
********************************************************************************
"""

from builtins import id

from jax import tree_util

from qrisp.jasp.primitives import (
    AbstractQubit,
    fuse_qb_array,
    get_qubit,
    get_size,
    slice_qb_array,
)


class DynamicQubitArray:
    """A Jasp-compatible dynamic array of qubits."""

    def __init__(self, tracer):
        self.tracer = tracer

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise NotImplementedError(
                    "Slicing with DynamicQubitArray only supports step=1"
                )
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else get_size(self.tracer)
            return DynamicQubitArray(slice_qb_array(self.tracer, start, stop))

        from qrisp.jasp.tracing_logic.tracing_quantum_session import (
            TracingQuantumSession,
        )

        qs = TracingQuantumSession.get_instance()
        id_tuple = (id(self.tracer), id(key))
        if not id_tuple in qs.qubit_cache:
            qs.qubit_cache[id_tuple] = get_qubit(self.tracer, key)
        return qs.qubit_cache[id_tuple]

    @property
    def size(self):
        return get_size(self.tracer)

    def __add__(self, other):
        if isinstance(other, DynamicQubitArray):
            other = other.tracer
        if isinstance(other, list):
            temp = self
            for x in other:
                if not isinstance(other, AbstractQubit):
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
            temp = self
            for x in other[::-1]:
                if not isinstance(other, AbstractQubit):
                    raise ValueError(
                        "Can only concatenate type AbstractQubit or list[AbstractQubit] to DynamicQubitArray"
                    )
                temp += x
            return temp

        return DynamicQubitArray(fuse_qb_array(other, self.tracer))

    @property
    def reg(self):
        return self

    def measure(self):
        from qrisp.jasp import Measurement_p, TracingQuantumSession

        qs = TracingQuantumSession.get_instance()
        res, qs.abs_qc = Measurement_p.bind(self.tracer, qs.abs_qc)
        return res


def flatten_dqa(dqa):
    return (dqa.tracer,), None


def unflatten_dqa(_, children):
    return DynamicQubitArray(children[0])


tree_util.register_pytree_node(DynamicQubitArray, flatten_dqa, unflatten_dqa)
