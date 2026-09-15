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

"""Provide a fixed-capacity JAX-compatible list backed by scalar leaves."""

# Scalar List
# ===========
#
# A JAX-pytree-compatible, fixed-capacity list backed by ``max_size``
# *individual* scalar leaves (a Python tuple of 0-d JAX values) instead of a
# single ranked ``jnp.array``.
#
# This exists as an alternative to :class:`~qrisp.jasp.interpreter_tools.dynamic_list.Jlist`
# for use in lowering paths (such as the CUDA-Q / Quake backend) that cannot
# represent ranked tensors at all: the Quake/CC dialects have no ``tensor`` or
# ``linalg`` equivalent, only scalar registers and pointer-addressed CC arrays.
# Any ``Jlist`` operation (``.at[i].set(...)``, ``jnp.roll``, dynamic
# ``fori_loop``-based writes) lowers to ``linalg.generic``/``tensor.insert``
# patterns that have no CC-dialect representation.
#
# ``ScalarList`` sidesteps this entirely by never constructing a ranked tensor:
# every "array access" becomes a chain of ``jax.lax.select`` over the
# statically-known ``max_size`` slots. This costs O(max_size) (or
# O(max_size**2) for slicing) scalar operations instead of O(1) tensor
# indexing, which is an acceptable trade-off for the small, fixed register
# sizes (tens of qubits) this class is intended for.
#
# API mirrors the subset of ``Jlist`` used by the static-register interpreter:
# ``append``, ``pop``, ``prepend``, ``extend``, ``clear``, ``__getitem__``
# (both integer and slice keys), ``__len__``, ``copy``.

import copy

import jax
import jax.numpy as jnp
from jax import lax


@jax.tree_util.register_pytree_node_class
class ScalarList:
    """Fixed-capacity, JAX-pytree-compatible list backed by scalar leaves instead of a tensor."""

    fill_value = 0

    def __init__(self, init_val=None, max_size=64):
        """Build a ``ScalarList`` of ``max_size`` slots, optionally seeded from ``init_val``."""
        self.max_size = max_size

        if init_val is None:
            self.slots = tuple(jnp.asarray(0, dtype=jnp.int64) for _ in range(max_size))
            self.counter = jnp.asarray(0, dtype=jnp.int64)
            return

        values = list(init_val)
        n = len(values)

        slots = []
        for i in range(max_size):
            if i < n:
                slots.append(jnp.asarray(values[i], dtype=jnp.int64))
            else:
                slots.append(jnp.asarray(0, dtype=jnp.int64))
        self.slots = tuple(slots)
        self.counter = jnp.asarray(min(n, max_size), dtype=jnp.int64)

    # -- internal helpers: dynamic positional access without tensors -------

    def _select_at(self, index):
        """Return the scalar stored at dynamic ``index`` (0 <= index < max_size)."""
        result = self.slots[0]
        for i in range(1, self.max_size):
            result = lax.select(index == i, self.slots[i], result)
        return result

    def _with_slot_set(self, index, value):
        """Return a new slots tuple with dynamic position ``index`` set to ``value``."""
        return tuple(lax.select(index == i, value, self.slots[i]) for i in range(self.max_size))

    # -- Jlist-compatible API ------------------------------------------------

    def append(self, value):
        """Append ``value`` as the last element, dropping it if the list is already full."""
        value = jnp.asarray(value, dtype=jnp.int64)
        self.slots = self._with_slot_set(self.counter, value)
        self.counter = jnp.minimum(self.counter + 1, self.max_size)
        return self

    def prepend(self, value):
        """Insert ``value`` as the first element, shifting the rest right by one slot."""
        value = jnp.asarray(value, dtype=jnp.int64)
        self.slots = (value,) + self.slots[:-1]
        self.counter = jnp.minimum(self.counter + 1, self.max_size)
        return self

    def pop(self):
        """Remove and return the last element."""
        new_counter = self.counter - 1
        value = self._select_at(new_counter)
        self.counter = new_counter
        return value

    def extend(self, values):
        """Append every element of another ``ScalarList`` to this one, in order."""
        for j in range(values.max_size):
            active = j < values.counter
            candidate = values.slots[j]
            self.slots = tuple(
                lax.select(jnp.logical_and(active, self.counter == i), candidate, self.slots[i])
                for i in range(self.max_size)
            )
            self.counter = jnp.minimum(self.counter + active.astype(jnp.int64), self.max_size)
        return self

    def clear(self):
        """Reset the list to empty without touching the underlying slots."""
        self.counter = jnp.asarray(0, dtype=jnp.int64)
        return self

    def __getitem__(self, key):
        """Return the element at an integer index, or a new ``ScalarList`` for a slice."""
        if isinstance(key, slice):
            if key.start is None:
                start = jnp.asarray(0, dtype=jnp.int64)
            else:
                start = key.start + (key.start < 0) * self.counter

            if key.stop is None:
                stop = self.counter
            else:
                stop = jnp.minimum(key.stop, self.counter)
                stop = stop + (stop < 0) * self.counter

            length = stop - start

            new_slots = []
            for k in range(self.max_size):
                active = k < length
                val = self._select_at(start + k)
                new_slots.append(lax.select(active, val, jnp.asarray(0, dtype=jnp.int64)))

            res = ScalarList.__new__(ScalarList)
            res.max_size = self.max_size
            res.slots = tuple(new_slots)
            res.counter = length
            return res
        else:
            norm_key = key + (key < 0) * self.counter
            return self._select_at(norm_key)

    def __len__(self):
        """Return the current number of elements (not ``max_size``)."""
        return int(self.counter)

    def copy(self):
        """Return a shallow copy of this ``ScalarList``."""
        return copy.copy(self)

    # -- pytree registration --------------------------------------------------

    def tree_flatten(self):
        """Flatten into the JAX-pytree ``(children, aux_data)`` representation."""
        return (self.slots + (self.counter,), self.max_size)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild a ``ScalarList`` from the flattened pytree representation."""
        obj = cls.__new__(cls)
        obj.max_size = aux_data
        obj.slots = tuple(children[:-1])
        obj.counter = children[-1]
        return obj
