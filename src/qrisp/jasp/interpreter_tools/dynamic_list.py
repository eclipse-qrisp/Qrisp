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

import copy

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class Jlist:

    fill_value = 0

    def __init__(self, init_val=None, max_size=int(2**10)):
        self.max_size = max_size
        self.array, self.counter = self._create_dynamic_array(init_val)

    def _create_dynamic_array(self, init_val):
        jax_array = jnp.zeros(self.max_size, dtype=jnp.int64)

        n = 0

        if init_val is not None:

            if isinstance(init_val, list):
                n = len(init_val)
            else:
                n = init_val.size

            if n == self.max_size:
                jax_array = init_val
            else:
                # Create an index array for updating
                idx = jnp.arange(min(n, jax_array.size), dtype=jnp.int64)

                # Use JAX's index_update to fill the array
                jax_array = jax_array.at[idx].set(
                    jnp.array(init_val[: jax_array.size], dtype=jnp.int64),
                    indices_are_sorted=True,
                    unique_indices=True,
                )

        return jax_array, jnp.array(min(n, self.max_size), dtype=jnp.int64)

    def append(self, value):
        self.array, self.counter = self._append(value)
        return self

    @jax.jit
    def _append(self, value):
        new_array = self.array.at[self.counter].set(value)
        new_counter = jnp.minimum(self.counter + 1, self.array.shape[0])
        return new_array, new_counter

    def prepend(self, value):
        self.array, self.counter = self._prepend(value)
        return self

    @jax.jit
    def _prepend(self, value):
        new_array = jnp.roll(self.array, 1)
        new_array = new_array.at[0].set(value)
        new_counter = jnp.minimum(self.counter + 1, self.array.shape[0])
        return new_array, new_counter

    def pop(self):
        self.counter, value = self._pop()
        return value

    @jax.jit
    def _pop(self):
        new_counter = self.counter - 1
        value = self.array[new_counter]
        return new_counter, value

    def extend(self, values):
        self.array, self.counter = self._extend(self.array, self.counter, values)
        return self

    @jax.jit
    def _extend(self, array, counter, values):
        def body_fun(i, state):
            curr_array, curr_counter = state
            new_array = curr_array.at[curr_counter].set(values[i])
            new_counter = jnp.minimum(curr_counter + 1, self.max_size)
            return new_array, new_counter

        return jax.lax.fori_loop(0, values.counter, body_fun, (array, counter))

    @jax.jit
    def clear(self):
        self.array, self.counter = self._clear(self.array, self.counter)
        return self

    @staticmethod
    def _clear(array, counter):
        return array, jnp.array(0)

    def __getitem__(self, key):
        if isinstance(key, slice):

            if key.start is None:
                start = 0
            else:
                start = key.start + (key.start < 0)*self.counter

            if key.stop is None:
                stop = self.counter
            else:
                stop = jnp.minimum(key.stop, self.counter)
                stop = stop + (stop < 0)*self.counter

            length = stop - start

            def body_fun(i, state):
                new_array, old_array = state
                new_array = new_array.at[i].set(old_array[i + start])
                return new_array, old_array

            new_array = jnp.zeros(self.max_size, dtype=jnp.int64)

            new_array, _ = jax.lax.fori_loop(
                0, length, body_fun, (new_array, self.array)
            )

            res = Jlist.__new__(Jlist)
            res.array = new_array
            res.counter = length
            res.max_size = self.max_size

            return res
        else:
            return self.array[key + (key < 0)*self.counter]

    @jax.jit
    def _slice(array, counter, start, end):
        start = jnp.maximum(0, start)
        end = jnp.minimum(counter, end)
        return array[start:end]

    def __len__(self):
        return int(self.counter)

    def copy(self):
        return copy.copy(self)

    def flatten(self):
        """
        Flatten the DynamicJaxArray into a tuple of arrays and auxiliary data.
        This is useful for JAX transformations and serialization.
        """
        return (self.array, self.counter), tuple()

    @classmethod
    def unflatten(cls, aux_data, children):
        """
        Recreate a DynamicJaxArray from flattened data.
        """
        array, counter = children
        obj = cls(max_size = array.shape[0])
        obj.array = array
        obj.counter = counter
        return obj

    # Add this method to make the class compatible with jax.tree_util
    def tree_flatten(self):
        return self.flatten()

    # Add this class method to make the class compatible with jax.tree_util
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.unflatten(aux_data, children)
