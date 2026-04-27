"""
********************************************************************************
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import Tracer

from qrisp.circuit.clbit import Clbit
from qrisp.circuit.qubit import Qubit
from qrisp.typing import ArrayLike, ClbitLike, QubitLike


class TestQubitLike:
    """Tests for the QubitLike type alias."""

    def test_qubit_object(self):
        """A Qubit object is an instance of QubitLike."""
        assert isinstance(Qubit("test_qubit"), QubitLike)

    @pytest.mark.parametrize("index", [0, 1, 5])
    def test_integer_index(self, index):
        """Integer indices are valid qubit specifiers."""
        assert isinstance(index, QubitLike)

    def test_list_of_qubits_and_ints(self):
        """A list of Qubit objects and integer indices is an instance of QubitLike."""
        assert isinstance([Qubit("q0"), Qubit("q1"), 2], QubitLike)

    def test_string_is_rejected(self):
        """A plain string is not a valid QubitLike specifier."""
        assert not isinstance("not_a_qubit", QubitLike)


class TestClbitLike:
    """Tests for the ClbitLike type alias."""

    def test_clbit_object(self):
        """A Clbit object is an instance of ClbitLike."""
        assert isinstance(Clbit("test_clbit"), ClbitLike)

    @pytest.mark.parametrize("index", [0, 1, 5])
    def test_integer_index(self, index):
        """Integer indices are valid classical bit specifiers."""
        assert isinstance(index, ClbitLike)

    def test_list_of_clbits_and_ints(self):
        """A list of Clbit objects and integer indices is an instance of ClbitLike."""
        assert isinstance([Clbit("c0"), Clbit("c1"), 2], ClbitLike)

    def test_string_is_rejected(self):
        """A plain string is not a valid ClbitLike specifier."""
        assert not isinstance("not_a_clbit", ClbitLike)


class TestArrayLike:
    """Tests for the ArrayLike type alias.

    ArrayLike contains only concrete types, so isinstance checks work for all
    component types, including JAX tracers (using from jax.core import Tracer,
    consistent with how the rest of the codebase checks for tracers).
    """

    # ------------------------------------------------------------------ #
    # Python scalars                                                      #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("value", [1, 1.0, 1 + 0j, True])
    def test_python_scalars(self, value):
        """Python scalar types (int, float, complex, bool) are instances of ArrayLike."""
        assert isinstance(value, ArrayLike)

    # ------------------------------------------------------------------ #
    # NumPy                                                               #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "value",
        [
            np.array([1, 2, 3]),
            np.array(1.0),
            np.float32(1.0),
            np.int64(1),
            np.complex128(1 + 2j),
        ],
    )
    def test_numpy_arrays_and_scalars(self, value):
        """NumPy arrays and typed scalars are instances of ArrayLike."""
        assert isinstance(value, ArrayLike)

    # ------------------------------------------------------------------ #
    # JAX                                                                 #
    # ------------------------------------------------------------------ #

    def test_jax_array(self):
        """A concrete jax.Array is an instance of ArrayLike."""
        assert isinstance(jnp.array([1.0, 2.0, 3.0]), ArrayLike)

    def test_jax_tracer(self):
        """JAX tracers encountered during tracing are instances of ArrayLike.

        Uses jax.make_jaxpr to trace a function, which passes abstract Tracer
        objects as inputs. This is the same pattern used throughout Qrisp.
        """
        results = []

        def f(x):
            results.append(isinstance(x, Tracer))
            results.append(isinstance(x, ArrayLike))
            return x

        jax.make_jaxpr(f)(1.0)
        assert results[
            0
        ], "make_jaxpr did not produce a Tracer (test precondition failed)"
        assert results[1], "Tracer is not an instance of ArrayLike"

    # ------------------------------------------------------------------ #
    # Rejection of non-array types                                        #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("value", ["string", [1, 2, 3], {"key": 1}, None, (1, 2)])
    def test_non_array_types_are_rejected(self, value):
        """Strings, plain lists, dicts, None, and tuples are not instances of ArrayLike."""
        assert not isinstance(value, ArrayLike)
