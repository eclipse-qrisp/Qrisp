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
import sympy
from jax.core import Tracer
from sympy import Symbol

from qrisp.circuit.clbit import Clbit
from qrisp.circuit.qubit import Qubit
from qrisp.typing import (
    ArrayLike,
    ClbitLike,
    FloatLike,
    NDArrayLike,
    QubitLike,
    ScalarLike,
)


class TestQubitLike:
    """Tests for the QubitLike type alias."""

    def test_qubit_object(self):
        """A Qubit object is an instance of QubitLike."""
        assert isinstance(Qubit("test_qubit"), QubitLike)  # type: ignore[arg-type]

    @pytest.mark.parametrize("index", [0, 1, 5])
    def test_integer_index(self, index):
        """Integer indices are valid qubit specifiers."""
        assert isinstance(index, QubitLike)  # type: ignore[arg-type]


class TestClbitLike:
    """Tests for the ClbitLike type alias."""

    def test_clbit_object(self):
        """A Clbit object is an instance of ClbitLike."""
        assert isinstance(Clbit("test_clbit"), ClbitLike)  # type: ignore[arg-type]

    @pytest.mark.parametrize("index", [0, 1, 5])
    def test_integer_index(self, index):
        """Integer indices are valid classical bit specifiers."""
        assert isinstance(index, ClbitLike)  # type: ignore[arg-type]


class TestScalarLike:
    """Tests for the ScalarLike type alias."""

    @pytest.mark.parametrize("value", [1, 1.0, 1 + 0j, True])
    def test_python_scalars(self, value):
        """Python built-in scalar types are instances of ScalarLike."""
        assert isinstance(value, ScalarLike)

    @pytest.mark.parametrize(
        "value",
        [np.float32(1.0), np.int64(1), np.complex128(1 + 2j)],
    )
    def test_numpy_scalars(self, value):
        """NumPy typed scalars (np.generic subclasses) are instances of ScalarLike."""
        assert isinstance(value, ScalarLike)

    def test_jax_tracer(self):
        """JAX tracers encountered during tracing are instances of ScalarLike."""
        results = []

        def f(x):
            results.append(isinstance(x, Tracer))
            results.append(isinstance(x, ScalarLike))
            return x

        jax.make_jaxpr(f)(1.0)
        assert results[0], (
            "make_jaxpr did not produce a Tracer (test precondition failed)"
        )
        assert results[1], "Tracer is not an instance of ScalarLike"

    def test_numpy_array_is_rejected(self):
        """A NumPy ndarray is not a ScalarLike."""
        assert not isinstance(np.array([1, 2, 3]), ScalarLike)

    def test_jax_array_is_rejected(self):
        """A concrete JAX array is not a ScalarLike."""
        assert not isinstance(jnp.array([1.0, 2.0]), ScalarLike)

    def test_string_is_rejected(self):
        """A plain string is not a ScalarLike."""
        assert not isinstance("not_a_scalar", ScalarLike)


class TestNDArrayLike:
    """Tests for the NDArrayLike type alias."""

    def test_numpy_array(self):
        """A NumPy ndarray is an instance of NDArrayLike."""
        assert isinstance(np.array([1, 2, 3]), NDArrayLike)

    def test_jax_array(self):
        """A concrete jax.Array is an instance of NDArrayLike."""
        assert isinstance(jnp.array([1.0, 2.0, 3.0]), NDArrayLike)

    def test_jax_tracer(self):
        """JAX tracers encountered during tracing are instances of NDArrayLike."""
        results = []

        def f(x):
            results.append(isinstance(x, Tracer))
            results.append(isinstance(x, NDArrayLike))
            return x

        jax.make_jaxpr(f)(1.0)
        assert results[0], (
            "make_jaxpr did not produce a Tracer (test precondition failed)"
        )
        assert results[1], "Tracer is not an instance of NDArrayLike"

    def test_python_scalar_is_rejected(self):
        """A plain Python float is not an NDArrayLike."""
        assert not isinstance(3.14, NDArrayLike)

    def test_numpy_scalar_is_rejected(self):
        """A NumPy scalar (np.float32) is not an NDArrayLike."""
        assert not isinstance(np.float32(1.0), NDArrayLike)


class TestArrayLike:
    """Tests for the ArrayLike type alias (union of ScalarLike and NDArrayLike)."""

    @pytest.mark.parametrize("value", [1, 1.0, 1 + 0j, True])
    def test_python_scalars(self, value):
        """Python scalar types are instances of ArrayLike."""
        assert isinstance(value, ArrayLike)

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

    def test_jax_array(self):
        """A concrete jax.Array is an instance of ArrayLike."""
        assert isinstance(jnp.array([1.0, 2.0, 3.0]), ArrayLike)

    def test_jax_tracer(self):
        """JAX tracers encountered during tracing are instances of ArrayLike."""
        results = []

        def f(x):
            results.append(isinstance(x, Tracer))
            results.append(isinstance(x, ArrayLike))
            return x

        jax.make_jaxpr(f)(1.0)
        assert results[0], (
            "make_jaxpr did not produce a Tracer (test precondition failed)"
        )
        assert results[1], "Tracer is not an instance of ArrayLike"

    @pytest.mark.parametrize("value", ["string", [1, 2, 3], {"key": 1}, None, (1, 2)])
    def test_non_array_types_are_rejected(self, value):
        """Strings, plain lists, dicts, None, and tuples are not instances of ArrayLike."""
        assert not isinstance(value, ArrayLike)


class TestFloatLike:
    """Tests for the FloatLike type alias."""

    @pytest.mark.parametrize("value", [1.5, 0.0, -3.14])
    def test_python_float(self, value):
        """Python floats are valid gate parameters."""
        assert isinstance(value, FloatLike)

    @pytest.mark.parametrize("value", [0, 1, -5])
    def test_python_int(self, value):
        """Python ints are valid gate parameters."""
        assert isinstance(value, FloatLike)

    @pytest.mark.parametrize(
        "value",
        [np.float64(1.0), np.float32(0.5), np.int32(3)],
    )
    def test_numpy_numeric_scalars(self, value):
        """NumPy floating-point and integer scalars are valid gate parameters."""
        assert isinstance(value, FloatLike)

    def test_sympy_symbol(self):
        """A sympy.Symbol is a valid gate parameter."""
        assert isinstance(Symbol("phi"), FloatLike)

    def test_sympy_expression(self):
        """An arbitrary sympy expression is a valid gate parameter."""
        phi = Symbol("phi")
        assert isinstance(sympy.Integer(2) * phi + sympy.pi, FloatLike)

    def test_jax_tracer(self):
        """JAX tracers encountered during tracing are valid gate parameters."""
        results = []

        def f(x):
            results.append(isinstance(x, Tracer))
            results.append(isinstance(x, FloatLike))
            return x

        jax.make_jaxpr(f)(1.0)
        assert results[0], (
            "make_jaxpr did not produce a Tracer (test precondition failed)"
        )
        assert results[1], "Tracer is not an instance of FloatLike"

    @pytest.mark.parametrize(
        "value", ["phi", [1.0], None, np.array([1.0]), 1 + 2j, np.complex128(1 + 2j)]
    )
    def test_non_floatlike_types_are_rejected(self, value):
        """Strings, lists, None, NumPy arrays, and complex numbers are not valid gate parameters."""
        assert not isinstance(value, FloatLike)
