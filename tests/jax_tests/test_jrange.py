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

"""Tests for jrange, JRangeIterator, make_tracer and jlen."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import Tracer

from qrisp import QuantumFloat, cx, invert, measure, x
from qrisp.jasp import boolean_simulation, jlen, jrange, make_jaspr, make_tracer, qache
from qrisp.jasp.program_control.jrange_iterator import JRANGE_MARKER_NAME, JRangeIterator

ARGUMENT_COUNT_MESSAGE = r"jrange takes 1 or 2 arguments \(\d given\)\. The step argument"


def flip_range(make_args):
    """Return a function that flips the qubits of QuantumFloat(n) in jrange(*make_args(qf))."""

    def main(n):
        qf = QuantumFloat(n)
        for i in jrange(*make_args(qf)):
            x(qf[i])
        return measure(qf)

    return main


# Outside of Jasp mode


@pytest.mark.parametrize(
    "args, expected",
    [
        ((5,), range(5)),
        ((2, 5), range(2, 5)),
        ((5, 2), range(5, 2)),
        ((np.int64(3),), range(3)),
        ((3.7,), range(3)),
        ((jnp.array(4),), range(4)),
    ],
)
def test_static_mode_returns_a_range(args, expected):
    """Outside of Jasp mode, the arguments are converted to int and a range is returned."""
    result = jrange(*args)
    assert isinstance(result, range)
    assert result == expected


@pytest.mark.parametrize("args", [(), (0, 4, 1)])
def test_wrong_number_of_arguments(args):
    """Only stop or start, stop are accepted, in static mode and in Jasp mode."""
    with pytest.raises(TypeError, match=ARGUMENT_COUNT_MESSAGE):
        jrange(*args)

    def main(n):
        qf = QuantumFloat(n)
        jrange(*args)
        return measure(qf)

    with pytest.raises(TypeError, match=ARGUMENT_COUNT_MESSAGE):
        make_jaspr(main)(3)


# Jasp mode


@pytest.mark.parametrize(
    "make_args, expected",
    [
        (lambda qf: (qf.size,), 0b111),
        (lambda qf: (1, qf.size), 0b110),
        (lambda qf: (2,), 0b011),
        (lambda qf: (1, 2), 0b010),
        (lambda qf: (True,), 0b001),
        (lambda qf: (np.int64(2),), 0b011),
        (lambda qf: (qf.size.astype(jnp.int32),), 0b111),
        (lambda qf: (qf.size, qf.size), 0),
        (lambda qf: (qf.size + 1, qf.size), 0),
    ],
    ids=[
        "traced",
        "traced-start",
        "int",
        "int-start",
        "bool",
        "numpy",
        "int32",
        "empty",
        "start-after-stop",
    ],
)
def test_jasp_bounds(make_args, expected):
    """Traced values, Python integers and booleans and NumPy integers are valid bounds."""
    assert int(boolean_simulation(flip_range(make_args))(3)) == expected


@pytest.mark.parametrize("n", [1, 4, 6])
def test_jasp_number_of_iterations_depends_on_the_arguments(n):
    """One Jaspr loops over QuantumFloats of any size."""
    assert int(boolean_simulation(flip_range(lambda qf: (qf.size,)))(n)) == 2**n - 1


@pytest.mark.parametrize("n, expected", [(4, 0b0101), (5, 0b10101)])
def test_jasp_nested_loops(n, expected):
    """An inner bound can depend on the outer loop index."""

    @boolean_simulation
    def main(n):
        qf = QuantumFloat(n)
        for i in jrange(qf.size):
            # Qubit i is flipped i + 1 times, so only the even qubits end up in |1>.
            for _ in jrange(i + 1):
                x(qf[i])
        return measure(qf)

    assert int(main(n)) == expected


@pytest.mark.parametrize("n, expected", [(2, [0b11, 0b11]), (4, [0b1111, 0b0011])])
def test_jasp_inverted_loop_runs_backwards(n, expected):
    """In an InversionEnvironment, the loop runs from the last index to the first."""

    def cx_chain(qf):
        for i in jrange(qf.size - 1):
            cx(qf[i], qf[i + 1])

    @boolean_simulation
    def main(n):
        forward, backward = QuantumFloat(n), QuantumFloat(n)
        x(forward[0])
        x(backward[0])
        cx_chain(forward)
        with invert():
            cx_chain(backward)
        return measure(forward), measure(backward)

    assert [int(result) for result in main(n)] == expected


def test_jasp_loop_is_traced_twice_and_compiled_into_a_while_loop():
    """Each traced iteration is an environment whose body ends with the marker."""
    main = flip_range(lambda qf: (qf.size,))
    unflattened = make_jaspr(main, flatten_envs=False)(3)
    iterations = [eqn for eqn in unflattened.eqns if eqn.primitive.name == "jasp.q_env"]
    markers = [eqn.params["jaspr"].eqns[-1].params["name"] for eqn in iterations]
    assert markers == [JRANGE_MARKER_NAME, JRANGE_MARKER_NAME]

    primitives = [eqn.primitive.name for eqn in make_jaspr(main)(3).eqns]
    assert "while" in primitives
    assert "jasp.q_env" not in primitives


def test_jasp_external_carry_value_raises():
    """Values computed in the loop cannot be used after it."""

    @qache
    def flip_all(qf):
        last = qf.size
        for i in jrange(qf.size):
            x(qf[i])
            last = i + 1
        return last

    def main(n):
        qf = QuantumFloat(n)
        flip_all(qf)
        return measure(qf)

    with pytest.raises(Exception, match="Found jrange with external carry value"):
        make_jaspr(main)(3)


def test_jasp_changing_iterations_raise():
    """Every iteration must apply the same operations."""

    @qache
    def flip_first_then_zero(qf):
        first = True
        for i in jrange(qf.size):
            x(qf[i] if first else qf[0])
            first = False

    def main(n):
        qf = QuantumFloat(n)
        flip_first_then_zero(qf)
        return measure(qf)

    with pytest.raises(Exception, match="Jax semantics changed during jrange iteration"):
        make_jaspr(main)(3)


@pytest.mark.parametrize(
    "start, stop, expected",
    [(None, jnp.array(2), 0b011), (jnp.array(1), None, 0b110)],
)
def test_jasp_concrete_array_bounds(start, stop, expected):
    """Concrete JAX arrays created outside of the traced function are valid bounds."""

    def make_args(qf):
        if start is None:
            return (stop,)
        return (start, qf.size)

    assert int(boolean_simulation(flip_range(make_args))(3)) == expected


# JRangeIterator


@pytest.mark.parametrize("args, start, stop", [((5,), None, 4), ((2, 5), 2, 4)])
def test_iterator_bounds(args, start, stop):
    """Without a start argument, start is None. The stop index is inclusive."""
    iterator = JRangeIterator(*args)
    assert iterator.start == start
    assert iterator.stop.dtype == jnp.int64
    assert int(iterator.stop) == stop


@pytest.mark.parametrize("args", [(), (0, 4, 1)])
def test_iterator_wrong_number_of_arguments(args):
    """JRangeIterator accepts stop or start, stop."""
    with pytest.raises(ValueError, match="jrange only supports 1 or 2 arguments"):
        JRangeIterator(*args)


def test_iterator_traces_two_iterations():
    """The iterator returns the loop index twice and then keeps raising StopIteration."""
    calls = []

    def main(n):
        qf = QuantumFloat(n)
        iterator = jrange(qf.size)
        assert isinstance(iterator, JRangeIterator)
        assert iter(iterator) is iterator
        for _ in range(2):
            x(qf[next(iterator)])
            calls.append("index")
        for _ in range(2):
            with pytest.raises(StopIteration):
                next(iterator)
            calls.append("stop")
        assert iterator.iteration == len(calls)
        return measure(qf)

    n = 3
    assert int(boolean_simulation(main)(n)) == 2**n - 1
    assert calls == ["index", "index", "stop", "stop"]


# make_tracer


@pytest.mark.parametrize("value, dtype", [(True, jnp.bool_), (3, jnp.int64), (2.5, jnp.float64)])
def test_make_tracer_dtypes(value, dtype):
    """Python scalars become 0-dimensional arrays of the matching 64-bit type."""
    result = make_tracer(value)
    assert result.shape == ()
    assert result.dtype == dtype
    assert result == value


def test_make_tracer_complex():
    """Complex numbers become complex128 arrays."""
    result = make_tracer(1 + 2j)
    assert result.dtype == jnp.complex128
    assert result == 1 + 2j


@pytest.mark.parametrize("value", [np.int64(3), "3", None, jnp.array(3)])
def test_make_tracer_unsupported_types(value):
    """Only Python scalars are accepted."""
    with pytest.raises(TypeError, match="Don't know how to tracerize type"):
        make_tracer(value)


def test_make_tracer_is_a_jit_equation():
    """While tracing, the value is the output of a jit equation instead of a constant."""
    (eqn,) = jax.make_jaxpr(lambda: make_tracer(3))().eqns
    assert eqn.params["name"] == "tracerizer"


# jlen


@pytest.mark.parametrize(
    "make_object, expected",
    [
        (lambda: [1, 2, 3], 3),
        (lambda: [], 0),
        (lambda: QuantumFloat(4), 4),
        (lambda: jnp.zeros((2, 3)), 6),
    ],
)
def test_jlen_static(make_object, expected):
    """Lists have a length, other objects a size."""
    assert jlen(make_object()) == expected


def test_jlen_traced():
    """In Jasp mode, the size of a QuantumVariable is traced, the length of a list is not."""
    lengths = {}

    def main(n):
        qf = QuantumFloat(n)
        qubits = [qf[0], qf[1]]
        lengths["quantum_float"] = jlen(qf)
        lengths["list"] = (jlen(qubits), len(qubits))
        return measure(qf)

    make_jaspr(main)(3)
    assert isinstance(lengths["quantum_float"], Tracer)
    length, expected = lengths["list"]
    assert length == expected
