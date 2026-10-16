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

"""Tests for measure in Jasp mode."""

import jax
import jax.numpy as jnp
import pytest

from qrisp import QuantumArray, QuantumBool, QuantumFloat, QuantumVariable, measure, x
from qrisp.jasp import boolean_simulation, make_jaspr

UNSUPPORTED_MESSAGE = "Tried to measure type"


def test_jasp_return_types():
    """Qubits give booleans, qubit arrays integers, and QuantumVariables their decoded value."""
    avals = {}

    def main():
        qf = QuantumFloat(3)
        avals["qubit"] = measure(qf[0]).aval
        avals["quantum_bool"] = measure(QuantumBool()).aval
        avals["quantum_float"] = measure(qf).aval
        avals["quantum_variable"] = measure(QuantumVariable(2)).aval
        avals["slice"] = measure(qf[0:2]).aval
        reg = qf.reg
        assert reg is not None
        avals["qubit_array"] = measure(reg.tracer).aval
        avals["quantum_array"] = measure(QuantumArray(QuantumBool(), shape=2)).aval
        return measure(qf[1])

    make_jaspr(main)()
    dtypes = {name: (str(aval.dtype), aval.shape) for name, aval in avals.items()}
    assert dtypes == {
        "qubit": ("bool", ()),
        "quantum_bool": ("bool", ()),
        "quantum_float": ("float64", ()),
        "quantum_variable": ("int64", ()),
        "slice": ("int64", ()),
        "qubit_array": ("int64", ()),
        "quantum_array": ("bool", (2,)),
    }


def test_boolean_simulation_measures_a_qubit_as_a_boolean():
    """Under boolean_simulation, a measured qubit is a boolean, so ~ negates it."""

    @boolean_simulation
    def main(n):
        qf = QuantumFloat(1)
        qf[:] = n
        result = measure(qf[0])
        return result, ~result

    for value in (False, True):
        result, negated = main(int(value))
        assert result.dtype == jnp.bool_
        assert (bool(result), bool(negated)) == (value, not value)


def test_jasp_measurement_values():
    """The traced outcomes match the encoded state."""

    @boolean_simulation
    def main(n):
        qf = QuantumFloat(3)
        qf[:] = n
        flag = QuantumBool()
        x(flag)
        flags = QuantumArray(QuantumBool(), shape=2)
        x(flags[1])
        return measure(qf), measure(qf[0]), measure(qf[1:3]), measure(flag), measure(flags)

    for value in (0, 5, 6):
        number, first_bit, upper_bits, flag, flags = main(value)
        assert float(number) == value
        assert bool(first_bit) == bool(value & 1)
        assert int(upper_bits) == value >> 1
        assert bool(flag)
        assert [bool(b) for b in flags] == [False, True]


@pytest.mark.parametrize(
    "make_input",
    [
        lambda qf: [qf[0], qf[1]],
        lambda qf: 1,
        lambda qf: jnp.array(1.0) + measure(qf),
    ],
)
def test_jasp_unsupported_input_raises_type_error(make_input):
    """Lists, Python numbers and classical tracers cannot be measured in Jasp mode."""

    def main():
        qf = QuantumFloat(2)
        return measure(make_input(qf))

    with pytest.raises(TypeError, match=UNSUPPORTED_MESSAGE):
        make_jaspr(main)()


def test_jasp_lost_track_raises_runtime_error():
    """Measuring inside a JAX control-flow primitive loses track of the quantum state."""

    def main():
        qb = QuantumBool()

        def body(_i, total):
            return total + measure(qb[0])

        return jax.lax.fori_loop(0, 2, body, 0)

    with pytest.raises(RuntimeError, match="Lost track of QuantumCircuit during tracing"):
        make_jaspr(main)()
