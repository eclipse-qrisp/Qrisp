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

"""Tests the argument validation, the deprecations and the Jasp restrictions of QubitOperator.expectation_value."""

import inspect

import numpy as np
import pytest

from qrisp import QuantumArray, QuantumCircuit, QuantumVariable, jaspify
from qrisp.default_backend import QrispSimulatorBackend
from qrisp.misc.exceptions import QrispDeprecationWarning
from qrisp.operators import X, Z


def test_expectation_value_accepts_deprecated_diagonalisation_alias():
    def state_prep():
        return QuantumVariable(1)

    with pytest.warns(QrispDeprecationWarning, match="use diagonalization_method instead"):
        X(0).expectation_value(state_prep, diagonalisation_method="commuting_qw")()


def test_expectation_value_rejects_both_diagonalization_spellings():
    method_kwargs = {
        "diagonalization_method": "commuting_qw",
        "diagonalisation_method": "commuting",
    }

    with pytest.raises(ValueError, match="Specify only one"):
        X(0).expectation_value(lambda: QuantumVariable(1), **method_kwargs)


def test_expectation_value_rejects_precision_combined_with_tolerance():
    with pytest.raises(ValueError, match="Specify only one of precision, atol and rtol"):
        X(0).expectation_value(lambda: QuantumVariable(1), precision=0.1, atol=0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"atol": 0.1, "rtol": 0.1}, "Specify only one of atol and rtol"),
        ({"atol": 0}, "atol must be a finite positive number"),
        ({"atol": -0.1}, "atol must be a finite positive number"),
        ({"atol": float("nan")}, "atol must be a finite positive number"),
        ({"atol": float("inf")}, "atol must be a finite positive number"),
        ({"atol": True}, "atol must be a finite positive number"),
        ({"rtol": "0.1"}, "rtol must be a finite positive number"),
        ({"rtol": 0}, "rtol must be a finite positive number"),
        ({"precision": 0}, "precision must be a finite positive number"),
        ({"diagonalization_method": "unknown"}, "Unknown diagonalization method"),
    ],
)
def test_expectation_value_rejects_invalid_parameters_at_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        X(0).expectation_value(lambda: QuantumVariable(1), **kwargs)


def test_expectation_value_accepts_numpy_tolerances():
    X(0).expectation_value(lambda: QuantumVariable(1), atol=np.float64(0.1))
    X(0).expectation_value(lambda: QuantumVariable(1), rtol=np.int64(1))


def test_expectation_value_options_are_keyword_only():
    parameters = list(inspect.signature(X(0).expectation_value).parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "state_prep",
        "atol",
        "rtol",
        "diagonalization_method",
        "backend",
        "compile",
        "compilation_kwargs",
        "precompiled_qc",
        "precision",
        "diagonalisation_method",
        "subs_dic",
    ]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:])


def test_expectation_value_subs_dic_at_creation_is_deprecated():
    from sympy import Symbol

    from qrisp import ry

    theta = Symbol("theta")

    def state_prep():
        qv = QuantumVariable(1)
        ry(theta, qv)
        return qv

    with pytest.warns(QrispDeprecationWarning, match="pass it to the returned function"):
        ev_function = Z(0).expectation_value(state_prep, atol=0.05, subs_dic={theta: 0})

    assert ev_function() == pytest.approx(1)
    assert ev_function(subs_dic={theta: np.pi}) == pytest.approx(-1)


STATIC_ONLY_ARGUMENTS = [
    ({"backend": QrispSimulatorBackend()}, "backend"),
    ({"compile": False}, "compile=False"),
    ({"compilation_kwargs": {}}, "compilation_kwargs"),
    ({"precompiled_qc": QuantumCircuit(1)}, "precompiled_qc"),
    ({"subs_dic": {}}, "subs_dic"),
]


@pytest.mark.parametrize(("kwargs", "name"), STATIC_ONLY_ARGUMENTS)
def test_expectation_value_rejects_static_only_arguments_in_jasp_mode(kwargs, name):
    def state_prep():
        return QuantumVariable(1)

    @jaspify
    def main():
        return X(0).expectation_value(state_prep, **kwargs)()

    with pytest.raises(ValueError, match=f"{name} cannot be used in Jasp mode"):
        main()


@pytest.mark.filterwarnings("ignore::qrisp.misc.exceptions.QrispDeprecationWarning")
@pytest.mark.parametrize(("kwargs", "name"), STATIC_ONLY_ARGUMENTS)
def test_static_expectation_value_function_rejects_jasp_mode(kwargs, name):
    def state_prep():
        return QuantumVariable(1)

    ev_function = X(0).expectation_value(state_prep, **kwargs)

    @jaspify
    def main():
        return ev_function()

    with pytest.raises(ValueError, match=f"{name} cannot be used in Jasp mode"):
        main()


def test_expectation_value_function_rejects_subs_dic_in_jasp_mode():
    def state_prep():
        return QuantumVariable(1)

    ev_function = X(0).expectation_value(state_prep)

    @jaspify
    def main():
        return ev_function(subs_dic={})

    with pytest.raises(ValueError, match="^subs_dic cannot be used in Jasp mode"):
        main()


def test_expectation_value_accepts_default_compile_in_jasp_mode():
    def state_prep():
        return QuantumVariable(1)

    @jaspify
    def main():
        return Z(0).expectation_value(state_prep, compile=True)()

    assert main() == pytest.approx(1)


def _quantum_array():
    return QuantumArray(QuantumVariable(1), shape=2)


def _list_of_variables():
    return [QuantumVariable(1), QuantumVariable(1)]


@pytest.mark.parametrize("jasp", [False, True])
@pytest.mark.parametrize("state_prep", [_quantum_array, _list_of_variables])
def test_expectation_value_requires_state_prep_to_return_a_quantum_variable(state_prep, jasp):
    ev_function = Z(0).expectation_value(state_prep)
    evaluate = jaspify(ev_function) if jasp else ev_function

    with pytest.raises(TypeError, match="state_prep must return a QuantumVariable"):
        evaluate()
