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

"""Tests expectation_value for qubit operators across backends and diagonalisation methods."""

import inspect
import random

import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable, cx, h, jaspify, x
from qrisp.default_backend import QrispSimulatorBackend
from qrisp.misc.exceptions import QrispDeprecationWarning
from qrisp.operators import P0, P1, A, C, X, Y, Z


def test_expectation_value(sample_size=100, seed=42, exhaustive=False):

    non_sampling_backend = QrispSimulatorBackend()

    def testing_helper(state_prep, operator_combinations):
        for H in operator_combinations:
            if isinstance(H, int):
                continue

            print(H)
            assert (
                abs(
                    H.expectation_value(state_prep, atol=0.0005, backend=non_sampling_backend)()
                    - H.to_pauli().expectation_value(state_prep, atol=0.0005, backend=non_sampling_backend)()
                )
                < 1e-1
            )
            assert (
                abs(
                    H.expectation_value(
                        state_prep,
                        atol=0.0005,
                        diagonalization_method="commuting",
                        backend=non_sampling_backend,
                    )()
                    - H.to_pauli().expectation_value(
                        state_prep,
                        atol=0.0005,
                        diagonalization_method="commuting",
                        backend=non_sampling_backend,
                    )()
                )
                < 1e-1
            )

            # Jasp tests
            @jaspify(terminal_sampling=True)
            def main():
                return H.expectation_value(state_prep, atol=0.01)()

            assert abs(main() - H.expectation_value(state_prep, atol=0.01, backend=non_sampling_backend)()) < 1e-1

    # Set the random seed for reproducibility
    random.seed(seed)

    # Define the full list of operators
    operator_list = [lambda x: 1, X, Y, Z, A, C, P0, P1]

    # Generate all possible combinations of operators
    all_combinations = []

    if exhaustive:
        for op1 in operator_list:
            for op2 in operator_list:
                for op3 in operator_list:
                    for op4 in operator_list:
                        H = op1(0) * op2(1) * op3(2) * op4(3)

                        if H == 1:
                            continue

                        all_combinations.append(H)
    else:
        for _ in range(sample_size):
            combination = [random.choice(operator_list) for _ in range(4)]  # Choose 4 operators
            H = combination[0](0) * combination[1](1) * combination[2](2) * combination[3](3)
            all_combinations.append(H)

    def state_prep():
        qv = QuantumFloat(4)
        return qv

    # Perform tests with the randomly generated operator combinations
    testing_helper(state_prep, all_combinations)

    def state_prep():
        qv = QuantumFloat(4)
        h(qv[0])
        return qv

    testing_helper(state_prep, all_combinations)

    def state_prep():
        qv = QuantumFloat(4)
        h(qv[0])
        cx(qv[0], qv[1])
        return qv

    testing_helper(state_prep, all_combinations)

    def state_prep():
        qv = QuantumFloat(4)
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[0], qv[2])
        return qv

    testing_helper(state_prep, all_combinations)


def test_expectation_value_issue_165():
    """Test the expectation value method for the specific case of issue #165."""

    def state_prep():
        qv = QuantumVariable(4)
        x(qv[0])
        x(qv[1])
        return qv

    H = A(0) * C(1) * C(2) * A(3) + P1(0) * P1(2) + P1(1) * P1(3)

    assert H.expectation_value(state_prep, diagonalization_method="commuting")() == 0
    assert H.expectation_value(state_prep, diagonalization_method="commuting_qw")() == 0


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


class _ShotRecordingBackend(QrispSimulatorBackend):
    """Simulator backend that records the shots of every run."""

    def __init__(self):
        super().__init__()
        self.shots = []

    def run(self, circuits, shots=None):
        self.shots.append(shots)
        return super().run(circuits, shots=shots)


def _recorded_shots(H, state_prep, **kwargs):
    backend = _ShotRecordingBackend()
    H.expectation_value(state_prep, backend=backend, **kwargs)()
    return backend.shots


def _heisenberg_chain(n):
    return sum(X(i) * X(i + 1) + Y(i) * Y(i + 1) + Z(i) * Z(i + 1) for i in range(n - 1))


def test_expectation_value_default_rtol_is_independent_of_operator_scale_and_offset():
    def state_prep():
        return QuantumVariable(4)

    H = _heisenberg_chain(4)

    shots = _recorded_shots(H, state_prep)

    assert _recorded_shots(100 * H, state_prep) == shots
    assert _recorded_shots(H + 50, state_prep) == shots
    assert _recorded_shots(H, state_prep, rtol=0.01) == shots
    assert sum(shots) <= 1 / 0.01**2 + len(shots)


def test_expectation_value_atol_scales_shots_with_operator():
    def state_prep():
        return QuantumVariable(2)

    H = X(0) * X(1) + Z(0)

    shots = _recorded_shots(H, state_prep, atol=0.1)
    scaled_shots = _recorded_shots(10 * H, state_prep, atol=0.1)

    assert sum(scaled_shots) == pytest.approx(100 * sum(shots), rel=0.01)


def test_expectation_value_rtol_is_relative_to_pauli_coefficient_norm():
    def state_prep():
        return QuantumVariable(2)

    # The sum of the absolute non-identity Pauli coefficients is 4.
    H = X(0) * X(1) - 3 * Z(0) + 7

    assert _recorded_shots(H, state_prep, rtol=0.05) == _recorded_shots(H, state_prep, atol=0.2)


def test_expectation_value_precision_is_deprecated_alias_for_atol():
    def state_prep():
        return QuantumVariable(2)

    H = X(0) * X(1) + Z(0)

    with pytest.warns(QrispDeprecationWarning, match="precision is deprecated; use atol"):
        shots = _recorded_shots(H, state_prep, precision=0.1)

    assert shots == _recorded_shots(H, state_prep, atol=0.1)


def test_expectation_value_measures_every_group_at_least_once():
    def state_prep():
        qv = QuantumVariable(1)
        x(qv)
        return qv

    # The Z group has a negligible variance, which used to round to zero shots.
    H = X(0) + 1e-6 * Z(0)

    assert min(_recorded_shots(H, state_prep, rtol=0.1)) == 1

    @jaspify
    def main():
        return H.expectation_value(state_prep, rtol=0.1)()

    assert abs(main()) < 0.5


def test_expectation_value_of_identity_operator():
    def state_prep():
        return QuantumVariable(1)

    H = 3 + 0 * X(0)

    assert H.expectation_value(state_prep)() == pytest.approx(3)

    @jaspify
    def main():
        return H.expectation_value(state_prep)()

    assert main() == pytest.approx(3)


@pytest.mark.parametrize("tolerance", [{"atol": 0.05}, {"rtol": 0.05}, {}])
def test_jasp_expectation_value_with_tolerances(tolerance):
    def state_prep():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return qv

    # Exact value on the Bell state: 1 + 1 - 1 = 1.
    H = X(0) * X(1) + Z(0) * Z(1) + Y(0) * Y(1)

    @jaspify
    def main():
        return H.expectation_value(state_prep, **tolerance)()

    assert abs(main() - 1) < 0.5


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
        "subs_dic",
        "precompiled_qc",
        "precision",
        "diagonalisation_method",
    ]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:])


def test_expectation_value_builds_measurement_plan_once(monkeypatch):
    import qrisp.operators.qubit.qubit_operator as qubit_operator_module

    calls = []
    create_plan = qubit_operator_module._create_measurement_plan

    def counting_create_plan(*args, **kwargs):
        calls.append(None)
        return create_plan(*args, **kwargs)

    monkeypatch.setattr(qubit_operator_module, "_create_measurement_plan", counting_create_plan)

    ev_function = (X(0) * X(1) + Z(0)).expectation_value(lambda: QuantumVariable(2))
    for _ in range(3):
        ev_function()

    @jaspify
    def main():
        return ev_function()

    main()

    assert len(calls) == 1


def test_expectation_value_measures_operator_as_it_was_when_created():
    def state_prep():
        return QuantumVariable(1)

    H = Z(0)
    ev_function = H.expectation_value(state_prep)
    H += 2 * Z(0)

    assert ev_function() == pytest.approx(1)
    assert H.expectation_value(state_prep)() == pytest.approx(3)


def test_expectation_value_accepts_subs_dic_per_evaluation():
    from sympy import Symbol

    from qrisp import ry

    theta = Symbol("theta")

    def state_prep():
        qv = QuantumVariable(1)
        ry(theta, qv)
        return qv

    H = Z(0)

    ev_function = H.expectation_value(state_prep, atol=0.05)
    assert ev_function(subs_dic={theta: 0}) == pytest.approx(1)
    assert ev_function(subs_dic={theta: np.pi}) == pytest.approx(-1)

    qc = state_prep().qs.compile()
    ev_function = H.expectation_value(state_prep, atol=0.05, precompiled_qc=qc, subs_dic={theta: 0})
    assert ev_function() == pytest.approx(1)
    assert ev_function(subs_dic={theta: np.pi}) == pytest.approx(-1)


def test_expectation_value_function_works_in_static_and_jasp_mode():
    def state_prep():
        qv = QuantumVariable(2)
        x(qv[0])
        return qv

    # Exact value: -1 - 1 = -2.
    ev_function = (Z(0) - Z(1)).expectation_value(state_prep, diagonalization_method="commuting")

    @jaspify
    def main():
        return ev_function()

    assert ev_function() == pytest.approx(-2)
    assert main() == pytest.approx(-2)


def test_expectation_value_batched_backend():
    """Test the expectation value method with the BatchedBackend."""

    def state_prep():
        d = QuantumFloat(4)
        e = QuantumFloat(3)
        d[:] = 2
        e[:] = 2
        f = d + e
        return f

    H = Z(0) * Z(1) * Z(2) * Z(3) + X(0) * X(1) * X(2) * X(3)

    bb = QrispSimulatorBackend().batched()

    ev = H.expectation_value(state_prep, backend=bb)()
