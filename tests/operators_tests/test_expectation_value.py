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

"""Tests the values of QubitOperator.expectation_value and the reuse of the returned function."""

import random

import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable, cx, h, jaspify, x
from qrisp.default_backend import QrispSimulatorBackend
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
            # With atol=0.002, the tolerance is at least 12 worst-case standard deviations of the
            # difference for every operator and state in this test.
            @jaspify(terminal_sampling=True)
            def main():
                return H.expectation_value(state_prep, atol=0.002)()

            assert abs(main() - H.expectation_value(state_prep, atol=0.002, backend=non_sampling_backend)()) < 1e-1

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


def test_expectation_value_of_identity_operator():
    def state_prep():
        return QuantumVariable(1)

    H = 3 + 0 * X(0)

    assert H.expectation_value(state_prep)() == pytest.approx(3)

    @jaspify
    def main():
        return H.expectation_value(state_prep)()

    assert main() == pytest.approx(3)


def _evaluate(H, state_prep, mode, **kwargs):
    """Evaluate an expectation value statically, in Jasp, or in Jasp with terminal sampling."""
    if mode == "static":
        return H.expectation_value(state_prep, **kwargs)()

    @jaspify(terminal_sampling=mode == "jasp_terminal")
    def main():
        return H.expectation_value(state_prep, **kwargs)()

    return float(main())


EVALUATION_MODES = ["static", "jasp", "jasp_terminal"]


def _quantum_float_with_exponent():
    qf = QuantumFloat(2, -1)  # Decodes the raw value 1 as 0.5.
    x(qf[0])
    return qf


def _signed_quantum_float():
    qf = QuantumFloat(3, -2, signed=True)
    x(qf[1])
    x(qf[3])  # Sign qubit
    return qf


@pytest.mark.parametrize("mode", EVALUATION_MODES)
@pytest.mark.parametrize(
    ("state_prep", "H", "expected"),
    [
        (_quantum_float_with_exponent, Z(0) - Z(1), -2),
        (_signed_quantum_float, Z(0) + 2 * Z(1) + 3 * Z(3), -4),
    ],
)
def test_expectation_value_measures_raw_bits_of_typed_variables(state_prep, H, expected, mode):
    assert _evaluate(H, state_prep, mode) == pytest.approx(expected)


def _wide_state():
    qv = QuantumVariable(130)
    x(qv[0])
    x(qv[60])
    x(qv[129])
    h(qv[70])  # X(70) = 1
    x(qv[100])
    h(qv[100])  # X(100) = -1
    # (|q1=0, q120=1> + |q1=1, q120=0>)/sqrt(2)
    x(qv[120])
    h(qv[1])
    cx(qv[1], qv[120])
    # (|q2=0, q110=0> + |q2=1, q110=1>)/sqrt(2)
    h(qv[2])
    cx(qv[2], qv[110])
    return qv


# The state is an eigenstate of every term, so every estimate is exact. The terms act on
# qubits in all three chunks of 53 qubits and on pairs of qubits in different chunks.
WIDE_TERMS = [
    (Z(0) * Z(129), 1),
    (2 * Z(60), -2),
    (3 * X(70) * X(100), -3),
    (4 * P1(0) * P1(129), 4),
    (5 * Z(5) * X(70), 5),
    (A(1) * C(120), 0.5),
    (2 * P0(30), 2),
    (X(2) * X(110) + Z(2) * Z(110), 2),
    (Y(2) * Y(110), -1),
]


@pytest.mark.parametrize("mode", EVALUATION_MODES)
@pytest.mark.parametrize("diagonalization_method", ["commuting_qw", "commuting"])
def test_expectation_value_of_operators_on_more_than_64_qubits(mode, diagonalization_method):
    H = sum(term for term, _ in WIDE_TERMS)
    expected = sum(value for _, value in WIDE_TERMS)

    result = _evaluate(H, _wide_state, mode, rtol=0.05, diagonalization_method=diagonalization_method)

    assert result == pytest.approx(expected)


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
    ev_function = H.expectation_value(state_prep, atol=0.05, precompiled_qc=qc)
    assert ev_function(subs_dic={theta: 0}) == pytest.approx(1)
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
