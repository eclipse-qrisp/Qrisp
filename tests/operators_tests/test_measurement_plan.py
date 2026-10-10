"""Tests for shared QubitOperator measurement planning."""

import numpy as np

from qrisp.operators import X, Y, Z
from qrisp.operators.qubit.measurement_plan import (
    _calculate_shot_counts,
    _calculate_shot_weights,
    _create_measurement_plan,
    _normalize_hamiltonian,
)


def test_measurement_plan_normalizes_and_tracks_static_data():
    hamiltonian = X(0) + 2 * Y(1) + Z(2)

    plan = _create_measurement_plan(hamiltonian)

    assert plan.hamiltonian.terms_dict == _normalize_hamiltonian(hamiltonian).terms_dict
    assert plan.num_qubits == 3
    assert len(plan.groups) == len(plan.measurement_operators) == len(plan.standard_deviations)
    assert len(plan.groups) == len(plan.shot_weights)
    assert all(standard_deviation >= 0 for standard_deviation in plan.standard_deviations)


def test_measurement_plan_shot_weights_are_variance_proportional():
    standard_deviations = [1.0, 2.0, 0.5]

    weights = _calculate_shot_weights(standard_deviations)

    np.testing.assert_allclose(weights, [3.5, 7.0, 1.75])


def test_measurement_plan_allocates_explicit_total_shots():
    assert _calculate_shot_counts([1.0, 2.0], shots=10) == [3, 7]
    assert _calculate_shot_counts([1.0, 2.0], precision=1.0) == [1, 2]


def test_measurement_plan_rejects_shot_budget_overflow():
    try:
        _calculate_shot_counts([1.0, 2.0], shots=10, max_shots=9)
    except ValueError as error:
        assert "exceeds max_shots=9" in str(error)
    else:
        raise AssertionError("Shot budget overflow was accepted")

    try:
        _calculate_shot_counts([1.0, 2.0], precision=1.0, max_shots=2)
    except ValueError as error:
        assert "exceeds max_shots=2" in str(error)
    else:
        raise AssertionError("Precision-based shot budget overflow was accepted")


def test_measurement_plan_rejects_unknown_diagonalization_method():
    try:
        _create_measurement_plan(X(0), diagonalization_method="unknown")
    except ValueError as error:
        assert str(error) == "Unknown diagonalization method: unknown."
    else:
        raise AssertionError("Unknown diagonalization method was accepted")


def test_measurement_plan_supports_empty_hamiltonians():
    plan = _create_measurement_plan(0 * X(0))

    assert plan.hamiltonian.terms_dict == {}
    assert plan.groups == []
    assert plan.measurement_operators == []
    assert plan.shot_weights == []
    assert plan.num_qubits == 0
