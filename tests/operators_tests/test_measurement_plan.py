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

"""Tests for shared QubitOperator measurement planning."""

import numpy as np
import pytest

from qrisp.operators import A, C, X, Y, Z
from qrisp.operators.qubit.measurement_plan import (
    _calculate_shot_counts,
    _calculate_shot_weights,
    _create_measurement_plan,
    _normalize_hamiltonian,
    _resolve_tolerances,
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


def test_shot_counts_round_up_and_measure_every_group():
    assert _calculate_shot_counts([1.0, 2.0], 1.0) == [1, 2]
    assert _calculate_shot_counts([0.5, 2.2], 1.0) == [1, 3]
    assert _calculate_shot_counts([0.0, 1e-12], 0.1) == [1, 1]
    assert _calculate_shot_counts([0.0, 0.0], 0.0) == [1, 1]


def test_coefficient_norm_ignores_identity_and_uses_pauli_form():
    assert _create_measurement_plan(X(0) - 2 * Y(1) + 0.5 * Z(2) + 7).coefficient_norm == pytest.approx(3.5)
    # The hermitian part of A(0)*C(1) is (X(0)*X(1) + Y(0)*Y(1))/4.
    assert _create_measurement_plan(A(0) * C(1) + 3).coefficient_norm == pytest.approx(0.5)


def test_allocate_shots_measures_rtol_against_coefficient_norm():
    plan = _create_measurement_plan(X(0) * X(1) + 3 * Z(0) + 7)

    assert plan.allocate_shots(rtol=0.05) == plan.allocate_shots(atol=0.2)


def test_resolve_tolerances():
    assert _resolve_tolerances(None, None) == (None, 0.01)
    assert _resolve_tolerances(0.1, None) == (0.1, None)
    assert _resolve_tolerances(None, 0.1) == (None, 0.1)

    with pytest.raises(ValueError, match="Specify only one of atol and rtol"):
        _resolve_tolerances(0.1, 0.1)
    with pytest.raises(ValueError, match="atol must be a finite positive number"):
        _resolve_tolerances(-1, None)


def test_measurement_plan_rejects_unknown_diagonalization_method():
    with pytest.raises(ValueError, match="Unknown diagonalization method: unknown."):
        _create_measurement_plan(X(0), diagonalization_method="unknown")


def test_measurement_plan_serializes_measurement_operators():
    plan = _create_measurement_plan(X(0) * X(1) + 2 * Z(0) + 3 * Z(1) * Z(2), "commuting")

    num_groups, num_terms, _ = plan.observables.shape
    assert num_groups == len(plan.groups)
    assert plan.coefficients.shape == (num_groups, num_terms)
    for observables, coefficients, operator in zip(plan.observables, plan.coefficients, plan.measurement_operators):
        size = len(operator.terms_dict)
        assert [tuple(row) for row in observables[:size]] == [term.serialize() for term in operator.terms_dict]
        np.testing.assert_allclose(coefficients[:size], np.real(list(operator.terms_dict.values())))
        assert not observables[size:].any()
        assert not coefficients[size:].any()


def test_measurement_plan_builds_change_of_basis_gates_on_first_use():
    plan = _create_measurement_plan(X(0) * X(1) + Z(0), "commuting")

    assert plan._change_of_basis_gates is None
    gates = plan.change_of_basis_gates()
    assert len(gates) == len(plan.groups)
    assert plan.change_of_basis_gates() is gates


def test_measurement_plan_supports_empty_hamiltonians():
    plan = _create_measurement_plan(0 * X(0))

    assert plan.hamiltonian.terms_dict == {}
    assert plan.groups == []
    assert plan.measurement_operators == []
    assert plan.shot_weights == []
    assert plan.observables.shape == (0, 0, 4)
    assert plan.coefficient_norm == 0
    assert plan.num_qubits == 0
