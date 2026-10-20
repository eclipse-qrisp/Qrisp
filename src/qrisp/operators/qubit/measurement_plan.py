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

"""Static planning shared by backend and Jasp expectation measurements."""

import math
import numbers
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from qrisp.core import QuantumVariable
from qrisp.operators.hamiltonian import Hamiltonian

_DEFAULT_RTOL = 0.01


@dataclass
class MeasurementPlan:
    """Static data needed to measure a QubitOperator.

    The plan is computed once from the Hamiltonian and shared by the backend
    path and the Jasp tracing path. Each path applies the basis changes and
    evaluates the samples itself.

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The hermitized, ladder-reduced and thresholded Hamiltonian.
    groups : list[Hamiltonian]
        Measurement groups after the selected commutativity decomposition.
    measurement_operators : list[Hamiltonian]
        Operators represented by computational-basis samples after each
        group's basis change.
    standard_deviations : list[float]
        Estimated standard deviation for each measurement operator.
    shot_weights : list[float]
        Relative shot weights. Dividing these by the squared target standard
        deviation of the estimate gives the shots of each group.
    coefficient_norm : float
        Sum of the absolute coefficients of the non-identity Pauli terms of the
        hermitian part of ``hamiltonian``. A relative tolerance is measured
        against this value.
    num_qubits : int
        Smallest register size required by the Hamiltonian.
    diagonalization_method : str
        Commutativity strategy used to construct the groups.
    change_of_basis_gates : list[Any]
        Basis-change gates for normal backend execution. Jasp plans leave this
        list empty because gates are applied while tracing state preparation.

    """

    hamiltonian: Hamiltonian
    groups: list[Hamiltonian]
    measurement_operators: list[Hamiltonian]
    standard_deviations: list[float]
    shot_weights: list[float]
    coefficient_norm: float
    num_qubits: int
    diagonalization_method: Literal["commuting", "commuting_qw"]
    change_of_basis_gates: list[Any]

    def allocate_shots(self, atol: float | None = None, rtol: float | None = None) -> list[int]:
        """Return the shots of each group for the given tolerance.

        The target standard deviation of the estimate is ``atol`` if it is
        given, and ``rtol * coefficient_norm`` otherwise.
        """
        target_standard_deviation = atol if atol is not None else rtol * self.coefficient_norm
        return _calculate_shot_counts(self.shot_weights, target_standard_deviation)


def _normalize_hamiltonian(hamiltonian: Hamiltonian) -> Hamiltonian:
    """Return the operator form used by expectation-value measurements."""
    return hamiltonian.hermitize().eliminate_ladder_conjugates().apply_threshold(0)


def _split_ladder_groups(groups: list[Hamiltonian]) -> list[Hamiltonian]:
    """Split groups so each basis-change construction has compatible ladders."""
    return [
        subgroup
        for group in groups
        for subgroup in group.group_up(lambda a, b: a.ladders_agree(b) or not a.ladders_intersect(b))
    ]


def _construct_measurement_groups(
    hamiltonian: Hamiltonian,
    diagonalization_method: Literal["commuting", "commuting_qw"],
) -> list[Hamiltonian]:
    """Construct measurement groups for a normalized Hamiltonian.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        A normalized QubitOperator-like Hamiltonian.
    diagonalization_method : str
        Either ``"commuting_qw"`` or ``"commuting"``.

    Returns
    -------
    list[Hamiltonian]
        Groups that can be measured after one basis change each.

    Raises
    ------
    ValueError
        If ``diagonalization_method`` is unknown.

    """
    if diagonalization_method == "commuting_qw":
        return _split_ladder_groups(hamiltonian.commuting_qw_groups())
    if diagonalization_method == "commuting":
        return _split_ladder_groups(hamiltonian.group_up(lambda a, b: a.commute_pauli(b)))
    raise ValueError(f"Unknown diagonalization method: {diagonalization_method}.")


def _calculate_shot_weights(standard_deviations: Sequence[float]) -> list[float]:
    """Return variance-proportional relative shot weights."""
    total_standard_deviation = sum(standard_deviations)
    return [total_standard_deviation * standard_deviation for standard_deviation in standard_deviations]


def _coefficient_norm(hamiltonian: Hamiltonian) -> float:
    """Return the sum of the absolute non-identity Pauli coefficients of the hermitian part."""
    pauli_terms = hamiltonian.hermitize().to_pauli().terms_dict
    return float(sum(abs(coeff) for term, coeff in pauli_terms.items() if term.factor_dict))


def _check_tolerance(name: str, value: Any) -> None:
    """Raise a ``ValueError`` unless ``value`` is a finite positive number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")


def _resolve_tolerances(atol: float | None, rtol: float | None) -> tuple[float | None, float | None]:
    """Validate ``atol`` and ``rtol`` and fall back to the default relative tolerance.

    Returns
    -------
    tuple
        ``(atol, rtol)`` with exactly one entry set.

    Raises
    ------
    ValueError
        If both tolerances are given or the given one is not a finite positive number.

    """
    if atol is not None and rtol is not None:
        raise ValueError("Specify only one of atol and rtol.")
    if atol is None and rtol is None:
        rtol = _DEFAULT_RTOL

    if atol is not None:
        _check_tolerance("atol", atol)
    if rtol is not None:
        _check_tolerance("rtol", rtol)

    return atol, rtol


def _calculate_shot_counts(shot_weights: Sequence[float], target_standard_deviation: float) -> list[int]:
    """Return the shots of each group for a target standard deviation of the estimate.

    Group ``i`` receives ``ceil(shot_weights[i] / target_standard_deviation**2)``
    shots, but at least one. Rounding up keeps the estimated standard deviation
    at or below the target.
    """
    if target_standard_deviation == 0:
        # Only reached for a relative tolerance on an operator without non-identity
        # terms. Every group then has zero variance and one shot gives the exact value.
        return [1] * len(shot_weights)
    return [max(1, math.ceil(weight / target_standard_deviation**2)) for weight in shot_weights]


def _create_measurement_plan(
    hamiltonian: Hamiltonian,
    diagonalization_method: Literal["commuting", "commuting_qw"] = "commuting_qw",
    construct_basis_gates: bool = True,
) -> MeasurementPlan:
    """Build the backend-independent expectation-value measurement plan.

    The returned plan contains the same grouping, basis-transformed operators,
    variance estimates and relative shot allocation used by both measurement
    implementations.  Circuit construction and sample post-processing are
    intentionally left to those implementations.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Operator to normalize and group.
    diagonalization_method : str, optional
        Grouping strategy, either ``"commuting_qw"`` or ``"commuting"``.
    construct_basis_gates : bool, optional
        Whether to create normal backend basis-change gates. Jasp passes
        ``False`` because its gates are created while tracing state prep.

    """
    normalized_hamiltonian = _normalize_hamiltonian(hamiltonian)
    if len(normalized_hamiltonian.terms_dict) == 0:
        return MeasurementPlan(
            hamiltonian=normalized_hamiltonian,
            groups=[],
            measurement_operators=[],
            standard_deviations=[],
            shot_weights=[],
            coefficient_norm=0.0,
            num_qubits=0,
            diagonalization_method=diagonalization_method,
            change_of_basis_gates=[],
        )

    num_qubits = normalized_hamiltonian.find_minimal_qubit_amount()
    groups = _construct_measurement_groups(normalized_hamiltonian, diagonalization_method)
    measurement_operators = []
    change_of_basis_gates = []
    for group in groups:
        if construct_basis_gates:
            qv = QuantumVariable(num_qubits)
            measurement_operators.append(group.change_of_basis(qv, diagonalization_method))
            change_of_basis_gates.append(qv.qs.to_gate())
        else:
            measurement_operators.append(group.change_of_basis(method=diagonalization_method))

    standard_deviations = [
        float(np.sqrt(measurement_operator.get_operator_variance(n=num_qubits)))
        for measurement_operator in measurement_operators
    ]

    return MeasurementPlan(
        hamiltonian=normalized_hamiltonian,
        groups=groups,
        measurement_operators=measurement_operators,
        standard_deviations=standard_deviations,
        shot_weights=_calculate_shot_weights(standard_deviations),
        coefficient_norm=_coefficient_norm(normalized_hamiltonian),
        num_qubits=num_qubits,
        diagonalization_method=diagonalization_method,
        change_of_basis_gates=change_of_basis_gates,
    )
