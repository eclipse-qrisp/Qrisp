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

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from qrisp.core import QuantumVariable
from qrisp.operators.hamiltonian import Hamiltonian


@dataclass
class MeasurementPlan:
    """Static data needed to measure a QubitOperator.

    The plan deliberately contains no circuit or backend state.  It can
    therefore be reused by the normal backend path and by the Jasp tracing
    path, while each path remains responsible for applying its basis changes
    and evaluating samples.

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
        Relative shot weights.  Dividing these by ``precision**2`` gives the
        floating-point shot allocation used by the current measurement code.
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
    num_qubits: int
    diagonalization_method: Literal["commuting", "commuting_qw"]
    change_of_basis_gates: list[Any]

    def allocate_shots(
        self,
        precision: float = 0.01,
        shots: int | None = None,
        max_shots: int | None = None,
    ) -> list[int]:
        """Return per-group shots for the requested precision or total."""

        return _calculate_shot_counts(self.shot_weights, precision, shots, max_shots)


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


def _calculate_shot_counts(
    shot_weights: Sequence[float],
    precision: float = 0.01,
    shots: int | None = None,
    max_shots: int | None = None,
) -> list[int]:
    """Calculate per-group shots and enforce an optional total-shot limit.

    When ``shots`` is omitted, the existing precision-based allocation is
    used: group ``i`` receives ``int(shot_weights[i] / precision**2)`` shots.
    When ``shots`` is supplied, it is the requested total and is distributed
    proportionally using largest-remainder rounding. ``max_shots`` applies to
    both modes and raises before any backend or sampler is called.
    """

    if shots is not None:
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be a positive integer.")
        requested_shots = shots
    else:
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("precision must be a finite positive number.")
        raw_shots = [weight / precision**2 for weight in shot_weights]
        requested_shots = int(np.ceil(sum(raw_shots)))

    if max_shots is not None:
        if not isinstance(max_shots, int) or max_shots < 1:
            raise ValueError("max_shots must be a positive integer or None.")
        if requested_shots > max_shots:
            raise ValueError(
                f"The measurement requires approximately {requested_shots} shots, "
                f"which exceeds max_shots={max_shots}. Increase precision or max_shots, "
                "or provide fewer explicit shots."
            )

    if shots is None:
        return [int(weight / precision**2) for weight in shot_weights]

    if not shot_weights:
        return []

    total_weight = sum(shot_weights)
    if total_weight == 0:
        quotient, remainder = divmod(shots, len(shot_weights))
        return [quotient + int(index < remainder) for index in range(len(shot_weights))]

    # Floor proportional allocations, then distribute leftover shots by remainder.
    exact_counts = [shots * weight / total_weight for weight in shot_weights]
    counts = [int(count) for count in exact_counts]
    remainder = shots - sum(counts)
    for index in sorted(range(len(counts)), key=lambda i: exact_counts[i] - counts[i], reverse=True)[:remainder]:
        counts[index] += 1
    return counts


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
        num_qubits=num_qubits,
        diagonalization_method=diagonalization_method,
        change_of_basis_gates=change_of_basis_gates,
    )
