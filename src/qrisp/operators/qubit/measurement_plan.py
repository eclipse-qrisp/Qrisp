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

"""Static planning shared by the backend and the Jasp expectation-value measurements.

An expectation value is estimated by splitting the operator into groups of terms that can
be measured simultaneously, measuring each group in its own circuits, and summing the
estimates of the groups. Everything that depends only on the operator is computed once,
when ``QubitOperator.expectation_value`` is called, and stored in a ``MeasurementPlan``:

1. Normalization: the operator is hermitized and zero terms are removed. A ladder term and
   its adjoint are merged into one term, which stands for both.
2. Grouping: the terms are split into groups whose terms commute qubit-wise
   (``"commuting_qw"``) or commute (``"commuting"``). Groups whose ladder operators act on
   overlapping but different qubits are split further.
3. Change of basis: for each group, a circuit is constructed that diagonalizes all of its
   terms at once. Conjugated with this circuit, the group becomes a diagonal measurement
   operator whose terms consist of Z, P0 and P1 factors, so their values can be read off
   a computational-basis outcome.
4. Serialization: each diagonal term is stored as a real coefficient and three bit masks
   over the measured qubits (see ``QubitTerm.serialize``): the Z mask z, the AND mask a and
   the control mask c. For an outcome x, the term has the value
   (-1)**popcount(z & x) if (x ^ c) & a == 0, and 0 otherwise. Terms with ladder operators
   take the values +-1/2, so their coefficients are halved.
5. Chunks: the masks, and later the outcomes, are integers with one bit per qubit. To
   handle any number of qubits with 64-bit integers, they are split into chunks of
   ``_CHUNK_SIZE`` = 53 qubits, lowest qubits first: chunk j holds the qubits 53 j to
   53 j + 52. 53 bits keep a chunk exact even when Jasp's ``sample`` returns it as a
   float64. A term is evaluated on all chunks of an outcome together: the sign uses the
   popcount summed over the chunks, and the AND condition must hold in every chunk. This
   gives the same value as evaluating the term on the unsplit integers.
6. Shot allocation: group i receives ceil(sigma_i * sum_j sigma_j / epsilon**2) shots, and
   at least one, where sigma_i is the standard deviation of the group, averaged over all
   states, and epsilon is the targeted standard deviation of the estimate. epsilon is
   ``atol``, or ``rtol`` times the sum of the absolute non-identity Pauli coefficients.
"""

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from qrisp.core import QuantumVariable
from qrisp.operators.hamiltonian import Hamiltonian

_DEFAULT_RTOL = 0.01

# Measurement outcomes and the bit masks of the observables are split into chunks
# of this many qubits, so that registers of any size can be evaluated with 64-bit
# integers. 53 bits keep every chunk exact when Jasp's sample stores its results
# as float64.
_CHUNK_SIZE = 53


@dataclass
class MeasurementPlan:
    r"""Static data needed to measure a QubitOperator.

    The plan depends only on the operator and the diagonalization method, not
    on the measured state. It is built once per expectation-value function and
    shared by the backend path and the Jasp tracing path.

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The hermitized, ladder-reduced and thresholded Hamiltonian.
    groups : list[Hamiltonian]
        Measurement groups after the selected commutativity decomposition.
    measurement_operators : list[Hamiltonian]
        Operators represented by computational-basis samples after each
        group's basis change.
    observables : numpy.ndarray
        Bit masks of the terms of each measurement operator, see
        ``QubitTerm.serialize``, with shape ``(groups, terms, 3, chunks)``. The
        third axis holds the Z, AND and control masks, the last axis splits
        them into chunks of ``_CHUNK_SIZE`` qubits. Groups with fewer terms
        are padded with zeros.
    coefficients : numpy.ndarray
        Real coefficients matching ``observables``, with shape
        ``(groups, terms)``. Terms with ladder operators are halved, because
        their measured value is :math:`\pm 1/2`. Padding entries are zero, so
        they do not contribute to the expectation value.
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

    """

    hamiltonian: Hamiltonian
    groups: list[Hamiltonian]
    measurement_operators: list[Hamiltonian]
    observables: np.ndarray
    coefficients: np.ndarray
    standard_deviations: list[float]
    shot_weights: list[float]
    coefficient_norm: float
    num_qubits: int
    diagonalization_method: Literal["commuting", "commuting_qw"]
    _change_of_basis_gates: list[Any] | None = field(default=None, init=False, repr=False)

    def allocate_shots(self, atol: float | None = None, rtol: float | None = None) -> list[int]:
        """Return the shots of each group for the given tolerance.

        The target standard deviation of the estimate is ``atol`` if it is
        given, and ``rtol * coefficient_norm`` otherwise.
        """
        target_standard_deviation = atol if atol is not None else rtol * self.coefficient_norm
        return _calculate_shot_counts(self.shot_weights, target_standard_deviation)

    def change_of_basis_gates(self) -> list[Any]:
        """Return the basis-change gate of each group for backend execution.

        The gates are built on first use, which must happen outside of tracing
        mode. The Jasp path applies the basis changes while tracing instead.
        """
        if self._change_of_basis_gates is None:
            gates = []
            for group in self.groups:
                qv = QuantumVariable(self.num_qubits)
                group.change_of_basis(qv, self.diagonalization_method)
                gates.append(qv.qs.to_gate())
            self._change_of_basis_gates = gates
        return self._change_of_basis_gates


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


def _num_chunks(num_qubits: int) -> int:
    """Return the number of chunks of ``_CHUNK_SIZE`` qubits needed for ``num_qubits`` qubits."""
    return max(1, math.ceil(num_qubits / _CHUNK_SIZE))


def _split_bits(value: int, num_chunks: int) -> list[int]:
    """Split a non-negative integer into ``num_chunks`` chunks of ``_CHUNK_SIZE`` bits, lowest bits first."""
    mask = (1 << _CHUNK_SIZE) - 1
    return [(value >> (_CHUNK_SIZE * chunk)) & mask for chunk in range(num_chunks)]


def _serialize_measurement_operators(
    measurement_operators: list[Hamiltonian], num_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the zero-padded bit masks and real coefficients of the measurement operators."""
    num_terms = max((len(operator.terms_dict) for operator in measurement_operators), default=0)
    num_chunks = _num_chunks(num_qubits)
    observables = np.zeros((len(measurement_operators), num_terms, 3, num_chunks), dtype=np.int64)
    coefficients = np.zeros((len(measurement_operators), num_terms), dtype=np.float64)
    for i, operator in enumerate(measurement_operators):
        for j, (term, coeff) in enumerate(operator.terms_dict.items()):
            z_mask, and_mask, ctrl_mask, contains_ladder = term.serialize()
            for k, mask in enumerate((z_mask, and_mask, ctrl_mask)):
                observables[i, j, k] = _split_bits(mask, num_chunks)
            coefficients[i, j] = np.real(coeff) * (0.5 if contains_ladder else 1)
    return observables, coefficients


def _create_measurement_plan(
    hamiltonian: Hamiltonian,
    diagonalization_method: Literal["commuting", "commuting_qw"] = "commuting_qw",
) -> MeasurementPlan:
    """Build the measurement plan of a Hamiltonian.

    The plan contains the grouping, the basis-transformed operators in
    serialized form, the variance estimates and the relative shot allocation
    used by both measurement paths. It can be built in static and in tracing
    mode.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Operator to normalize and group.
    diagonalization_method : str, optional
        Grouping strategy, either ``"commuting_qw"`` or ``"commuting"``.

    """
    normalized_hamiltonian = _normalize_hamiltonian(hamiltonian)
    if len(normalized_hamiltonian.terms_dict) == 0:
        groups = []
        num_qubits = 0
    else:
        groups = _construct_measurement_groups(normalized_hamiltonian, diagonalization_method)
        num_qubits = normalized_hamiltonian.find_minimal_qubit_amount()

    measurement_operators = [group.change_of_basis(method=diagonalization_method) for group in groups]
    observables, coefficients = _serialize_measurement_operators(measurement_operators, num_qubits)
    standard_deviations = [
        float(np.sqrt(measurement_operator.get_operator_variance(n=num_qubits)))
        for measurement_operator in measurement_operators
    ]

    return MeasurementPlan(
        hamiltonian=normalized_hamiltonian,
        groups=groups,
        measurement_operators=measurement_operators,
        observables=observables,
        coefficients=coefficients,
        standard_deviations=standard_deviations,
        shot_weights=_calculate_shot_weights(standard_deviations),
        coefficient_norm=_coefficient_norm(normalized_hamiltonian),
        num_qubits=num_qubits,
        diagonalization_method=diagonalization_method,
    )
