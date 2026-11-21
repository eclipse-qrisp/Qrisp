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

"""Tests that the measurement of QubitOperator.expectation_value creates no arrays larger than the measured data."""

import math

import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr

from qrisp import QuantumVariable
from qrisp.jasp import make_jaspr
from qrisp.operators import X, Y, Z
from qrisp.operators.qubit.measurement import _outcome_arrays
from qrisp.operators.qubit.measurement_plan import _create_measurement_plan


def _largest_array(jaxpr):
    """Return the largest number of elements of an array created anywhere in a jaxpr, including nested jaxprs."""
    largest = 0
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            largest = max(largest, math.prod(getattr(var.aval, "shape", ())))
        for param in eqn.params.values():
            for value in param if isinstance(param, (list, tuple)) else (param,):
                if isinstance(value, ClosedJaxpr):
                    largest = max(largest, _largest_array(value.jaxpr))
                elif isinstance(value, Jaxpr):
                    largest = max(largest, _largest_array(value))
    return largest


def test_jasp_expectation_value_creates_no_array_larger_than_one_group():
    """The traced Jasp measurement creates no array larger than the samples of one group."""
    # Three groups with 200 shots each.
    H = X(0) + Y(0) + Z(0)
    shots = _create_measurement_plan(H).allocate_shots(atol=0.1)
    assert len(shots) == 3

    def state_prep():
        return QuantumVariable(1)

    def main():
        return H.expectation_value(state_prep, atol=0.1)()

    jaspr = make_jaspr(main)()

    # Each group is evaluated on its own samples, so no array covers the samples of several groups.
    assert _largest_array(jaspr) <= max(shots)


def test_static_outcomes_are_stored_without_padding():
    """The static outcomes of all groups are stored one after the other, without padding to the largest group."""
    results = [{0: 0.5, 1: 0.5}, {outcome: 0.01 for outcome in range(100)}, {7: 1.0}]

    outcomes, probabilities, offsets = _outcome_arrays(results, 1)

    assert outcomes.shape == (103, 1)
    assert probabilities.shape == (103,)
    assert offsets.tolist() == [0, 2, 102, 103]
    assert outcomes[102, 0] == 7
    np.testing.assert_allclose(probabilities[2:102], 0.01)


def test_static_outcomes_are_split_into_chunks():
    """Outcomes wider than one chunk are split into chunks, lowest qubits first."""
    outcome = (1 << 100) | (1 << 53) | 5
    results = [{outcome: 1.0}, {3: 0.25, 1 << 60: 0.75}]

    outcomes, probabilities, offsets = _outcome_arrays(results, 2)

    assert outcomes.tolist() == [[5, 1 | (1 << 47)], [3, 0], [0, 1 << 7]]
    assert probabilities.tolist() == [1.0, 0.25, 0.75]
    assert offsets.tolist() == [0, 1, 3]
