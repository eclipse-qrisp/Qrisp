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

"""Tests the tolerances and the shot allocation of QubitOperator.expectation_value."""

import pytest

from qrisp import QuantumVariable, cx, h, jaspify, x
from qrisp.default_backend import QrispSimulatorBackend
from qrisp.misc.exceptions import QrispDeprecationWarning
from qrisp.operators import X, Y, Z


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
