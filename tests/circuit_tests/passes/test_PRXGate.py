"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import numpy as np

from qrisp import U3Gate
from qrisp.circuit import PRXGate


class TestPRXGateConstruction:
    """Tests for PRXGate parameter mapping and U3 inheritance."""

    def test_u3_parameter_mapping(self):
        """PRX(alpha, beta) maps to U3(alpha, beta - pi/2, pi/2 - beta)."""
        alpha, beta = np.pi / 2, np.pi / 4
        prx = PRXGate(alpha, beta)

        assert prx.alpha == alpha
        assert prx.beta == beta
        assert abs(prx.theta - alpha) < 1e-10
        assert abs(prx.phi - (beta - np.pi / 2)) < 1e-10
        assert abs(prx.lam - (np.pi / 2 - beta)) < 1e-10

    def test_is_subclass_of_U3Gate(self):
        """PRXGate must inherit from U3Gate."""
        prx = PRXGate(0.5, 0.3)
        assert isinstance(prx, U3Gate)
        assert issubclass(PRXGate, U3Gate)

    def test_name_is_prx(self):
        assert PRXGate(0.1, 0.2).name == "prx"

    def test_u3_equivalence(self):
        """PRXGate parameters should reproduce an equivalent U3Gate."""
        alpha, beta = np.pi / 2, np.pi / 4
        prx = PRXGate(alpha, beta)
        u3 = U3Gate(alpha, beta - np.pi / 2, np.pi / 2 - beta)
        assert abs(prx.theta - u3.theta) < 1e-10
        assert abs(prx.phi - u3.phi) < 1e-10
        assert abs(prx.lam - u3.lam) < 1e-10


class TestPRXGateEdgeCases:
    """Parameter edge cases: zero, negative, and large values."""

    def test_zero_parameters(self):
        prx = PRXGate(0, 0)
        assert prx.alpha == 0
        assert prx.beta == 0
        assert abs(prx.phi - (-np.pi / 2)) < 1e-10
        assert abs(prx.lam - np.pi / 2) < 1e-10

    def test_negative_parameters(self):
        prx = PRXGate(-np.pi / 3, -np.pi / 6)
        assert prx.alpha == -np.pi / 3
        assert prx.beta == -np.pi / 6

    def test_large_parameters(self):
        alpha = 2 * np.pi + np.pi / 2
        beta = 3 * np.pi
        prx = PRXGate(alpha, beta)
        assert prx.alpha == alpha
        assert prx.beta == beta


class TestPRXGateInverse:
    """Tests for the inverse() method."""

    def test_inverse_returns_prxgate(self):
        prx = PRXGate(0.7, 1.2)
        inv = prx.inverse()
        assert isinstance(inv, PRXGate)
        assert inv.name == "prx"

    def test_inverse_negates_alpha_preserves_beta(self):
        alpha, beta = 0.7, 1.2
        prx = PRXGate(alpha, beta)
        inv = prx.inverse()
        assert abs(inv.alpha - (-alpha)) < 1e-10
        assert abs(inv.beta - beta) < 1e-10

    def test_double_inverse_is_identity(self):
        prx = PRXGate(0.7, 1.2)
        inv_inv = prx.inverse().inverse()
        assert abs(inv_inv.alpha - prx.alpha) < 1e-10
        assert abs(inv_inv.beta - prx.beta) < 1e-10


class TestPRXGateUnitary:
    """Verify the PRXGate unitary against the IQM SDK definition."""

    def test_matches_iqm_definition(self):
        """PRXGate unitary must match IQM's PRX = RZ(b) RX(a) RZ(-b)."""
        from qrisp import QuantumCircuit

        a, b = 0.7, 1.2
        qc = QuantumCircuit(1)
        qc.append(PRXGate(a, b), [0])
        U_prx = qc.get_unitary()

        U_expected = np.array(
            [
                [np.cos(a / 2), -1j * np.exp(-1j * b) * np.sin(a / 2)],
                [-1j * np.exp(1j * b) * np.sin(a / 2), np.cos(a / 2)],
            ]
        )
        assert np.allclose(U_prx, U_expected, atol=1e-6)
