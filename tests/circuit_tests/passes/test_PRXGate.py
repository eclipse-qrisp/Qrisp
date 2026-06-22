"""
********************************************************************************
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

from qrisp.circuit import PRXGate
from qrisp import U3Gate
import numpy as np


def test_PRXGate():
    """Test the PRXGate class functionality."""

    print("Testing PRXGate...")

    # Test case 1: Basic PRXGate creation
    alpha = np.pi / 2
    beta = np.pi / 4
    prx_gate = PRXGate(alpha, beta)

    assert hasattr(prx_gate, 'alpha'), "PRXGate should have alpha parameter"
    assert hasattr(prx_gate, 'beta'), "PRXGate should have beta parameter"
    assert hasattr(prx_gate, 'theta'), "PRXGate should have theta parameter (inherited)"
    assert hasattr(prx_gate, 'phi'), "PRXGate should have phi parameter (inherited)"
    assert hasattr(prx_gate, 'lam'), "PRXGate should have lam parameter (inherited)"

    assert prx_gate.alpha == alpha, f"Alpha should be {alpha}, got {prx_gate.alpha}"
    assert prx_gate.beta == beta, f"Beta should be {beta}, got {prx_gate.beta}"

    # PRX(alpha, beta) = U3(alpha, beta - pi/2, pi/2 - beta)
    assert abs(prx_gate.theta - alpha) < 1E-10, f"Theta should be {alpha}, got {prx_gate.theta}"
    assert abs(prx_gate.phi - (beta - np.pi / 2)) < 1E-10, \
        f"Phi should be {beta - np.pi/2}, got {prx_gate.phi}"
    assert abs(prx_gate.lam - (np.pi / 2 - beta)) < 1E-10, \
        f"Lambda should be {np.pi/2 - beta}, got {prx_gate.lam}"

    print(f"  Basic PRXGate creation: alpha={alpha:.3f}, beta={beta:.3f}")

    # Test case 2: PRXGate is subclass of U3Gate
    assert isinstance(prx_gate, U3Gate), "PRXGate should be an instance of U3Gate"
    assert issubclass(PRXGate, U3Gate), "PRXGate should be a subclass of U3Gate"

    print("  PRXGate inheritance test passed")

    # Test case 3: PRXGate name attribute
    assert prx_gate.name == "prx", f"Name should be 'prx', got '{prx_gate.name}'"

    print("  PRXGate name test passed")

    # Test case 4: PRXGate with zero parameters
    prx_zero = PRXGate(0, 0)

    assert prx_zero.alpha == 0, "Zero alpha should be preserved"
    assert prx_zero.beta == 0, "Zero beta should be preserved"
    assert abs(prx_zero.phi - (-np.pi / 2)) < 1E-10, "Phi should be -pi/2"
    assert abs(prx_zero.lam - (np.pi / 2)) < 1E-10, "Lambda should be pi/2"

    print("  Zero parameters test passed")

    # Test case 5: PRXGate with negative parameters
    alpha_neg = -np.pi / 3
    beta_neg = -np.pi / 6
    prx_neg = PRXGate(alpha_neg, beta_neg)

    assert prx_neg.alpha == alpha_neg, "Negative alpha should be preserved"
    assert prx_neg.beta == beta_neg, "Negative beta should be preserved"

    print("  Negative parameters test passed")

    # Test case 6: PRXGate with large parameters
    alpha_large = 2 * np.pi + np.pi / 2
    beta_large = 3 * np.pi
    prx_large = PRXGate(alpha_large, beta_large)

    assert prx_large.alpha == alpha_large, "Large alpha should be preserved"
    assert prx_large.beta == beta_large, "Large beta should be preserved"

    print("  Large parameters test passed")

    # Test case 7: PRXGate U3 equivalence
    # PRX(alpha, beta) should match U3(alpha, beta - pi/2, pi/2 - beta)
    u3_equivalent = U3Gate(alpha, beta - np.pi / 2, np.pi / 2 - beta)

    assert abs(prx_gate.theta - u3_equivalent.theta) < 1E-10, "Theta should match U3Gate"
    assert abs(prx_gate.phi - u3_equivalent.phi) < 1E-10, "Phi should match U3Gate"
    assert abs(prx_gate.lam - u3_equivalent.lam) < 1E-10, "Lambda should match U3Gate"

    print("  U3Gate equivalence test passed")

    # Test case 8: Inverse
    prx_inv = prx_gate.inverse()
    assert isinstance(prx_inv, PRXGate), "Inverse should be a PRXGate"
    assert abs(prx_inv.alpha - (-alpha)) < 1E-10, \
        f"Inverse alpha should be {-alpha}, got {prx_inv.alpha}"
    assert abs(prx_inv.beta - beta) < 1E-10, \
        f"Inverse beta should be {beta}, got {prx_inv.beta}"
    assert prx_inv.name == "prx", "Inverse name should be 'prx'"

    # Double inverse = identity
    prx_inv_inv = prx_inv.inverse()
    assert abs(prx_inv_inv.alpha - prx_gate.alpha) < 1E-10, "Double inverse should restore alpha"
    assert abs(prx_inv_inv.beta - prx_gate.beta) < 1E-10, "Double inverse should restore beta"

    print("  Inverse test passed")

    # Test case 9: Unitary correctness (vs IQM SDK definition)
    from qrisp import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.append(PRXGate(0.7, 1.2), [0])
    U_prx = qc.get_unitary()

    # IQM PRX unitary: [[cos(a/2), -i e^{-ib} sin(a/2)], [-i e^{ib} sin(a/2), cos(a/2)]]
    a, b = 0.7, 1.2
    U_expected = np.array([
        [np.cos(a / 2), -1j * np.exp(-1j * b) * np.sin(a / 2)],
        [-1j * np.exp(1j * b) * np.sin(a / 2), np.cos(a / 2)],
    ])
    assert np.allclose(U_prx, U_expected, atol=1E-6), "PRXGate unitary should match IQM definition"

    print("  Unitary correctness test passed")

    print("All PRXGate tests passed!")
