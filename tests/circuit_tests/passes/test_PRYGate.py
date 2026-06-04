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

from qrisp.circuit import PRYGate
from qrisp import U3Gate
import numpy as np

def test_PRYGate():
    """Test the PRYGate class functionality."""
    
    print("Testing PRYGate...")
    
    # Test case 1: Basic PRYGate creation
    theta = np.pi/2
    phi = np.pi/4
    pry_gate = PRYGate(theta, phi)
    
    assert hasattr(pry_gate, 'theta'), "PRYGate should have theta parameter"
    assert hasattr(pry_gate, 'phi'), "PRYGate should have phi parameter"
    assert hasattr(pry_gate, 'lam'), "PRYGate should have lam parameter"
    
    assert pry_gate.theta == theta, f"Theta should be {theta}, got {pry_gate.theta}"
    assert pry_gate.phi == phi, f"Phi should be {phi}, got {pry_gate.phi}"
    assert pry_gate.lam == -phi, f"Lambda should be {-phi}, got {pry_gate.lam}"
    
    print(f"✓ Basic PRYGate creation: θ={theta:.3f}, φ={phi:.3f}, λ={-phi:.3f}")
    
    # Test case 2: PRYGate is subclass of U3Gate
    assert isinstance(pry_gate, U3Gate), "PRYGate should be an instance of U3Gate"
    assert issubclass(PRYGate, U3Gate), "PRYGate should be a subclass of U3Gate"
    
    print("✓ PRYGate inheritance test passed")
    
    # Test case 3: PRYGate with zero parameters
    pry_zero = PRYGate(0, 0)
    
    assert pry_zero.theta == 0, "Zero theta should be preserved"
    assert pry_zero.phi == 0, "Zero phi should be preserved"
    assert pry_zero.lam == 0, "Zero lambda should result from -phi"
    
    print("✓ Zero parameters test passed")
    
    # Test case 4: PRYGate with negative parameters
    theta_neg = -np.pi/3
    phi_neg = -np.pi/6
    pry_neg = PRYGate(theta_neg, phi_neg)
    
    assert pry_neg.theta == theta_neg, "Negative theta should be preserved"
    assert pry_neg.phi == phi_neg, "Negative phi should be preserved"
    assert pry_neg.lam == -phi_neg, "Lambda should be negative of phi"
    
    print("✓ Negative parameters test passed")
    
    # Test case 5: PRYGate with large parameters
    theta_large = 2*np.pi + np.pi/2
    phi_large = 3*np.pi
    pry_large = PRYGate(theta_large, phi_large)
    
    assert pry_large.theta == theta_large, "Large theta should be preserved"
    assert pry_large.phi == phi_large, "Large phi should be preserved"
    assert pry_large.lam == -phi_large, "Lambda should be negative of large phi"
    
    print("✓ Large parameters test passed")
    
    # Test case 6: PRYGate equivalence with U3Gate for special case
    # When lambda = -phi, PRYGate(theta, phi) should equal U3Gate(theta, phi, -phi)
    u3_equivalent = U3Gate(theta, phi, -phi)
    
    assert pry_gate.theta == u3_equivalent.theta, "Theta should match U3Gate"
    assert pry_gate.phi == u3_equivalent.phi, "Phi should match U3Gate"
    assert pry_gate.lam == u3_equivalent.lam, "Lambda should match U3Gate"
    
    print("✓ U3Gate equivalence test passed")
    
    print("✓ All PRYGate tests passed!")


if __name__ == "__main__":
    test_PRYGate()