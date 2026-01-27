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

from qrisp import *
import pytest

def test_cdkpm_adder_valid_input():
    """Verify the function works as expected for valid inputs."""

    # both inputs are quantum in static mode
    i, j = 13, 14
    a = QuantumFloat(i)
    a[:] = 20
    b = QuantumFloat(j)
    b[:] = 14

    cdkpm_adder(a, b)
    calculated_out_b = b.get_measurement()
    calculated_out_a = a.get_measurement()
    assert calculated_out_a == {20: 1.0}
    assert calculated_out_b == {34: 1.0}

    # one input is classical, the other is quantum in static mode
    a = 20
    b = QuantumFloat(15)
    b[:] = 14
    cdkpm_adder(a, b)
    calculated_out_b = b.get_measurement()
    calculated_out_a = a
    assert calculated_out_a == 20
    assert calculated_out_b == {34: 1.0}

    # one input is quantum, the other is classical in static mode
    b = 20
    a = QuantumFloat(15)
    a[:] = 14
    cdkpm_adder(a, b)
    # when the first input is quantum and the second is classical
    # the inputs are switched internally to be able to perform the addition
    # on the quantum object
    calculated_out_a = a.get_measurement()
    assert calculated_out_a == {34: 1.0}
    # verify b is unchanged outside of the adder
    assert b == 20
    

    # both quantum inputs in dynamic mode
    def run_jasp_adder(i, j):
        a = QuantumFloat(i)
        a[:] = 20
        b = QuantumFloat(j)
        b[:] = 14

        cdkpm_adder(a, b)
        return measure(b)

    jaspr = make_jaspr(run_jasp_adder)(2, 3)

    assert jaspr(16, 19) == 34.0

    # one classical input in dynamic mode
    def run_jasp_adder(i, j):
        a = 20
        b = QuantumFloat(j)
        b[:] = 14

        cdkpm_adder(a, b)
        return measure(b)

    jaspr = make_jaspr(run_jasp_adder)(2, 3)
    assert jaspr(16, 19) == 34.0


def test_invalid_input():
    """Verify function raises error for both classical inputs."""
    with pytest.raises(ValueError):
        cdkpm_adder(10, 15)