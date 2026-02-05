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

import pytest

@pytest.mark.parametrize("input_a, input_b, expected_a, expected_b", [
    # both inputs are quantum in static mode, inputs are of unequal size
    (QuantumFloat(13), QuantumFloat(14), {20: 1.0}, {34: 1.0}),

    # both inputs are quantum in static mode, inputs are of equal size
    (QuantumFloat(13), QuantumFloat(13), {20: 1.0}, {34: 1.0}),

    # one input is classical, the other is quantum in static mode
    (20, QuantumFloat(15), 20, {34: 1.0}),
])
def test_cdkpm_adder_valid_input_static_mode(input_a, input_b, expected_a, expected_b):
    """Verify the function works as expected for valid inputs in static mode."""
    if isinstance(input_a, QuantumFloat):
        input_a[:] = 20
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 14

    cdkpm_adder(input_a, input_b)

    calculated_out_b = input_b.get_measurement() if isinstance(input_b, QuantumFloat) else input_b
    calculated_out_a = input_a.get_measurement() if isinstance(input_a, QuantumFloat) else input_a

    assert calculated_out_a == expected_a
    assert calculated_out_b == expected_b

@pytest.mark.parametrize("input_a_type, input_b_type, input_a_size, input_b_size, expected_output", [
    # both quantum inputs in dynamic mode, inputs are of unequal size
    (QuantumFloat, QuantumFloat, 16, 19, 34.0),

    # both quantum inputs in dynamic mode, inputs are of equal size
    (QuantumFloat, QuantumFloat, 19, 19, 34.0),

    # one classical input in dynamic mode
    (int, QuantumFloat, 16, 19, 34.0),
])
def test_jaspr_mode_cdkpm_adder(input_a_type, input_b_type, input_a_size, input_b_size, expected_output):
    def run_jasp_adder(i, j):
        a = input_a_type(i) if input_a_type != int else 20
        if input_a_type == QuantumFloat:
            a[:] = 20
        b = input_b_type(j)
        b[:] = 14
        cdkpm_adder(a, b)
        return measure(b)

    jaspr = make_jaspr(run_jasp_adder)(2, 3)
    assert jaspr(input_a_size, input_b_size) == expected_output


@pytest.mark.parametrize("input_a, input_b, expected_error_message", [
    # both inputs are classical
    (10, 15, "Attempted to call the CDKPM adder on invalid inputs"),
    # first input is quantum in static mode, second is classical
    (QuantumFloat(5), 10, "The second argument must be of type QuantumFloat."),
])
def test_invalid_input(input_a, input_b, expected_error_message):
    """Verify function raises error for invalid inputs."""
    if isinstance(input_a, QuantumFloat):
        input_a[:] = 2
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 14

    with pytest.raises(ValueError, match=expected_error_message):
        cdkpm_adder(input_a, input_b)


@pytest.mark.parametrize("input_a, input_b, expected_error_message", [
    # first input is quantum in dynamic mode, second is classical
    (QuantumFloat, 20, "The second argument must be of type QuantumFloat."),
    # both inputs are classical in dynamic mode
    (20, 20, "Attempted to call the CDKPM adder on invalid inputs"),
])
def test_invalid_input_dynamic_mode(input_a, input_b, expected_error_message):
    """Verify function raises error for invalid inputs in dynamic mode."""
    def run_jasp_adder(i, j):
        a = input_a(j) if input_a != 20 else 20
        if input_a == QuantumFloat:
            a[:] = 14
        cdkpm_adder(a, input_b)
        return measure(a)

    with pytest.raises(ValueError, match=expected_error_message):
        jaspr = make_jaspr(run_jasp_adder)(2, 3)
        jaspr(16, 19)

def test_inputs_modified():
    """Verify the size of inputs are unmodified (in-place) when they are initially of unequal size."""
    a = QuantumFloat(10)
    b = QuantumFloat(12)
    original_size_a = a.size
    original_size_b = b.size
    a[:] = 5
    b[:] = 7

    cdkpm_adder(a, b)

    assert a.size == original_size_a
    assert b.size == original_size_b


@pytest.mark.parametrize("i, j, a_value, b_value, ctrl_qbl_value, expected_result", [
    (10, 11, 3, 5, True, {8: 1.0}),  
    (10, 11, 3, 5, False, {5: 1.0}),  
])
def test_cdkpm_adder_static_mode_with_control(i, j, a_value, b_value, ctrl_qbl_value, expected_result):
    """Verify the CDKPM adder is triggered when the control qubit is in the |1> state
    in static mode. """
    a = QuantumFloat(i)
    b = QuantumFloat(j)
    a[:] = a_value
    b[:] = b_value
    ctrl_qbl = QuantumBool()
    if ctrl_qbl_value:
        x(ctrl_qbl[0])
    cdkpm_adder(a, b, ctrl=ctrl_qbl)
    result = b.get_measurement()
    assert result == expected_result

@pytest.mark.parametrize("i, j, a_value, b_value, ctrl_qbl_value, expected_result", [
    (16, 19, 4, 6, True, (4.0, 10.0)),  
    (16, 19, 4, 6, False, (4.0, 6.0)),  
])
def test_cdkpm_adder_dynamic_mode_with_control(i, j, a_value, b_value, ctrl_qbl_value, expected_result):
    """Verify the CDKPM adder is triggered when the control qubit is in the |1> state
    in dynamic mode. """
    def run_jasp_adder_with_control(i, j):
        a = QuantumFloat(i)
        b = QuantumFloat(j)
        a[:] = a_value
        b[:] = b_value
        ctrl_qbl = QuantumBool()
        if ctrl_qbl_value:
            x(ctrl_qbl[0])
        cdkpm_adder(a, b, ctrl=ctrl_qbl)
        return measure(a), measure(b)
    
    jaspr = make_jaspr(run_jasp_adder_with_control)(2, 3)
    result_a, result_b = jaspr(i, j)
    assert result_a == expected_result[0]
    assert result_b == expected_result[1]