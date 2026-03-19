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
    (QuantumFloat(14), QuantumFloat(13), {20: 1.0}, {34: 1.0}),

    # both inputs are quantum in static mode, inputs are of equal size
    (QuantumFloat(13), QuantumFloat(13), {20: 1.0}, {34: 1.0}),

    # one input is classical, the other is quantum in static mode
    (20, QuantumFloat(15), 20, {34: 1.0}),
])
def test_cuccaro_adder_valid_input_static_mode(input_a, input_b, expected_a, expected_b):
    """Verify the function works as expected for valid inputs in static mode."""
    if isinstance(input_a, QuantumFloat):
        input_a[:] = 20
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 14

    cuccaro_adder(input_a, input_b)

    calculated_out_b = input_b.get_measurement() if isinstance(input_b, QuantumFloat) else input_b
    calculated_out_a = input_a.get_measurement() if isinstance(input_a, QuantumFloat) else input_a

    assert calculated_out_a == expected_a
    assert calculated_out_b == expected_b

@pytest.mark.parametrize("input_a, input_b, expected_a, expected_b", [
    # both inputs are quantum in static mode, inputs are of unequal size
    (QuantumFloat(13), QuantumFloat(14), {20: 1.0}, {34: 1.0}),

    # both inputs are quantum in static mode, inputs are of equal size
    (QuantumFloat(13), QuantumFloat(13), {20: 1.0}, {34: 1.0}),

    # one input is classical, the other is quantum in static mode
    (20, QuantumFloat(15), 20, {34: 1.0}),
])
def test_cuccaro_adder_valid_input_static_mode_with_cout(input_a, input_b, expected_a, expected_b):
    """Verify the function works as expected for valid inputs in static mode when c_out is provided."""
    if isinstance(input_a, QuantumFloat):
        input_a[:] = 20
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 14

    c_out = QuantumFloat(2)
    cuccaro_adder(input_a, input_b, c_out=c_out)

    calculated_out_b = input_b.get_measurement() if isinstance(input_b, QuantumFloat) else input_b
    calculated_out_a = input_a.get_measurement() if isinstance(input_a, QuantumFloat) else input_a
    calculated_out_cout = c_out.get_measurement()

    assert calculated_out_a == expected_a
    assert calculated_out_b == expected_b
    assert calculated_out_cout == {0: 1.0}  # since no overflow is expected in these test cases


def test_jaspr_mode_cuccaro_adder():
    """Verify the function works as expected in dynamic mode.
     This test covers both cases where the inputs are of equal size and where they are of unequal size."""

    @boolean_simulation
    def main(N, L, j, k):
    
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k

        cuccaro_adder(j, B)
        return measure(A), measure(B)
        
    for N in range(2, 5):
        for L in range(2, 5):
            for j in range(2**N):
                for k in range(2**L):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k+j)%(2**L)


def test_inputs_modified():
    """Verify the size of inputs are unmodified (in-place) when they are initially of unequal size."""
    a = QuantumFloat(10)
    b = QuantumFloat(12)
    original_size_a = a.size
    original_size_b = b.size
    a[:] = 5
    b[:] = 7

    cuccaro_adder(a, b)

    assert a.size == original_size_a
    assert b.size == original_size_b


@pytest.mark.parametrize("i, j, a_value, b_value, ctrl_qbl_value, expected_result", [
    (10, 11, 3, 5, True, {8: 1.0}),  
    (10, 11, 3, 5, False, {5: 1.0}),  
])
def test_cuccaro_adder_static_mode_with_control(i, j, a_value, b_value, ctrl_qbl_value, expected_result):
    """Verify the CDKPM adder is triggered when the control qubit is in the |1> state
    in static mode. """
    a = QuantumFloat(i)
    b = QuantumFloat(j)
    a[:] = a_value
    b[:] = b_value
    ctrl_qbl = QuantumBool()
    if ctrl_qbl_value:
        x(ctrl_qbl[0])
    cuccaro_adder(a, b, ctrl=ctrl_qbl)
    result = b.get_measurement()
    assert result == expected_result


def test_cuccaro_adder_dynamic_mode_with_control():
    """Verify the CDKPM adder is triggered when the control qubit is in the |1> state
    in dynamic mode. """
    @boolean_simulation
    def main(N, L, j, k):
        
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        qbl.flip()

        with control(qbl):
            cuccaro_adder(A, B)
        return measure(A), measure(B)
        
    for N in range(2, 5):
        for L in range(2, 5):
            for j in range(2**N):
                for k in range(2**L):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k+j)%(2**L)
