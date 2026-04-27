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

from qrisp import QuantumFloat, gidney_adder, QuantumBool, x, control, boolean_simulation, measure
from qrisp import cuccaro_adder
import pytest


@pytest.mark.parametrize(
    "input_a, input_b, expected_a, expected_b",
    [
        # Both inputs are quantum in static mode, inputs are of unequal size
        (QuantumFloat(13), QuantumFloat(14), {20: 1.0}, {34: 1.0}),
        (QuantumFloat(14), QuantumFloat(13), {20: 1.0}, {34: 1.0}),
        # Both inputs are quantum in static mode, inputs are of equal size
        (QuantumFloat(13), QuantumFloat(13), {20: 1.0}, {34: 1.0}),
        # One input is classical, the other is quantum in static mode
        (20, QuantumFloat(15), 20, {34: 1.0}),
    ],
)
def test_gidney_adder_valid_input_static_mode(input_a, input_b, expected_a, expected_b):
    """Verify the function works as expected for valid inputs in static mode."""
    if isinstance(input_a, QuantumFloat):
        input_a[:] = 20
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 14

    gidney_adder(input_a, input_b)

    calculated_out_b = (
        input_b.get_measurement() if isinstance(input_b, QuantumFloat) else input_b
    )
    calculated_out_a = (
        input_a.get_measurement() if isinstance(input_a, QuantumFloat) else input_a
    )

    assert calculated_out_a == expected_a
    assert calculated_out_b == expected_b


@pytest.mark.parametrize(
    "input_a, input_b, expected_a, expected_b",
    [
        # Both inputs are quantum in static mode, inputs are of unequal size
        (QuantumFloat(13), QuantumFloat(14), {20: 1.0}, {34: 1.0}),
        # Both inputs are quantum in static mode, inputs are of equal size
        (QuantumFloat(13), QuantumFloat(13), {20: 1.0}, {34: 1.0}),
        # One input is classical, the other is quantum in static mode
        (20, QuantumFloat(15), 20, {34: 1.0}),
    ],
)
def test_gidney_adder_valid_input_static_mode_with_cout(
    input_a, input_b, expected_a, expected_b
):
    """Verify the function works as expected for valid inputs in static mode when c_out is provided."""
    if isinstance(input_a, QuantumFloat):
        input_a[:] = 20
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 14

    c_out = QuantumBool()
    gidney_adder(input_a, input_b, c_out=c_out)

    calculated_out_b = (
        input_b.get_measurement() if isinstance(input_b, QuantumFloat) else input_b
    )
    calculated_out_a = (
        input_a.get_measurement() if isinstance(input_a, QuantumFloat) else input_a
    )
    calculated_out_cout = c_out.get_measurement()

    assert calculated_out_a == expected_a
    assert calculated_out_b == expected_b
    assert calculated_out_cout == {0: 1.0}  # No overflow expected in these test cases


@pytest.mark.parametrize(
    "input_a, input_b, c_in_val, expected_b",
    [
        # Test with carry in, no overflow
        (5, QuantumFloat(8), 0, {5: 1.0}),  # 0 + 5 = 5
        (5, QuantumFloat(8), 1, {6: 1.0}),  # 1 + 5 = 6
        # Test with larger values and carry in
        (20, QuantumFloat(10), 1, {21: 1.0}),  # 0 + 20 + 1 = 21 (mod 2^10)
    ],
)
def test_gidney_adder_with_carry_in(input_a, input_b, c_in_val, expected_b):
    """Verify the function works correctly with carry_in parameter."""
    if isinstance(input_b, QuantumFloat):
        input_b[:] = 0

    c_in = QuantumBool()
    if c_in_val:
        x(c_in[0])

    gidney_adder(input_a, input_b, c_in=c_in)

    calculated_out_b = input_b.get_measurement()
    assert calculated_out_b == expected_b


@pytest.mark.parametrize(
    "a_val, b_val_init, expected_sum, expected_cout",
    [
        # Test overflow cases where carry out should be 1
        (255, 128, {127: 1.0}, {1: 1.0}),  # 255 + 128 = 383 = 127 (mod 256), cout=1
        (128, 128, {0: 1.0}, {1: 1.0}),  # 128 + 128 = 256 = 0 (mod 256), cout=1
        (200, 100, {44: 1.0}, {1: 1.0}),  # 200 + 100 = 300 = 44 (mod 256), cout=1
    ],
)
def test_gidney_adder_carry_out_overflow(
    a_val, b_val_init, expected_sum, expected_cout
):
    """Verify carry out is correctly set when overflow occurs."""
    b_val = QuantumFloat(8)
    b_val[:] = b_val_init

    c_out = QuantumBool()
    gidney_adder(a_val, b_val, c_out=c_out)

    result_b = b_val.get_measurement()
    result_cout = c_out.get_measurement()

    assert result_b == expected_sum
    assert result_cout == expected_cout


def test_gidney_adder_jasp_mode():
    """Verify the function works as expected in dynamic (jasp) mode.
    This test covers both cases where inputs are of equal and unequal size."""

    @boolean_simulation
    def main(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k

        gidney_adder(A, B)
        return measure(A), measure(B)

    for N in range(2, 5):
        for L in range(2, 5):
            for j in range(2**N):
                for k in range(2**L):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % (2**L)


def test_gidney_adder_inputs_unmodified_size():
    """Verify the size of inputs are unmodified (in-place) when they are initially of unequal size."""
    a = QuantumFloat(10)
    b = QuantumFloat(12)
    original_size_a = a.size
    original_size_b = b.size
    a[:] = 5
    b[:] = 7

    gidney_adder(a, b)

    assert a.size == original_size_a
    assert b.size == original_size_b


@pytest.mark.parametrize(
    "i, j, a_value, b_value, ctrl_qbl_value, expected_result",
    [
        (10, 11, 3, 5, True, {8: 1.0}),  # Control on: 5 + 3 = 8
        (10, 11, 3, 5, False, {5: 1.0}),  # Control off: 5 remains 5
    ],
)
def test_gidney_adder_static_mode_with_control(
    i, j, a_value, b_value, ctrl_qbl_value, expected_result
):
    """Verify the gidney adder is triggered when the control qubit is in the |1> state
    in static mode."""
    a = QuantumFloat(i)
    b = QuantumFloat(j)
    a[:] = a_value
    b[:] = b_value
    ctrl_qbl = QuantumBool()
    if ctrl_qbl_value:
        x(ctrl_qbl[0])

    gidney_adder(a, b, ctrl=ctrl_qbl)
    result = b.get_measurement()
    assert result == expected_result


def test_gidney_adder_dynamic_mode_with_control():
    """Verify the gidney adder is triggered when the control qubit is in the |1> state
    in dynamic (jasp) mode."""

    @boolean_simulation
    def main(N, L, j, k):
        A = QuantumFloat(N)
        B = QuantumFloat(L)
        A[:] = j
        B[:] = k
        qbl = QuantumBool()
        qbl.flip()

        with control(qbl):
            gidney_adder(A, B)
        return measure(A), measure(B)

    for N in range(2, 5):
        for L in range(2, 5):
            for j in range(2**N):
                for k in range(2**L):
                    A, B = main(N, L, j, k)
                    assert A == j
                    assert B == (k + j) % (2**L)


def test_gidney_adder_with_carry_in_and_carry_out():
    """Verify the function works correctly when both c_in and c_out are provided."""
    a = QuantumFloat(8)
    b = QuantumFloat(8)
    a[:] = 100
    b[:] = 150

    c_in = QuantumBool()
    x(c_in[0])  # Set carry in to 1

    c_out = QuantumBool()

    gidney_adder(a, b, c_in=c_in, c_out=c_out)

    result_b = b.get_measurement()
    result_cout = c_out.get_measurement()

    # 150 + 100 + 1 = 251, no overflow
    assert result_b == {251: 1.0}
    assert result_cout == {0: 1.0}


def test_gidney_adder_single_qubit():
    """Verify the function works correctly with single-qubit inputs."""
    a = QuantumFloat(1)
    b = QuantumFloat(1)
    a[:] = 0
    b[:] = 1

    gidney_adder(a, b)

    result_b = b.get_measurement()
    assert result_b == {1: 1.0}


def test_gidney_adder_single_qubit_with_carry():
    """Verify single-qubit addition with carry out."""
    a = QuantumFloat(1)
    b = QuantumFloat(1)
    a[:] = 1
    b[:] = 1

    c_out = QuantumBool()
    gidney_adder(a, b, c_out=c_out)

    result_b = b.get_measurement()
    result_cout = c_out.get_measurement()

    # 1 + 1 = 2 = 0 (mod 2), with carry out 1
    assert result_b == {0: 1.0}
    assert result_cout == {1: 1.0}


def test_gidney_adder_zero_addition():
    """Verify adding zero to a quantum variable produces the original value."""
    a = 0
    b = QuantumFloat(10)
    b[:] = 42

    gidney_adder(a, b)

    result_b = b.get_measurement()
    assert result_b == {42: 1.0}


def test_gidney_adder_quantum_zero():
    """Verify adding a quantum zero to another quantum variable."""
    a = QuantumFloat(10)
    b = QuantumFloat(10)
    a[:] = 0
    b[:] = 42

    gidney_adder(a, b)

    result_a = a.get_measurement()
    result_b = b.get_measurement()
    assert result_a == {0: 1.0}
    assert result_b == {42: 1.0}


@pytest.mark.parametrize(
    "size_a, size_b, val_a, val_b",
    [
        (5, 10, 7, 15),
        (10, 5, 15, 7),
        (8, 8, 100, 50),
        (16, 4, 255, 3),
    ],
)
def test_gidney_adder_various_sizes(size_a, size_b, val_a, val_b):
    """Verify the function works with various input sizes."""
    a = QuantumFloat(size_a)
    b = QuantumFloat(size_b)
    a[:] = val_a
    b[:] = val_b

    gidney_adder(a, b)

    result_a = a.get_measurement()
    result_b = b.get_measurement()

    assert result_a == {val_a: 1.0}
    # Result should be (val_b + val_a) mod 2^size_b
    expected_b = (val_b + val_a) % (2**size_b)
    assert result_b == {expected_b: 1.0}


def test_gidney_adder_string_input():
    """Verify the function handles string inputs (little-endian bit strings) correctly."""
    b = QuantumFloat(8)
    b[:] = 5

    # "10" in little-endian = "01" in big-endian = 1 in decimal
    gidney_adder("10", b)

    result_b = b.get_measurement()
    assert result_b == {6: 1.0}  # 5 + 1 = 6


def test_gidney_adder_empty_string_input():
    """Verify the function handles empty string input (zero)."""
    b = QuantumFloat(8)
    b[:] = 42

    gidney_adder("", b)

    result_b = b.get_measurement()
    assert result_b == {42: 1.0}  # 42 + 0 = 42


def test_gidney_adder_t_depth_significantly_lower_than_cuccaro():
    """Verify that the gidney adder achieves significant T-depth reduction across inputs of
    different sizes.

    The Gidney adder is specifically designed to minimize T-depth.
    """

    # Test with multiple bit widths to ensure consistent advantage
    for qf_size in [8, 12, 16]:
        # Cuccaro adder T-depth
        a_cuccaro = QuantumFloat(qf_size)
        b_cuccaro = QuantumFloat(qf_size)
        a_cuccaro[:] = 5
        b_cuccaro[:] = 10

        cuccaro_adder(a_cuccaro, b_cuccaro)

        cuccaro_circuit = b_cuccaro.qs.compile()
        cuccaro_t_depth = cuccaro_circuit.t_depth()

        # Gidney adder T-depth
        a_gidney = QuantumFloat(qf_size)
        b_gidney = QuantumFloat(qf_size)
        a_gidney[:] = 5
        b_gidney[:] = 10

        gidney_adder(a_gidney, b_gidney)

        gidney_circuit = b_gidney.qs.compile()
        gidney_t_depth = gidney_circuit.t_depth()

        # Gidney should have lower T-depth
        assert gidney_t_depth < cuccaro_t_depth


def test_gidney_adder_classical_a_modulo():
    """
    Verify that a classical `a` larger than b's range is reduced mod 2**len(b).
    """
    b_bits = 4

    test_cases = [
        (0, 17),  # 17 % 16 = 1, expect 0 + 1 = 1
        (3, 19),  # 19 % 16 = 3, expect 3 + 3 = 6
        (5, 32),  # 32 % 16 = 0, expect 5 + 0 = 5
        (7, 255),  # 255 % 16 = 15, expect (7 + 15) % 16 = 6
    ]

    for b_val, a_val in test_cases:
        b = QuantumFloat(b_bits)
        b[:] = b_val
        gidney_adder(a_val, b)
        result = b.get_measurement()
        expected = {(b_val + a_val) % (1 << b_bits): 1.0}
        assert result == expected


def test_gidney_adder_t_count():
    """Verify that the Gidney adder for two n-qubit inputs uses 4*(n-1) T gates.

    This follows from Fig. 1 of the Gidney paper (https://arxiv.org/abs/1709.06648):
    each of the (n-1) forward Toffolis decomposes into 4 T gates via the logical-AND
    construction, while the (n-1) inverse Toffolis use measurement-based uncomputation
    with 0 T gates, giving a total T count of 4*(n-1).
    """
    for n in [3, 5, 8, 12]:
        a = QuantumFloat(n)
        b = QuantumFloat(n)
        a[:] = 2
        b[:] = 3

        gidney_adder(a, b)

        qc = b.qs.compile(compile_mcm=True).transpile()
        ops = qc.count_ops()
        t_count = ops.get("t", 0) + ops.get("t_dg", 0)

        expected_t_count = 4 * (n - 1)
        assert t_count == expected_t_count
