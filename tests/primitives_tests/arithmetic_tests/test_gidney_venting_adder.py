"""
Unit tests for Gidney Figure 1 gate implementations (symbolic parity version).
These helpers are dynamic-mode only and rely on JASP tracing semantics.
"""

import pytest
from qrisp import QuantumFloat, QuantumVariable, x, measure, multi_measurement
from qrisp.jasp import jaspify

from qrisp.alg_primitives.arithmetic.adders.gidney_venting_adder import (
    bit_inverted_mcx,
    zz_mcx,
    zz_zz_mcx,
    bit_inverted_zz_zz_mcx,
    carry_venting_adder,
    carry_xor_block,
)



# bit_inverted_mcx: fires when (control XOR b) == 1 (symbolic parity)
BIT_INVERTED_CASES = [
    (0, False, 0),    # control=0, b=False -> (0 XOR 0)=0 -> no fire
    (1, False, 1),    # control=1, b=False -> (1 XOR 0)=1 -> fires
    (0, True,  1),    # control=0, b=True  -> (0 XOR 1)=1 -> fires
    (1, True,  0),    # control=1, b=True  -> (1 XOR 1)=0 -> no fire
]

# zz_mcx: fires when z0 != z1 (odd parity, per Gidney's paper)
ZZ_PARITY_CASES = [
    (0, 0, 0),    # both 0 -> equal -> no fire
    (1, 1, 0),    # both 1 -> equal -> no fire
    (0, 1, 1),    # different -> fires
    (1, 0, 1),    # different -> fires
]

# zz_zz_mcx: fires when (z_left!=z_left_right) AND (z_left_right!=z_right) (both odd parity, per Gidney's paper)
# New signature: (z_left, z_left_right, z_right, target) with z_left_right shared between both pairs
DUAL_ZZ_CASES = [
    (0, 0, 0, 0),    # z_left⊕z_left_right=0, z_left_right⊕z_right=0 (both even) -> no fire
    (1, 1, 1, 0),    # z_left⊕z_left_right=0, z_left_right⊕z_right=0 (both even) -> no fire
    (1, 0, 0, 0),    # z_left⊕z_left_right=1, z_left_right⊕z_right=0 (left odd, right even) -> no fire
    (0, 0, 1, 0),    # z_left⊕z_left_right=0, z_left_right⊕z_right=1 (left even, right odd) -> no fire
    (1, 0, 1, 1),    # z_left⊕z_left_right=1, z_left_right⊕z_right=1 (both odd) -> fires
]

# bit_inverted_zz_zz_mcx: fires when (ctrl1 XOR b) AND (ctrl2 XOR b) == 1
BIT_INVERTED_ZZ_ZZ_CASES = [
    (0, 0, False, 0),    # (0⊕0) AND (0⊕0) = 0 AND 0 = 0 -> no fire
    (1, 1, False, 1),    # (1⊕0) AND (1⊕0) = 1 AND 1 = 1 -> fires
    (0, 1, False, 0),    # (0⊕0) AND (1⊕0) = 0 AND 1 = 0 -> no fire
    (1, 0, False, 0),    # (1⊕0) AND (0⊕0) = 1 AND 0 = 0 -> no fire
    (0, 0, True,  1),    # (0⊕1) AND (0⊕1) = 1 AND 1 = 1 -> fires
    (1, 1, True,  0),    # (1⊕1) AND (1⊕1) = 0 AND 0 = 0 -> no fire
    (0, 1, True,  0),    # (0⊕1) AND (1⊕1) = 1 AND 0 = 0 -> no fire
    (1, 0, True,  0),    # (1⊕1) AND (0⊕1) = 0 AND 1 = 0 -> no fire
]



@pytest.mark.parametrize("ctrl_val, b, expected_tgt", BIT_INVERTED_CASES)
def test_bit_inverted_mcx_jasp(ctrl_val, b, expected_tgt):
    """Verify bit_inverted_mcx works in JASP dynamic mode (explicit parity extraction)."""
    @jaspify
    def run():
        ctrl = QuantumVariable(1)
        always_one = QuantumVariable(1)
        tgt = QuantumVariable(1)

        if ctrl_val:
            x(ctrl[0])
        x(always_one[0])  # always set to 1

        bit_inverted_mcx(ctrl[0], always_one[0], tgt[0], b)

        return measure(ctrl), measure(always_one), measure(tgt)

    measured_ctrl, measured_always_one, measured_tgt = run()
    assert int(measured_ctrl) == ctrl_val
    assert int(measured_always_one) == 1
    assert int(measured_tgt) == expected_tgt


@pytest.mark.parametrize("q0v, q1v, expected_tgt", ZZ_PARITY_CASES)
def test_zz_mcx_jasp(q0v, q1v, expected_tgt):
    """Verify zz_mcx works in JASP dynamic mode (explicit parity extraction)."""
    @jaspify
    def run():
        q0 = QuantumVariable(1)
        q1 = QuantumVariable(1)
        ctrl = QuantumVariable(1)
        tgt = QuantumVariable(1)

        if q0v:
            x(q0[0])
        if q1v:
            x(q1[0])
        x(ctrl[0])  # always set control to 1

        zz_mcx(q0[0], q1[0], ctrl[0], tgt[0])

        return measure(q0), measure(q1), measure(ctrl), measure(tgt)

    measured_q0, measured_q1, measured_ctrl, measured_tgt = run()
    assert int(measured_q0) == q0v
    assert int(measured_q1) == q1v
    assert int(measured_ctrl) == 1
    assert int(measured_tgt) == expected_tgt


@pytest.mark.parametrize("z_left_v, z_left_right_v, z_right_v, expected_tgt", DUAL_ZZ_CASES)
def test_zz_zz_mcx_jasp(z_left_v, z_left_right_v, z_right_v, expected_tgt):
    """Verify zz_zz_mcx works in JASP dynamic mode (with shared z_left_right qubit)."""
    @jaspify
    def run():
        z_left = QuantumVariable(1)
        z_left_right = QuantumVariable(1)
        z_right = QuantumVariable(1)
        tgt = QuantumVariable(1)

        if z_left_v:
            x(z_left[0])
        if z_left_right_v:
            x(z_left_right[0])
        if z_right_v:
            x(z_right[0])

        zz_zz_mcx(z_left[0], z_left_right[0], z_right[0], tgt[0])

        return measure(z_left), measure(z_left_right), measure(z_right), measure(tgt)

    measured_z_left, measured_z_left_right, measured_z_right, measured_tgt = run()
    assert int(measured_z_left) == z_left_v
    assert int(measured_z_left_right) == z_left_right_v
    assert int(measured_z_right) == z_right_v
    assert int(measured_tgt) == expected_tgt


@pytest.mark.parametrize("ctrl1_v, ctrl2_v, b, expected_tgt", BIT_INVERTED_ZZ_ZZ_CASES)
def test_bit_inverted_zz_zz_mcx_jasp(ctrl1_v, ctrl2_v, b, expected_tgt):
    """Verify bit_inverted_zz_zz_mcx fires when both (ctrl_i XOR b) == 1."""
    @jaspify
    def run():
        ctrl1 = QuantumVariable(1)
        ctrl2 = QuantumVariable(1)
        tgt = QuantumVariable(1)
 
        if ctrl1_v:
            x(ctrl1[0])
        if ctrl2_v:
            x(ctrl2[0])
 
        bit_inverted_zz_zz_mcx(ctrl1[0], ctrl2[0], tgt[0], b)
 
        return measure(ctrl1), measure(ctrl2), measure(tgt)
 
    measured_ctrl1, measured_ctrl2, measured_tgt = run()
    assert int(measured_ctrl1) == ctrl1_v
    assert int(measured_ctrl2) == ctrl2_v
    assert int(measured_tgt) == expected_tgt
 
 
CARRY_VENTING_CASES = [
    (0b01, 1, 0, 0b10),   # simple: 1 + 1 = 2
    (0b01, 1, 1, 0b11),   # carry-in: 1 + 1 + 1 = 3
    (0b11, 1, 0, 0b00),   # overflow: 3 + 1 = 0 mod 4
    (0b01, 0, 0, 0b01),   # zero addend: 1 + 0 = 1
    (0b11, 3, 0, 0b10),   # all ones: 3 + 3 = 2 mod 4
]

@pytest.mark.parametrize("init, d, c_in_val, expected", CARRY_VENTING_CASES)
def test_carry_venting_adder_jasp(init, d, c_in_val, expected):
    """Test carry_venting_adder first block with various 2-bit additions."""
    @jaspify
    def run():
        target = QuantumVariable(2)

        for i in range(2):
            if (init >> i) & 1:
                x(target[i])

        c_in = QuantumVariable(1)
        if c_in_val:
            x(c_in[0])

        carry_venting_adder(target, d, c_in)

        return measure(target)

    result_target = run()
    result_val = int(result_target)

    assert result_val == expected


def test_carry_venting_adder_jasp_large_register():
    """Test carry_venting_adder arithmetic on larger registers."""
    @jaspify
    def run():
        target = QuantumVariable(8)

        for i in [3, 4, 7]:
            x(target[i])

        d = 55
        c_in = QuantumVariable(1)

        vented = carry_venting_adder(target, d, c_in)

        return measure(target), len(vented)

    result_target, num_vented = run()
    result_val = int(result_target)

    # init=152 (bits 3,4,7), d=55, c_in=0 → expected = (152+55) % 256 = 207
    assert result_val == 207, f"Expected 207, got {result_val}"

    # The vented list captures trace-time appends (2 from jrange tracing).
    # Actual runtime vents = n-2 = 6.
    assert num_vented == 4
 
 
CARRY_XOR_CASES = [
    (0b0101, 3, None),    # target=5, d=3, no carry-in
    (0b0111, 2, 1),       # target=7, d=2, carry-in=1
]

@pytest.mark.parametrize("init, d, c_in_val", CARRY_XOR_CASES)
def test_carry_xor_block_jasp(init, d, c_in_val):
    """Test carry_xor_block executes without error."""
    @jaspify
    def run():
        target = QuantumVariable(4)
        dirty_ancillas = QuantumVariable(4)

        for i in range(4):
            if (init >> i) & 1:
                x(target[i])

        if c_in_val is None:
            c_in = None
        else:
            c_in = QuantumVariable(1)
            if c_in_val:
                x(c_in[0])

        carry_xor_block(target, dirty_ancillas, d, c_in)

        if c_in is None:
            return measure(target), measure(dirty_ancillas)
        return measure(target), measure(dirty_ancillas), measure(c_in)

    result = run()
    target_result, dirty_result = result[0], result[1]
    assert target_result is not None
    assert dirty_result is not None
    if c_in_val is not None:
        assert int(result[2]) == c_in_val
 
 
