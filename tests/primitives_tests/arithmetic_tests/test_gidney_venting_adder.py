"""
Unit tests for Gidney Figure 1-4 gate implementations.
"""

import pytest
import jax.numpy as jnp
from qrisp import QuantumFloat, x, h, z, measure, control, multi_measurement, cx
from qrisp.jasp import jaspify
from qrisp.alg_primitives.arithmetic.adders.gidney_venting_adder import (
    _extract_bit,
    bit_inverted_mcx,
    zz_mcx,
    zz_zz_mcx,
    bit_inverted_zz_zz_mcx,
    carry_venting_adder,
    carry_xor_block,
    dirty_ancillae_adder,
    gidney_cq_venting_adder,
)

EXTRACT_BIT_CASES = [
    (0, 0, 0),
    (10, 1, 1),
    (15, 0, 1),
    (15, 3, 1),
]

@pytest.mark.parametrize("val, bit, expected", EXTRACT_BIT_CASES)
def test_extract_bit_int(val, bit, expected):
    # Verify bit extraction via shift-and-mask for small integers
    assert bool(_extract_bit(val, bit, False)) == bool(expected)

EXTRACT_BIT_BIGINT_CASES = [
    (0, True),
    (2, True),
]

@pytest.mark.parametrize("idx, expected", EXTRACT_BIT_BIGINT_CASES)
def test_extract_bit_bigint(idx, expected):
    class _MockBigInt:
        def __init__(self, bits):
            self.bits = bits
        def get_bit(self, i):
            return self.bits[i]

    b = _MockBigInt([1, 0, 1, 1])
    # Verify bit extraction via the BigInteger interface
    assert bool(_extract_bit(b, idx, True)) == expected


ZZ_PARITY_CASES = [
    (0, 0, 0),
    (1, 1, 0),
]

DUAL_ZZ_CASES = [
    (0, 0, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 1),
]

ALL_BIT_INVERTED_CASES = [
    # (ctrl_val, b, expected_tgt, ctrl_qbl_val)
    # uncontrolled
    (0, False, 0, None),
    (1, True,  0, None),
    # ctrl=1 → matches uncontrolled
    (0, True,  1, 1),
    # ctrl=0 → parity flips suppressed, MCX fires on un-flipped state (= behaves as b=False)
    (1, True,  1, 0),
]

@pytest.mark.parametrize("ctrl_val, b, expected_tgt, ctrl_qbl_val", ALL_BIT_INVERTED_CASES)
def test_bit_inverted_mcx_jasp(ctrl_val, b, expected_tgt, ctrl_qbl_val):
    @jaspify
    def run():
        ctrl = QuantumFloat(1)
        ctrl[:] = ctrl_val
        always_one = QuantumFloat(1)
        always_one[:] = 1  # ensure simple_ctrl is |1⟩ so MCX fires based on parity
        tgt = QuantumFloat(1)
        ctrl_qbl = QuantumFloat(1)
        if ctrl_qbl_val is not None:
            ctrl_qbl[:] = ctrl_qbl_val
        bit_inverted_mcx(ctrl[0], always_one[0], tgt[0], b, ctrl=ctrl_qbl[0] if ctrl_qbl_val is not None else None)
        return measure(ctrl), measure(always_one), measure(tgt), measure(ctrl_qbl)
    measured_ctrl, measured_always_one, measured_tgt, measured_ctrl_qbl = run()
    assert int(measured_ctrl) == ctrl_val
    assert int(measured_always_one) == 1
    # target toggles when (ctrl_val XOR b) = 1; always_one=1 ensures simple_ctrl fires
    assert int(measured_tgt) == expected_tgt
    if ctrl_qbl_val is not None:
        assert int(measured_ctrl_qbl) == ctrl_qbl_val


@pytest.mark.parametrize("q0v, q1v, expected_tgt", ZZ_PARITY_CASES)
def test_zz_mcx_jasp(q0v, q1v, expected_tgt):
    @jaspify
    def run():
        q0 = QuantumFloat(1)
        q0[:] = q0v
        q1 = QuantumFloat(1)
        q1[:] = q1v
        ctrl = QuantumFloat(1)
        ctrl[:] = 1  # ensure control is |1⟩
        tgt = QuantumFloat(1)
        zz_mcx(q0[0], q1[0], ctrl[0], tgt[0])
        return measure(q0), measure(q1), measure(ctrl), measure(tgt)
    measured_q0, measured_q1, measured_ctrl, measured_tgt = run()
    assert int(measured_q0) == q0v
    assert int(measured_q1) == q1v
    assert int(measured_ctrl) == 1
    # target toggles iff q0v XOR q1v = 1
    assert int(measured_tgt) == expected_tgt


@pytest.mark.parametrize("z_left_v, z_left_right_v, z_right_v, expected_tgt", DUAL_ZZ_CASES)
def test_zz_zz_mcx_jasp(z_left_v, z_left_right_v, z_right_v, expected_tgt):
    @jaspify
    def run():
        z_left = QuantumFloat(1)
        z_left[:] = z_left_v
        z_left_right = QuantumFloat(1)
        z_left_right[:] = z_left_right_v
        z_right = QuantumFloat(1)
        z_right[:] = z_right_v
        tgt = QuantumFloat(1)
        zz_zz_mcx(z_left[0], z_left_right[0], z_right[0], tgt[0])
        return measure(z_left), measure(z_left_right), measure(z_right), measure(tgt)
    measured_z_left, measured_z_left_right, measured_z_right, measured_tgt = run()
    assert int(measured_z_left) == z_left_v
    assert int(measured_z_left_right) == z_left_right_v
    assert int(measured_z_right) == z_right_v
    assert int(measured_tgt) == expected_tgt


ALL_BIT_INVERTED_ZZ_ZZ_CASES = [
    # (ctrl1_v, ctrl2_v, b, expected_tgt, ctrl_qbl_val)
    # uncontrolled
    (0, 0, False, 0, None),
    (1, 1, False, 1, None),
    (1, 0, True,  0, None),
    # ctrl=1 → matches uncontrolled
    (0, 0, True,  1, 1),
    # ctrl=0 → parity flips suppressed, MCX fires on un-flipped state (= behaves as b=False)
    (1, 1, False, 1, 0),
]

@pytest.mark.parametrize("ctrl1_v, ctrl2_v, b, expected_tgt, ctrl_qbl_val", ALL_BIT_INVERTED_ZZ_ZZ_CASES)
def test_bit_inverted_zz_zz_mcx_jasp(ctrl1_v, ctrl2_v, b, expected_tgt, ctrl_qbl_val):
    @jaspify
    def run():
        ctrl1 = QuantumFloat(1)
        ctrl1[:] = ctrl1_v
        ctrl2 = QuantumFloat(1)
        ctrl2[:] = ctrl2_v
        tgt = QuantumFloat(1)
        ctrl_qbl = QuantumFloat(1)
        if ctrl_qbl_val is not None:
            ctrl_qbl[:] = ctrl_qbl_val
        bit_inverted_zz_zz_mcx(ctrl1[0], ctrl2[0], tgt[0], b, ctrl=ctrl_qbl[0] if ctrl_qbl_val is not None else None)
        return measure(ctrl1), measure(ctrl2), measure(tgt), measure(ctrl_qbl)
    measured_ctrl1, measured_ctrl2, measured_tgt, measured_ctrl_qbl = run()
    assert int(measured_ctrl1) == ctrl1_v
    assert int(measured_ctrl2) == ctrl2_v
    assert int(measured_tgt) == expected_tgt
    if ctrl_qbl_val is not None:
        assert int(measured_ctrl_qbl) == ctrl_qbl_val



ALL_VENTING_CASES = [
    # (init, d, c_in_val, expected, n, ctrl_val)
    # uncontrolled
    (1, 1, 0, 2, 2, None),
    (1, 2, None, 3, 2, None),
    # ctrl=1 → matches uncontrolled
    (1, 1, 0, 2, 2, 1),
    # ctrl=0 → no-op
    (1, 1, 0, 1, 2, 0),
]

@pytest.mark.parametrize("init, d, c_in_val, expected, n, ctrl_val", ALL_VENTING_CASES)
def test_carry_venting_adder_jasp(init, d, c_in_val, expected, n, ctrl_val):
    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumFloat(1)
            c_in[:] = c_in_val
        else:
            c_in = None
        ctrl_qbl = QuantumFloat(1)
        if ctrl_val is not None:
            ctrl_qbl[:] = ctrl_val
        anc = QuantumFloat(2)
        ventmask = carry_venting_adder(d, target, anc, c_in=c_in if c_in is None else c_in[0], ctrl=ctrl_qbl[0] if ctrl_val is not None else None)
        return measure(target), measure(ctrl_qbl), ventmask
    result_target, result_ctrl, ventmask = run()
    assert int(result_target) == expected
    if ctrl_val is not None:
        assert int(result_ctrl) == ctrl_val
    assert ventmask >= 0



CARRY_XOR_CASES = [
    (5, 3, None),
]

@pytest.mark.parametrize("init, d, c_in_val", CARRY_XOR_CASES)
def test_carry_xor_block_jasp(init, d, c_in_val):
    @jaspify
    def run():
        target = QuantumFloat(4)
        target[:] = init
        dirty_ancillas = QuantumFloat(4)
        if c_in_val is None:
            c_in = None
        else:
            c_in = QuantumFloat(1)
            c_in[:] = c_in_val
        carry_xor_block(d, dirty_ancillas, target, c_in if c_in is None else c_in[0])
        if c_in is None:
            return measure(target), measure(dirty_ancillas)
        return measure(target), measure(dirty_ancillas), measure(c_in)
    result = run()
    target_result, dirty_result = result[0], result[1]
    assert target_result is not None
    assert dirty_result is not None
    if c_in_val is not None:
        assert int(result[2]) == c_in_val



ALL_DIRTY_ADD_CASES = [
    # (init, d, c_in_val, expected, n, ctrl_val)
    # uncontrolled
    (1, 1, 0, 2, 3, None),
    (7, 2, 1, 10, 5, None),
    # ctrl=1 → matches uncontrolled
    (1, 1, 0, 2, 3, 1),
    # ctrl=0 → no-op
    (1, 1, 0, 1, 3, 0),
]

@pytest.mark.parametrize("init, d, c_in_val, expected, n, ctrl_val", ALL_DIRTY_ADD_CASES)
def test_dirty_ancillae_adder_jasp(init, d, c_in_val, expected, n, ctrl_val):
    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumFloat(1)
            c_in[:] = c_in_val
        else:
            c_in = None
        ctrl_qbl = QuantumFloat(1)
        if ctrl_val is not None:
            ctrl_qbl[:] = ctrl_val
        dirty = QuantumFloat(n - 2)
        for i in range(n - 2):
            h(dirty[i])  # initialize dirty qubits in |+⟩
        anc = QuantumFloat(2)
        ventmask = dirty_ancillae_adder(d, target, dirty, anc, c_in=c_in if c_in is None else c_in[0], ctrl=ctrl_qbl[0] if ctrl_val is not None else None)
        for i in range(n - 2):
            h(dirty[i])  # undo |+⟩ → measure in Z-basis
        return measure(target), measure(dirty), measure(ctrl_qbl), ventmask
    result_target, result_dirty, result_ctrl, ventmask = run()
    assert int(result_target) == expected
    assert int(result_dirty) == 0  # dirty qubits restored to |0⟩
    if ctrl_val is not None:
        assert int(result_ctrl) == ctrl_val
    assert ventmask >= 0


def test_dirty_ancillae_adder_preserves_dirty():
    @jaspify
    def run():
        target = QuantumFloat(4)
        target[:] = 1
        dirty = QuantumFloat(2)
        dirty[:] = 1
        for i in range(2):
            h(dirty[i])
        anc = QuantumFloat(2)
        dirty_ancillae_adder(1, target, dirty, anc)
        for i in range(2):
            h(dirty[i])
        return measure(target), measure(dirty)
    result_target, result_dirty = run()
    assert int(result_target) == 2   # 1 + 1 = 2
    assert int(result_dirty) & 1 == 1  # dirty[0] preserved (phase-corrected back to original)



ALL_GIDNEY_CQ_CASES = [
    # (init, d, c_in_val, expected, n, ctrl_val)
    # --- basic uncontrolled ---
    (1, 1, 0, 2, 3, None),
    (15, 1, 0, 16, 6, None),
    (5, 2, None, 7, 3, None),       # n=3 (smallest split)
    (7, 4, None, 11, 4, None),      # n=4 (h=1, minimal split)
    # --- zero addend / zero target ---
    (13, 0, None, 13, 5, None),     # adding 0 is identity
    (0, 7, None, 7, 5, None),       # adding to zero
    # --- modular overflow: result wraps at 2^n ---
    (15, 3, 0, 2, 4, None),         # 15+3=18 is too large for 4 bits (max 15),
                                     # so only the low 4 bits remain: 18 % 16 = 2
    (31, 31, 0, 30, 5, None),       # 31+31=62, 5-bit max is 31 → 62 % 32 = 30
    (63, 1, 0, 0, 6, None),         # 63+1=64, but 6-bit wraps at 64 → 64 % 64 = 0
    # --- truncation: d is masked to n bits before adding ---
    (1, 8, 0, 1, 3, None),          # d=8 is 1000 in binary; only bottom 3 bits
                                     # (000) are kept → effectively d=0, so 1+0=1
    (1, 17, 0, 2, 4, None),         # d=17 is 10001; bottom 4 bits are 0001=1
                                     # → effectively d=1, so 1+1=2
    (5, 31, 0, 4, 4, None),         # d=31 is 11111; bottom 4 bits are 1111=15
                                     # → effectively d=15, so (5+15)%16=4
    # --- with carry-in ---
    (3, 2, 1, 6, 3, None),          # 3+2+carry_in=6 (straightforward, no overflow)
    (15, 1, 1, 1, 4, None),         # 15+1+carry_in=17, but 4-bit max is 15
                                     # → 17 % 16 = 1 (wraps around like an odometer)
    # --- controlled: ctrl=1 matches uncontrolled ---
    (1, 1, 0, 2, 3, 1),
    (10, 5, None, 15, 5, 1),
    # --- controlled: ctrl=0 → identity ---
    (1, 1, 0, 1, 3, 0),
    (10, 5, None, 10, 5, 0),
    # --- larger register ---
    (512, 256, 0, 768, 10, None),   # n=10
    # --- alternating bit patterns ---
    (0b101010, 0b010101, 0, 0b111111, 6, None),  # 42+21=63
]

@pytest.mark.parametrize("init, d, c_in_val, expected, n, ctrl_val", ALL_GIDNEY_CQ_CASES)
def test_gidney_cq_venting_adder_jasp(init, d, c_in_val, expected, n, ctrl_val):
    # The register is only n bits wide, so the result is always
    # (init + d + c_in) mod 2^n, with d trimmed to n bits before addition.
    # That's why some cases don't match a plain init + d — the value
    # wraps around like an odometer when it exceeds 2^n-1.
    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumFloat(1)
            c_in[:] = c_in_val
        else:
            c_in = None
        ctrl_qbl = QuantumFloat(1)
        if ctrl_val is not None:
            ctrl_qbl[:] = ctrl_val
        if ctrl_val is None:
            ventmask = gidney_cq_venting_adder(d, target, c_in if c_in is None else c_in[0])
        else:
            with control(ctrl_qbl[0]):
                ventmask = gidney_cq_venting_adder(d, target, c_in if c_in is None else c_in[0])
        return measure(target), measure(ctrl_qbl), ventmask
    result_target, result_ctrl, ventmask = run()
    assert int(result_target) == expected
    if ctrl_val is not None:
        assert int(result_ctrl) == ctrl_val
    assert ventmask >= 0


# Phase-indexing tests for CZ correction
PHASE_Z_CASES = [
    ("correct > > k",                  2, 3, lambda k: k,       2),
    ("wrong > > (k+1)",                2, 3, lambda k: k + 1,   1),
    ("carry_mid at bit h-1",           2, 2, lambda k: k,       2),
]

@pytest.mark.parametrize("desc, ventmask, nq, shift, expected", PHASE_Z_CASES,
                         ids=[c[0] for c in PHASE_Z_CASES])
def test_phase_z_bit_indexing(desc, ventmask, nq, shift, expected):
    qv = QuantumFloat(nq)
    for i in range(nq):
        h(qv[i])

    # Apply CZ corrections based on ventmask bits at shifted positions
    for k in range(nq):
        with control(bool((ventmask >> shift(k)) & 1)):
            z(qv[k])

    for i in range(nq):
        h(qv[i])

    res = multi_measurement([qv])
    val, = next(iter(res.keys()))
    assert val == expected


def test_non_jasp_mode_raises_error():
    """Verify that calling gidney_cq_venting_adder outside JASP mode raises RuntimeError."""
    target = QuantumFloat(3)
    with pytest.raises(RuntimeError, match="The Gidney Classical-Quantum adder does not work in standard python execution mode."):
        gidney_cq_venting_adder(1, target)


def test_type_errors():
    """Verify that type errors are raised for invalid argument types (dynamic mode)."""

    @jaspify
    def first_arg_type_error():
        qv = QuantumFloat(3)
        gidney_cq_venting_adder(qv, qv)

    @jaspify
    def second_arg_type_error():
        gidney_cq_venting_adder(1, 2)

    with pytest.raises(TypeError, match="The first argument must be a classical integer"):
        first_arg_type_error()
    with pytest.raises(TypeError, match="The second argument must be a QuantumVariable"):
        second_arg_type_error()


def test_no_additional_toffoli_cost():
    """Verify that controlled addition uses zero extra Toffoli gates.

    With the zero-Toffoli control trick, the controlled adder uses the same
    number of T and H gates (Toffoli-equivalent resources) as the uncontrolled
    adder. Only CX count may differ (extra parity-flip CX gates).
    """
    from qrisp.jasp import count_ops

    @count_ops(meas_behavior="0")
    def unctrl():
        target = QuantumFloat(5)
        target[:] = 7
        anc = QuantumFloat(2)
        carry_venting_adder(3, target, anc)
        return measure(target)

    @count_ops(meas_behavior="0")
    def ctrl_on():
        target = QuantumFloat(5)
        target[:] = 7
        anc = QuantumFloat(2)
        ctrl_qbl = QuantumFloat(1)
        ctrl_qbl[:] = 1
        carry_venting_adder(3, target, anc, ctrl=ctrl_qbl[0])
        return measure(target)

    a = unctrl()
    b = ctrl_on()

    # count_ops captures gates after compilation, which decomposes
    # MCX into basis gates (T, H, CX). No "mcx" key appears in the
    # output, so we compare T count as the Toffoli-equivalent metric.
    assert a.get("t", 0) == b.get("t", 0)
    assert a.get("t", 0) > 0
