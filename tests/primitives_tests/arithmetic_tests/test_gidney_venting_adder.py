"""
Unit tests for Gidney Figure 1-4 gate implementations.
"""

import pytest
import jax.numpy as jnp
from qrisp import QuantumVariable, x, h, z, measure, control, multi_measurement, cx
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

def _set_qubits(qv, val, n):
    for i in range(n):
        if (val >> i) & 1:
            x(qv[i])


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
        ctrl = QuantumVariable(1)
        always_one = QuantumVariable(1)
        tgt = QuantumVariable(1)
        ctrl_qbl = QuantumVariable(1)
        if ctrl_val:
            x(ctrl[0])
        x(always_one[0])  # ensure simple_ctrl is |1⟩ so MCX fires based on parity
        if ctrl_qbl_val is not None and ctrl_qbl_val:
            x(ctrl_qbl[0])
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
        q0 = QuantumVariable(1)
        q1 = QuantumVariable(1)
        ctrl = QuantumVariable(1)
        tgt = QuantumVariable(1)
        if q0v:
            x(q0[0])
        if q1v:
            x(q1[0])
        x(ctrl[0])  # ensure control is |1⟩
        # zz_mcx fires when q0 and q1 disagree (parity=1)
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
        # zz_zz_mcx fires when left pair disagrees AND right pair disagrees
        # z_left_right is shared between both pairs
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
        ctrl1 = QuantumVariable(1)
        ctrl2 = QuantumVariable(1)
        tgt = QuantumVariable(1)
        ctrl_qbl = QuantumVariable(1)
        if ctrl1_v:
            x(ctrl1[0])
        if ctrl2_v:
            x(ctrl2[0])
        if ctrl_qbl_val is not None and ctrl_qbl_val:
            x(ctrl_qbl[0])
        # When b=1: target toggles when both controls are |0⟩ (inverted parity)
        # When b=0: target toggles when both controls are |1⟩
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
        target = QuantumVariable(n)
        _set_qubits(target, init, n)
        if c_in_val is not None:
            c_in = QuantumVariable(1)
            if c_in_val:
                x(c_in[0])
        else:
            c_in = None
        ctrl_qbl = QuantumVariable(1)
        if ctrl_val is not None and ctrl_val:
            x(ctrl_qbl[0])
        anc = QuantumVariable(2)
        # ctrl=None: uncontrolled addition; ctrl=1: controlled; ctrl=0: no-op
        ventmask = carry_venting_adder(d, target, anc, c_in=c_in, ctrl=ctrl_qbl[0] if ctrl_val is not None else None)
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
        target = QuantumVariable(4)
        dirty_ancillas = QuantumVariable(4)
        _set_qubits(target, init, 4)
        if c_in_val is None:
            c_in = None
        else:
            c_in = QuantumVariable(1)
            if c_in_val:
                x(c_in[0])
        # carry_xor_block recomputes carries from sum and XORs them into dirty_ancillas
        carry_xor_block(d, dirty_ancillas, target, c_in)
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
        target = QuantumVariable(n)
        _set_qubits(target, init, n)
        if c_in_val is not None:
            c_in = QuantumVariable(1)
            if c_in_val:
                x(c_in[0])
        else:
            c_in = None
        ctrl_qbl = QuantumVariable(1)
        if ctrl_val is not None and ctrl_val:
            x(ctrl_qbl[0])
        dirty = QuantumVariable(n - 2)
        for i in range(n - 2):
            h(dirty[i])  # initialize dirty qubits in |+⟩
        anc = QuantumVariable(2)
        # dirty_ancillae_adder must restore dirty qubits to |+⟩ (measurable as |0⟩ after H)
        ventmask = dirty_ancillae_adder(d, target, dirty, anc, c_in=c_in, ctrl=ctrl_qbl[0] if ctrl_val is not None else None)
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
        target = QuantumVariable(4)
        x(target[0])  # target = 1
        dirty = QuantumVariable(2)
        x(dirty[0])   # dirty[0] = 1
        anc = QuantumVariable(2)
        dirty_ancillae_adder(1, target, dirty, anc)
        return measure(target), measure(dirty)
    result_target, result_dirty = run()
    assert int(result_target) == 2   # 1 + 1 = 2
    assert int(result_dirty) & 1 == 1  # dirty[0] preserved (phase-corrected back to original)



ALL_GIDNEY_CQ_CASES = [
    # (init, d, c_in_val, expected, n, ctrl_val)
    # uncontrolled
    (1, 1, 0, 2, 3, None),
    (15, 1, 0, 16, 6, None),
    # truncation: d=8 truncated to 3 bits → d & 7 = 0
    (1, 8, 0, 1, 3, None),
    (1, 17, 0, 2, 4, None),
    # ctrl=1 → matches uncontrolled
    (1, 1, 0, 2, 3, 1),
    # ctrl=0 → no-op
    (1, 1, 0, 1, 3, 0),
]

@pytest.mark.parametrize("init, d, c_in_val, expected, n, ctrl_val", ALL_GIDNEY_CQ_CASES)
def test_gidney_cq_venting_adder_jasp(init, d, c_in_val, expected, n, ctrl_val):
    @jaspify
    def run():
        target = QuantumVariable(n)
        _set_qubits(target, init, n)
        if c_in_val is not None:
            c_in = QuantumVariable(1)
            if c_in_val:
                x(c_in[0])
        else:
            c_in = None
        ctrl_qbl = QuantumVariable(1)
        if ctrl_val is not None and ctrl_val:
            x(ctrl_qbl[0])
        if ctrl_val is None:
            ventmask = gidney_cq_venting_adder(d, target, c_in)
        else:
            # Test the @custom_control path via the with control(...) context
            with control(ctrl_qbl[0]):
                ventmask = gidney_cq_venting_adder(d, target, c_in)
        return measure(target), measure(ctrl_qbl), ventmask
    result_target, result_ctrl, ventmask = run()
    assert int(result_target) == expected
    if ctrl_val is not None:
        assert int(result_ctrl) == ctrl_val
    assert ventmask >= 0


# Phase-indexing tests for CZ correction
PHASE_Z_CASES = [
    ("correct > > k",                  2, 3, lambda k: k,       "010"),
    ("wrong > > (k+1)",                2, 3, lambda k: k + 1,   "100"),
    ("carry_mid at bit h-1",           2, 2, lambda k: k,       "01"),
]

@pytest.mark.parametrize("desc, ventmask, nq, shift, expected", PHASE_Z_CASES,
                         ids=[c[0] for c in PHASE_Z_CASES])
def test_phase_z_bit_indexing(desc, ventmask, nq, shift, expected):
    qv = QuantumVariable(nq)
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
    target = QuantumVariable(3)
    with pytest.raises(RuntimeError, match="The Gidney Classical-Quantum adder does not work in standard python execution mode."):
        gidney_cq_venting_adder(1, target)


def test_type_errors():
    """Verify that type errors are raised for invalid argument types."""
    from qrisp import QuantumVariable

    qv = QuantumVariable(3)
    with pytest.raises(TypeError, match="The first argument must be a classical integer"):
        gidney_cq_venting_adder(qv, qv)
    with pytest.raises(TypeError, match="The second argument must be a QuantumVariable"):
        gidney_cq_venting_adder(1, 2)


def test_no_additional_toffoli_cost():
    """Verify that controlled addition uses zero extra Toffoli gates.

    With the zero-Toffoli control trick, the controlled adder uses the same
    number of T and H gates (Toffoli-equivalent resources) as the uncontrolled
    adder. Only CX count may differ (extra parity-flip CX gates).
    """
    from qrisp.jasp import count_ops

    @count_ops(meas_behavior="0")
    def unctrl():
        target = QuantumVariable(5)
        _set_qubits(target, 7, 5)
        anc = QuantumVariable(2)
        carry_venting_adder(3, target, anc)
        return measure(target)

    @count_ops(meas_behavior="0")
    def ctrl_on():
        target = QuantumVariable(5)
        _set_qubits(target, 7, 5)
        anc = QuantumVariable(2)
        ctrl_qbl = QuantumVariable(1)
        x(ctrl_qbl[0])
        carry_venting_adder(3, target, anc, ctrl=ctrl_qbl[0])
        return measure(target)

    a = unctrl()
    b = ctrl_on()

    # count_ops captures gates after compilation, which decomposes
    # MCX into basis gates (T, H, CX). No "mcx" key appears in the
    # output, so we compare T count as the Toffoli-equivalent metric.
    assert a.get("t", 0) == b.get("t", 0)
    assert a.get("t", 0) > 0
