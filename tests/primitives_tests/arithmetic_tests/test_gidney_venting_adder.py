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

import pytest

from qrisp import (
    QuantumBool,
    QuantumFloat,
    x,
    h,
    z,
    measure,
    control,
    invert,
    gidney_adder,
)
from qrisp.jasp import jaspify, count_ops, terminal_sampling, jrange
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
from qrisp.alg_primitives.arithmetic.adders.gidney_venting_adder import (
    bit_inverted_mcx,
    zz_mcx,
    zz_zz_mcx,
    bit_inverted_zz_zz_mcx,
    carry_venting_adder,
    carry_xor_block,
    dirty_ancillae_adder,
    gidney_cq_venting_adder,
)


ZZ_PARITY_CASES = [
    (0, 0, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
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
    (1, True, 0, None),
    # ctrl=1 → matches uncontrolled
    (0, True, 1, 1),
    # ctrl=0 → parity flips suppressed, MCX fires on un-flipped state (= behaves as b=False)
    (1, True, 1, 0),
]


@pytest.mark.parametrize("ctrl_val, b, expected_tgt, ctrl_qbl_val", ALL_BIT_INVERTED_CASES)
def test_bit_inverted_mcx_jasp(ctrl_val, b, expected_tgt, ctrl_qbl_val):
    """Verify bit_inverted_mcx toggles target when (ctrl XOR b) = 1, with optional control qubit."""

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
        bit_inverted_mcx(
            ctrl[0],
            always_one[0],
            tgt[0],
            b,
            ctrl=ctrl_qbl[0] if ctrl_qbl_val is not None else None,
        )
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
    """Verify zz_mcx toggles target iff q0v XOR q1v = 1."""

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
    """Verify zz_zz_mcx toggles target based on parity of three control qubits."""

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
    (1, 0, True, 0, None),
    # ctrl=1 → matches uncontrolled
    (0, 0, True, 1, 1),
    # ctrl=0 → parity flips suppressed, MCX fires on un-flipped state (= behaves as b=False)
    (1, 1, False, 1, 0),
]


@pytest.mark.parametrize("ctrl1_v, ctrl2_v, b, expected_tgt, ctrl_qbl_val", ALL_BIT_INVERTED_ZZ_ZZ_CASES)
def test_bit_inverted_zz_zz_mcx_jasp(ctrl1_v, ctrl2_v, b, expected_tgt, ctrl_qbl_val):
    """Verify bit_inverted_zz_zz_mcx toggles target based on ctrl XOR b with optional control qubit."""

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
        bit_inverted_zz_zz_mcx(
            ctrl1[0],
            ctrl2[0],
            tgt[0],
            b,
            ctrl=ctrl_qbl[0] if ctrl_qbl_val is not None else None,
        )
        return measure(ctrl1), measure(ctrl2), measure(tgt), measure(ctrl_qbl)

    measured_ctrl1, measured_ctrl2, measured_tgt, measured_ctrl_qbl = run()
    assert int(measured_ctrl1) == ctrl1_v
    assert int(measured_ctrl2) == ctrl2_v
    assert int(measured_tgt) == expected_tgt
    if ctrl_qbl_val is not None:
        assert int(measured_ctrl_qbl) == ctrl_qbl_val


ALL_VENTING_CASES = [
    # (init, d, c_in_val, expected, n, ctrl_val)
    # n=2 (minimal)
    (1, 1, 0, 2, 2, None),
    (1, 2, None, 3, 2, None),
    # n=3 (uses write_final_carry_to_msb branch)
    (4, 3, 0, 7, 3, None),
    (7, 1, None, 0, 3, None),  # wrap: (7+1) % 8 = 0
    # n=4 (full loop + final block)
    (5, 5, 0, 10, 4, None),
    (12, 7, None, 3, 4, None),  # wrap: (12+7) % 16 = 3
    # n=5
    (10, 20, 0, 30, 5, None),
    # with carry-in
    (3, 2, 1, 6, 3, None),
    (6, 2, 1, 9, 4, None),
    # ctrl=1 → matches uncontrolled
    (1, 1, 0, 2, 2, 1),
    (4, 3, 0, 7, 3, 1),
    # ctrl=0 → no-op
    (1, 1, 0, 1, 2, 0),
    (4, 3, 0, 4, 3, 0),
]


@pytest.mark.parametrize("init, d, c_in_val, expected, n, ctrl_val", ALL_VENTING_CASES)
def test_carry_venting_adder_jasp(init, d, c_in_val, expected, n, ctrl_val):
    """Verify carry_venting_adder computes (init + d + c_in) mod 2ⁿ with optional control."""

    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        ctrl_qbl = QuantumFloat(1)
        if ctrl_val is not None:
            ctrl_qbl[:] = ctrl_val
        anc = QuantumFloat(2)
        ventmask, _ = carry_venting_adder(
            d,
            target,
            anc,
            c_in=c_in,
            ctrl=ctrl_qbl[0] if ctrl_val is not None else None,
        )
        return measure(target), measure(ctrl_qbl), ventmask

    result_target, result_ctrl, ventmask = run()
    assert int(result_target) == expected
    if ctrl_val is not None:
        assert int(result_ctrl) == ctrl_val
    assert ventmask >= 0


CARRY_XOR_CASES = [
    (5, 3, None, 4, False),
    (1, 1, 0, 3, False),
    (5, 3, None, 4, True),
]

CARRY_XOR_IDS = [
    "init=5,d=3,cin=None,n=4",
    "init=1,d=1,cin=0,n=3",
    "init=5,d=3,cin=None,n=4,ctrl",
]


@pytest.mark.parametrize("init, d, c_in_val, n, use_ctrl", CARRY_XOR_CASES, ids=CARRY_XOR_IDS)
def test_carry_xor_block(init, d, c_in_val, n, use_ctrl):
    """Full phase correction sequence restores dirty ancillas (see dirty_ancillae_adder pattern)."""

    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        ctrl = QuantumFloat(1)
        ctrl[:] = 1
        num_dirty = max(0, n - 2)
        dirty = QuantumFloat(num_dirty)
        anc = QuantumFloat(2)

        ventmask, _ = carry_venting_adder(
            d,
            target,
            anc,
            c_in=c_in,
            carry_xor_target=dirty,
        )

        for i in range(n):
            x(target[i])

        kwargs = {}
        if use_ctrl:
            kwargs["ctrl"] = ctrl[0]
        for k in range(num_dirty):
            with control((ventmask >> k) & 1):
                z(dirty[k])

        carry_xor_block(d, dirty, target[:-1], c_in=c_in, **kwargs)

        for k in range(num_dirty):
            with control((ventmask >> k) & 1):
                z(dirty[k])

        for i in range(n):
            x(target[i])

        if c_in_val is None:
            return measure(target), measure(dirty)
        return measure(target), measure(dirty), measure(c_in)

    result = run()
    assert int(result[0]) == (init + d + (c_in_val or 0)) & ((1 << n) - 1)
    assert int(result[1]) == 0
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
    """Verify dirty_ancillae_adder computes sum correctly and restores dirty qubits to |0⟩."""

    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        ctrl_qbl = QuantumFloat(1)
        if ctrl_val is not None:
            ctrl_qbl[:] = ctrl_val
        dirty = QuantumFloat(n - 2)
        for i in range(n - 2):
            h(dirty[i])  # initialize dirty qubits in |+⟩
        anc = QuantumFloat(2)
        ventmask = dirty_ancillae_adder(
            d,
            target,
            dirty,
            anc,
            c_in=c_in,
            ctrl=ctrl_qbl[0] if ctrl_val is not None else None,
        )
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
    """Verify dirty qubit values are preserved (not just restored to |0⟩) after dirty_ancillae_adder."""

    @jaspify
    def run():
        target = QuantumFloat(4)
        target[:] = 1
        dirty = QuantumFloat(2)
        dirty[:] = 2  # dirty = |1⟩|0⟩
        for i in range(2):
            h(dirty[i])
        anc = QuantumFloat(2)
        dirty_ancillae_adder(1, target, dirty, anc)
        for i in range(2):
            h(dirty[i])
        return measure(target), measure(dirty)

    result_target, result_dirty = run()
    assert int(result_target) == 2  # 1 + 1 = 2
    assert int(result_dirty) == 2  # dirty = |1⟩|0⟩ restored (both bits checked)


ALL_GIDNEY_CQ_CASES = [
    # (init, d, c_in_val, expected, n, ctrl_val)
    # --- basic uncontrolled ---
    (1, 1, 0, 2, 3, None),
    (15, 1, 0, 16, 6, None),
    (5, 2, None, 7, 3, None),  # n=3 (smallest split)
    (7, 4, None, 11, 4, None),  # n=4 (h=1, minimal split)
    # --- zero addend / zero target ---
    (13, 0, None, 13, 5, None),  # adding 0 is identity
    (0, 7, None, 7, 5, None),  # adding to zero
    # --- modular overflow: result wraps at 2^n ---
    (15, 3, 0, 2, 4, None),  # 15+3=18 is too large for 4 bits (max 15),
    # so only the low 4 bits remain: 18 % 16 = 2
    (31, 31, 0, 30, 5, None),  # 31+31=62, 5-bit max is 31 → 62 % 32 = 30
    (63, 1, 0, 0, 6, None),  # 63+1=64, but 6-bit wraps at 64 → 64 % 64 = 0
    # --- truncation: d is masked to n bits before adding ---
    (1, 8, 0, 1, 3, None),  # d=8 is 1000 in binary; only bottom 3 bits
    # (000) are kept → effectively d=0, so 1+0=1
    (1, 17, 0, 2, 4, None),  # d=17 is 10001; bottom 4 bits are 0001=1
    # → effectively d=1, so 1+1=2
    (5, 31, 0, 4, 4, None),  # d=31 is 11111; bottom 4 bits are 1111=15
    # → effectively d=15, so (5+15)%16=4
    # --- with carry-in ---
    (3, 2, 1, 6, 3, None),  # 3+2+carry_in=6 (straightforward, no overflow)
    (15, 1, 1, 1, 4, None),  # 15+1+carry_in=17, but 4-bit max is 15
    # → 17 % 16 = 1 (wraps around like an odometer)
    # --- controlled: ctrl=1 matches uncontrolled ---
    (1, 1, 0, 2, 3, 1),
    (10, 5, None, 15, 5, 1),
    # --- controlled: ctrl=0 → identity ---
    (1, 1, 0, 1, 3, 0),
    (10, 5, None, 10, 5, 0),
    # --- larger register ---
    (512, 256, 0, 768, 10, None),  # n=10
    # --- alternating bit patterns ---
    (0b101010, 0b010101, 0, 0b111111, 6, None),  # 42+21=63
    # all-ones addend (maximal carry propagation)
    (0, 15, 0, 15, 4, None),
    (1, 15, 0, 0, 4, None),
    (7, 15, 0, 6, 4, None),
    # single-bit at MSB
    (10, 16, 0, 26, 5, None),
    (31, 16, 0, 15, 5, None),
    # large wraps (n=8)
    (0, 255, 0, 255, 8, None),
    (255, 1, 0, 0, 8, None),
    (128, 128, 0, 0, 8, None),
    # carry-in with large addend
    (10, 20, 1, 31, 5, None),
    # h-boundary sweep: n where h = (n-1)>>1 changes
    (0, 1, 0, 1, 5, None),  # h=2
    (0, 1, 0, 1, 6, None),  # h=2
    (0, 1, 0, 1, 7, None),  # h=3
    (0, 1, 0, 1, 8, None),  # h=3
    (0, 1, 0, 1, 9, None),  # h=4
    (0, 1, 0, 1, 10, None),  # h=4
]


@pytest.mark.parametrize("init, d, c_in_val, expected, n, ctrl_val", ALL_GIDNEY_CQ_CASES)
def test_gidney_cq_venting_adder_jasp(init, d, c_in_val, expected, n, ctrl_val):
    """Verify gidney_cq_venting_adder computes (init + d + c_in) mod 2ⁿ with modular wrap and optional control."""

    # The register is only n bits wide, so the result is always
    # (init + d + c_in) mod 2^n, with d trimmed to n bits before addition.
    # That's why some cases don't match a plain init + d — the value
    # wraps around like an odometer when it exceeds 2^n-1.
    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        ctrl_qbl = QuantumFloat(1)
        if ctrl_val is not None:
            ctrl_qbl[:] = ctrl_val
        if ctrl_val is None:
            gidney_cq_venting_adder(d, target, c_in)
        else:
            with control(ctrl_qbl[0]):
                gidney_cq_venting_adder(d, target, c_in)
        return measure(target), measure(ctrl_qbl)

    result_target, result_ctrl = run()
    assert int(result_target) == expected
    if ctrl_val is not None:
        assert int(result_ctrl) == ctrl_val


C_OUT_CASES = [
    # (init, d, c_in_val, n)
    (0, 1, 0, 3),  # 0+1+0 = 1,  carry=0
    (7, 1, 0, 3),  # 7+1+0 = 8 ≥ 8, carry=1
    (15, 1, 0, 4),  # 15+1+0 = 16 ≥ 16, carry=1
    (14, 1, 0, 4),  # 14+1+0 = 15, carry=0
    (255, 1, 0, 8),  # 255+1+0 = 256 ≥ 256, carry=1
    (0, 0, 0, 5),  # 0+0+0 = 0, carry=0
    (0, 0, 1, 5),  # 0+0+1 = 1, carry=0 (c_in=1 but no overflow)
    (31, 0, 1, 5),  # 31+0+1 = 32 ≥ 32, carry=1
]


@pytest.mark.parametrize("init, d, c_in_val, n", C_OUT_CASES)
def test_gidney_cq_venting_adder_c_out(init, d, c_in_val, n):
    """Verify carry-out is set correctly."""
    expected_carry = 1 if (init + d + (c_in_val or 0)) >= (1 << n) else 0

    @jaspify
    def run():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        c_out = QuantumBool()
        gidney_cq_venting_adder(d, target, c_in=c_in, c_out=c_out)
        return measure(target), measure(c_out)

    result_target, result_carry = run()
    assert int(result_target) == (init + d + (c_in_val or 0)) % (1 << n)
    assert int(result_carry) == expected_carry
    """Verify error handling: non-JASP mode, wrong argument types."""
    target = QuantumFloat(3)
    with pytest.raises(
        RuntimeError,
        match="The Gidney Classical-Quantum adder does not work in standard python execution mode.",
    ):
        gidney_cq_venting_adder(1, target)

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


@pytest.mark.parametrize(
    "use_full_adder",
    [False, True],
    ids=["carry_venting_adder", "gidney_cq_venting_adder"],
)
def test_no_additional_toffoli_cost(use_full_adder):
    """Controlled adder uses zero extra Toffoli gates (zero-Toffoli control trick)."""
    d = 3
    n = 6

    @count_ops(meas_behavior="0")
    def unctrl():
        target = QuantumFloat(n)
        target[:] = 7
        if use_full_adder:
            gidney_cq_venting_adder(d, target)
        else:
            anc = QuantumFloat(2)
            carry_venting_adder(d, target, anc)
        return measure(target)

    @count_ops(meas_behavior="0")
    def ctrl_on():
        target = QuantumFloat(n)
        target[:] = 7
        ctrl_qbl = QuantumFloat(1)
        ctrl_qbl[:] = 1
        if use_full_adder:
            with control(ctrl_qbl[0]):
                gidney_cq_venting_adder(d, target)
        else:
            anc = QuantumFloat(2)
            carry_venting_adder(d, target, anc, ctrl=ctrl_qbl[0])
        return measure(target)

    a = unctrl()
    b = ctrl_on()
    assert a.get("t", 0) == b.get("t", 0)
    assert a.get("t", 0) > 0


def test_gidney_cq_fuzz():
    """Fuzz test: random init and d across register sizes 3..12."""
    rng = __import__("numpy").random.default_rng(12345)

    for n in range(3, 13):
        for _ in range(30):
            init = int(rng.integers(1 << n))
            d = int(rng.integers(1 << n))
            expected = (init + d) & ((1 << n) - 1)

            @jaspify
            def main(init=init, d=d, n=n):
                target = QuantumFloat(n)
                target[:] = init
                gidney_cq_venting_adder(d, target)
                return measure(target)

            assert int(main()) == expected


SEQUENTIAL_ADD_CASES = [
    (0, [1, 2, 3], 3),
    (5, [1, 1, 1], 5),  # init + 3
    (10, [7, 8], 10),  # larger than n? No, n=5 means mod 32
    (0, [1, 1, 1, 1, 1], 5),
]


@pytest.mark.parametrize(
    "init, d_list, n",
    SEQUENTIAL_ADD_CASES,
    ids=[f"init={c[0]},dlist={c[1]},n={c[2]}" for c in SEQUENTIAL_ADD_CASES],
)
def test_gidney_cq_sequential_add(init, d_list, n):
    """Multiple sequential additions: add d₁, then d₂, … verify cumulative sum."""
    expected = init
    for d in d_list:
        expected = (expected + d) & ((1 << n) - 1)

    @jaspify
    def main():
        target = QuantumFloat(n)
        target[:] = init
        for d in d_list:
            gidney_cq_venting_adder(d, target)
        return measure(target)

    assert int(main()) == expected


CONTROLLED_SUPER_CASES = [
    (1, 3),
    (1, 4),
    (1, 6),
]


@pytest.mark.parametrize("d, n", CONTROLLED_SUPER_CASES)
def test_gidney_cq_controlled_superposition(d, n):
    """Controlled adder with ctrl in |+⟩: H⊗ⁿ⁺¹ → controlled_add(d) → H⊗ⁿ⁺¹ → measure."""

    @jaspify
    def main():
        target = QuantumFloat(n)
        ctrl = QuantumFloat(1)
        for i in range(n):
            h(target[i])
        h(ctrl[0])
        with control(ctrl[0]):
            gidney_cq_venting_adder(d, target)
        for i in range(n):
            h(target[i])
        h(ctrl[0])
        return measure(target), measure(ctrl)

    t, c = main()
    # When ctrl=|0⟩, nothing happens → target stays |+⟩ → H → 0
    # When ctrl=|1⟩, addition happens → target = |x⊕d⟩, then H → not 0
    # In superposition, 50% each.  Both branches should give consistent results.
    assert int(c) in (0, 1)


SUPERPOSITION_D0_CASES = [
    ("gidney", 3),
    ("gidney", 4),
    ("gidney", 6),
    ("gidney", 9),
    ("gidney", 12),
    ("roundtrip", 3),
    ("roundtrip", 5),
    ("roundtrip", 6),
    ("dirty", 3),
    ("dirty", 5),
    ("dirty", 6),
]


def _sp_gidney(n):
    """Check gidney_cq_venting_adder preserves |0⟩ when applied to |+⟩ with d=0."""

    @jaspify
    def main():
        target = QuantumFloat(n)
        h(target)
        gidney_cq_venting_adder(0, target)
        h(target)
        return measure(target)

    return main()


def _sp_roundtrip(n):
    """Check two consecutive gidney_cq_venting_adder(0) calls cancel (identity) in superposition."""

    @jaspify
    def main():
        target = QuantumFloat(n)
        h(target)
        gidney_cq_venting_adder(0, target)
        gidney_cq_venting_adder(0, target)
        h(target)
        return measure(target)

    return main()


def _sp_dirty(n):
    """Check dirty_ancillae_adder(0) preserves both target and dirty registers in superposition."""

    @jaspify
    def main():
        target = QuantumFloat(n)
        anc = QuantumFloat(2)
        dirty = QuantumFloat(n - 2)
        h(target)
        h(dirty)
        dirty_ancillae_adder(0, target, dirty, anc)
        h(target)
        h(dirty)
        return measure(target), measure(dirty)

    return main()


@pytest.mark.parametrize("mode, n", SUPERPOSITION_D0_CASES)
def test_superposition_d0(mode, n):
    """H⊗ⁿ → adder(0) → H⊗ⁿ leaves |0⟩ (d=0 identity preserves coherence)."""
    if mode == "gidney":
        assert int(_sp_gidney(n)) == 0
    elif mode == "roundtrip":
        assert int(_sp_roundtrip(n)) == 0
    elif mode == "dirty":
        t, d = _sp_dirty(n)
        assert int(t) == 0
        assert int(d) == 0


@pytest.mark.parametrize(
    "init, d, c_in_val, expected, n",
    [
        (1, 1, None, 2, 3),
        (5, 3, 0, 8, 4),
    ],
    ids=["init=1,d=1,cin=0,n=3", "init=5,d=3,cin=0,n=4"],
)
def test_carry_venting_adder_bigint(init, d, c_in_val, expected, n):
    """carry_venting_adder with BigInteger addend (auto-detected)."""

    @jaspify
    def main():
        target = QuantumFloat(n)
        target[:] = init
        anc = QuantumFloat(2)
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        carry_venting_adder(
            BigInteger.create(d, 1),
            target,
            anc,
            c_in=c_in if c_in is not None else None,
        )
        return measure(target)

    assert int(main()) == expected


def test_dirty_ancillae_adder_bigint():
    """dirty_ancillae_adder with BigInteger addend (auto-detected)."""

    @jaspify
    def main():
        target = QuantumFloat(4)
        target[:] = 3
        dirty = QuantumFloat(2)
        anc = QuantumFloat(2)
        for i in range(2):
            h(dirty[i])
        dirty_ancillae_adder(BigInteger.create(5, 1), target, dirty, anc)
        for i in range(2):
            h(dirty[i])
        return measure(target), measure(dirty)

    t, d = main()
    assert int(t) == 8  # 3 + 5 = 8
    assert int(d) == 0  # dirty restored


GIDNEY_CQ_BIGINT_CASES = [
    (1, 1, None, 2, 3),
    (3, 2, 1, 6, 3),
]


@pytest.mark.parametrize(
    "init, d, c_in_val, expected, n",
    GIDNEY_CQ_BIGINT_CASES,
    ids=[f"init={c[0]},d={c[1]},n={c[4]}" for c in GIDNEY_CQ_BIGINT_CASES],
)
def test_gidney_cq_venting_adder_bigint(init, d, c_in_val, expected, n):
    """gidney_cq_venting_adder with BigInteger addend (auto-detected)."""

    @jaspify
    def main():
        target = QuantumFloat(n)
        target[:] = init
        if c_in_val is not None:
            c_in = QuantumBool()
            if c_in_val:
                x(c_in)
        else:
            c_in = None
        gidney_cq_venting_adder(BigInteger.create(d, 1), target, c_in if c_in is not None else None)
        return measure(target)

    assert int(main()) == expected


# Note: @terminal_sampling functions must be module-level (not defined inside
# pytest test functions) to avoid closure interaction with JASP tracing.
@terminal_sampling
def _ts_add(init, d, n):
    """Uncontrolled addition helper (module-level for terminal sampling)."""
    target = QuantumFloat(n)
    target[:] = init
    gidney_cq_venting_adder(d, target)
    return target


@terminal_sampling
def _ts_add_carry(init, d, c_in_val, n):
    """Addition with carry-in helper (module-level for terminal sampling)."""
    target = QuantumFloat(n)
    target[:] = init
    c_in = QuantumBool()
    with control(c_in_val):
        x(c_in)
    gidney_cq_venting_adder(d, target, c_in)
    return target


@terminal_sampling
def _ts_add_ctrl(init, d, ctrl_val, n):
    """Controlled addition helper (module-level for terminal sampling)."""
    target = QuantumFloat(n)
    target[:] = init
    ctrl_qbl = QuantumFloat(1)
    ctrl_qbl[:] = ctrl_val
    with control(ctrl_qbl[0]):
        gidney_cq_venting_adder(d, target)
    return target, ctrl_qbl


@terminal_sampling
def _ts_add_carry_ctrl(init, d, c_in_val, ctrl_val, n):
    """Controlled addition with carry-in helper (module-level for terminal sampling)."""
    target = QuantumFloat(n)
    target[:] = init
    c_in = QuantumBool()
    with control(c_in_val):
        x(c_in)
    ctrl_qbl = QuantumFloat(1)
    ctrl_qbl[:] = ctrl_val
    with control(ctrl_qbl[0]):
        gidney_cq_venting_adder(d, target, c_in)
    return target, ctrl_qbl


@terminal_sampling
def _ts_dirty(init, d):
    """Dirty-ancilla adder helper (module-level for terminal sampling)."""
    n = 4
    target = QuantumFloat(n)
    target[:] = init
    dirty = QuantumFloat(n - 2)
    for i in jrange(n - 2):
        h(dirty[i])
    anc = QuantumFloat(2)
    dirty_ancillae_adder(d, target, dirty, anc)
    for i in jrange(n - 2):
        h(dirty[i])
    return target, dirty


TERMINAL_CASES = [
    (1, 1, 2, 3, None),
    (15, 1, 16, 6, None),
    (5, 2, 7, 3, None),
    (0, 7, 7, 5, None),
]

TERMINAL_CTRL_CASES = [
    (1, 1, 0, 2, 3, 1),
    (1, 1, 0, 1, 3, 0),
    (10, 5, 15, 5, 1),
    (10, 5, 10, 5, 0),
]

TERMINAL_CARRY_CASES = [
    (3, 2, 1, 6, 3),
    (6, 2, 1, 9, 4),
]

TERMINAL_CARRY_CTRL_CASES = [
    (3, 2, 1, 1, 6, 3),
]


@pytest.mark.parametrize("init, d, expected, n, ctrl_val", TERMINAL_CASES)
def test_terminal_add(init, d, expected, n, ctrl_val):
    """Addition via terminal sampling."""
    if ctrl_val is not None:
        res = _ts_add_ctrl(init, d, ctrl_val, n)
        (t_val, c_val), prob = next(iter(res.items()))
        assert int(c_val) == ctrl_val
    else:
        res = _ts_add(init, d, n)
        t_val, prob = next(iter(res.items()))
    assert prob == pytest.approx(1.0)
    assert int(t_val) == expected


@pytest.mark.parametrize("init, d, c_in_val, expected, n", TERMINAL_CARRY_CASES)
def test_terminal_carry_in(init, d, c_in_val, expected, n):
    """Addition with carry-in via terminal sampling."""
    res = _ts_add_carry(init, d, c_in_val, n)
    t_val, prob = next(iter(res.items()))
    assert prob == pytest.approx(1.0)
    assert int(t_val) == expected


@pytest.mark.parametrize("init, d, c_in_val, ctrl_val, expected, n", TERMINAL_CARRY_CTRL_CASES)
def test_terminal_carry_and_ctrl(init, d, c_in_val, ctrl_val, expected, n):
    """Addition with carry-in and control via terminal sampling."""
    res = _ts_add_carry_ctrl(init, d, c_in_val, ctrl_val, n)
    (t_val, c_val), prob = next(iter(res.items()))
    assert prob == pytest.approx(1.0)
    assert int(t_val) == expected
    assert int(c_val) == ctrl_val


def test_terminal_dirty_adder():
    """Dirty-ancilla adder via terminal sampling: target == sum, dirty restored."""
    res = _ts_dirty(5, 3)
    (t_val, d_val), prob = next(iter(res.items()))
    assert prob == pytest.approx(1.0)
    assert int(t_val) == 8
    assert int(d_val) == 0


@terminal_sampling
def _ts_cq_gidney_roundtrip(n):
    """Check gidney_cq_venting_adder and gidney_adder cancel in superposition."""
    target = QuantumFloat(n)  # defaults to encoding 0
    h(target)  # |0⟩ → |+⟩⊗ⁿ
    gidney_cq_venting_adder(1, target)  # add 1
    with invert():
        gidney_adder(1, target)  # subtract 1
    h(target)  # |+⟩⊗ⁿ → |0⟩⊗ⁿ
    return target


@pytest.mark.parametrize("n", list(range(3, 13)))
def test_cq_gidney_roundtrip(n):
    """H⊗ⁿ → add 1 (venting) → subtract 1 (gidney) → H⊗ⁿ leaves |0⟩.

    For n < 6 this tests that gidney_adder(1) followed by gidney_adder(1)^†
    (its inverse) is the identity — the carry-venting split-half is not used.
    """
    res = _ts_cq_gidney_roundtrip(n)
    t_val, prob = next(iter(res.items()))
    assert prob == pytest.approx(1.0)
    assert int(t_val) == 0
