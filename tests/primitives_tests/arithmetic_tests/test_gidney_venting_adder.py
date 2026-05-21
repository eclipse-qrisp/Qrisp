"""
Unit tests for Gidney Figure 1 gate implementations (symbolic parity version).
These helpers are dynamic-mode only and rely on JASP tracing semantics.
"""

import pytest
from qrisp import QuantumVariable, x, measure, multi_measurement
from qrisp.jasp import jaspify

from qrisp.alg_primitives.arithmetic.adders.gidney_venting_adder import (
    bit_inverted_mcx,
    zz_mcx,
    zz_zz_mcx,
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


@pytest.mark.parametrize(
    "gate_kind",
    ["bit_inverted", "zz", "zz_zz"],
)
def test_dynamic_only_helpers_reject_static_mode(gate_kind):
    """All Figure 1 helpers are dynamic-only and must fail in static mode."""
    if gate_kind == "bit_inverted":
        ctrl = QuantumVariable(1)
        always_one = QuantumVariable(1)
        tgt = QuantumVariable(1)

        with pytest.raises(RuntimeError, match="requires JASP dynamic mode"):
            bit_inverted_mcx(ctrl[0], always_one[0], tgt[0], False)
        return

    if gate_kind == "zz":
        q0 = QuantumVariable(1)
        q1 = QuantumVariable(1)
        ctrl = QuantumVariable(1)
        tgt = QuantumVariable(1)

        with pytest.raises(RuntimeError, match="requires JASP dynamic mode"):
            zz_mcx(q0[0], q1[0], ctrl[0], tgt[0])
        return

    z_left = QuantumVariable(1)
    z_left_right = QuantumVariable(1)
    z_right = QuantumVariable(1)
    tgt = QuantumVariable(1)

    with pytest.raises(RuntimeError, match="requires JASP dynamic mode"):
        zz_zz_mcx(z_left[0], z_left_right[0], z_right[0], tgt[0])



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
