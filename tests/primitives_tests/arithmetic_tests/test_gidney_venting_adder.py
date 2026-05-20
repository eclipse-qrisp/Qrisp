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
from qrisp import QuantumVariable, h, measure, multi_measurement, x
from qrisp.jasp import jaspify

from qrisp.alg_primitives.arithmetic.adders.gidney_venting_adder import (
    bit_inverted_controlled_gate,
    bit_inverted_cx,
    bit_inverted_cz,
    dual_zz_controlled_gate,
    dual_zz_controlled_x,
    dual_zz_controlled_z,
    zz_parity_controlled_gate,
    zz_parity_controlled_x,
    zz_parity_controlled_z,
)


BIT_INVERTED_CASES = [
    (0, False, 0),
    (1, False, 1),
    (0, True, 1),
    (1, True, 0),
]

BIT_INVERTED_WRAPPER_CASES = [
    ("cz", 1, False, 1),
    ("gate", 0, True, 1),
]

BIT_INVERTED_DYNAMIC_CASES = [
    ("cx", 1, True, 0),
    ("cz", 0, True, 1),
    ("gate", 1, False, 1),
]

ZZ_PARITY_CASES = [
    (0, 0, True),
    (1, 1, True),
    (0, 1, False),
    (1, 0, False),
]

ZZ_PARITY_WRAPPER_CASES = [
    ("z", 1, 1, True),
    ("gate", 0, 1, False),
]

ZZ_PARITY_DYNAMIC_CASES = [
    ("x", 1, 0, False),
    ("z", 0, 0, True),
    ("gate", 1, 1, True),
]

DUAL_ZZ_CASES = [
    (0, 0, 0, 0, True),
    (1, 1, 1, 1, True),
    (1, 0, 0, 0, False),
    (0, 0, 1, 0, False),
]

DUAL_ZZ_WRAPPER_CASES = [
    ("z", 1, 1, 1, 1, True),
    ("gate", 0, 0, 1, 0, False),
]

DUAL_ZZ_DYNAMIC_CASES = [
    ("x", 0, 0, 0, 0, True),
    ("z", 1, 1, 1, 1, True),
    ("gate", 1, 0, 0, 0, False),
]


@pytest.mark.parametrize("ctrl_val, b, expected_tgt", BIT_INVERTED_CASES)
def test_bit_inverted_cx_static(ctrl_val, b, expected_tgt):
    """Verify the full bit-inverted CX truth table in static mode."""
    ctrl = QuantumVariable(1)
    tgt = QuantumVariable(1)

    if ctrl_val:
        x(ctrl[0])

    bit_inverted_cx(ctrl[0], tgt[0], b)

    result = multi_measurement([ctrl, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (ctrl_val, expected_tgt)
    assert got == expected, f"expected {expected}, got {got}"


@pytest.mark.parametrize("op_name, ctrl_val, b, expected_tgt", BIT_INVERTED_WRAPPER_CASES)
def test_bit_inverted_wrappers_static(op_name, ctrl_val, b, expected_tgt):
    """Verify the CZ and generic controlled-gate wrappers on representative inputs."""
    ctrl = QuantumVariable(1)
    tgt = QuantumVariable(1)

    if ctrl_val:
        x(ctrl[0])

    if op_name == "cz":
        h(tgt[0])
        bit_inverted_cz(ctrl[0], tgt[0], b)
        h(tgt[0])
    else:
        bit_inverted_controlled_gate(ctrl[0], b, lambda: x(tgt[0]))

    result = multi_measurement([ctrl, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (ctrl_val, expected_tgt)
    assert got == expected, f"expected {expected}, got {got}"


@pytest.mark.parametrize("op_name, ctrl_val, b, expected_tgt", BIT_INVERTED_DYNAMIC_CASES)
def test_bit_inverted_dynamic(op_name, ctrl_val, b, expected_tgt):
    """Smoke-test bit-inverted helpers in dynamic mode."""

    @jaspify
    def run():
        ctrl = QuantumVariable(1)
        tgt = QuantumVariable(1)

        if ctrl_val:
            x(ctrl[0])

        if op_name == "cx":
            bit_inverted_cx(ctrl[0], tgt[0], b)
        elif op_name == "cz":
            h(tgt[0])
            bit_inverted_cz(ctrl[0], tgt[0], b)
            h(tgt[0])
        else:
            bit_inverted_controlled_gate(ctrl[0], b, lambda: x(tgt[0]))

        return measure(ctrl), measure(tgt)

    measured_ctrl, measured_tgt = run()
    assert (int(measured_ctrl), int(measured_tgt)) == (ctrl_val, expected_tgt)


@pytest.mark.parametrize("q0v, q1v, expected_tgt", ZZ_PARITY_CASES)
def test_zz_parity_controlled_x_static(q0v, q1v, expected_tgt):
    """Verify the full ZZ-parity controlled-X truth table in static mode."""
    q0 = QuantumVariable(1)
    q1 = QuantumVariable(1)
    tgt = QuantumVariable(1)

    if q0v:
        x(q0[0])
    if q1v:
        x(q1[0])

    zz_parity_controlled_x(q0[0], q1[0], tgt[0])

    result = multi_measurement([q0, q1, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (q0v, q1v, int(expected_tgt))
    assert got == expected, f"expected {expected}, got {got}"


@pytest.mark.parametrize("op_name, q0v, q1v, expected_tgt", ZZ_PARITY_WRAPPER_CASES)
def test_zz_parity_wrappers_static(op_name, q0v, q1v, expected_tgt):
    """Verify the ZZ-parity Z and generic-gate wrappers on representative inputs."""
    q0 = QuantumVariable(1)
    q1 = QuantumVariable(1)
    tgt = QuantumVariable(1)

    if q0v:
        x(q0[0])
    if q1v:
        x(q1[0])

    if op_name == "z":
        h(tgt[0])
        zz_parity_controlled_z(q0[0], q1[0], tgt[0])
        h(tgt[0])
    else:
        zz_parity_controlled_gate(q0[0], q1[0], lambda: x(tgt[0]))

    result = multi_measurement([q0, q1, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (q0v, q1v, int(expected_tgt))
    assert got == expected, f"expected {expected}, got {got}"


@pytest.mark.parametrize("op_name, q0v, q1v, expected_tgt", ZZ_PARITY_DYNAMIC_CASES)
def test_zz_parity_controlled_dynamic(op_name, q0v, q1v, expected_tgt):
    """Smoke-test ZZ-parity helpers in dynamic mode."""

    @jaspify
    def run():
        q0 = QuantumVariable(1)
        q1 = QuantumVariable(1)
        tgt = QuantumVariable(1)

        if q0v:
            x(q0[0])
        if q1v:
            x(q1[0])

        if op_name == "x":
            zz_parity_controlled_x(q0[0], q1[0], tgt[0])
        elif op_name == "z":
            h(tgt[0])
            zz_parity_controlled_z(q0[0], q1[0], tgt[0])
            h(tgt[0])
        else:
            zz_parity_controlled_gate(q0[0], q1[0], lambda: x(tgt[0]))

        return measure(q0), measure(q1), measure(tgt)

    measured_q0, measured_q1, measured_tgt = run()
    assert (int(measured_q0), int(measured_q1), int(measured_tgt)) == (
        q0v,
        q1v,
        int(expected_tgt),
    )


@pytest.mark.parametrize("a0v, a1v, b0v, b1v, expected_tgt", DUAL_ZZ_CASES)
def test_dual_zz_controlled_x_static(a0v, a1v, b0v, b1v, expected_tgt):
    """Verify the core dual-ZZ controlled-X behavior on representative inputs."""
    a0 = QuantumVariable(1)
    a1 = QuantumVariable(1)
    b0 = QuantumVariable(1)
    b1 = QuantumVariable(1)
    tgt = QuantumVariable(1)

    if a0v:
        x(a0[0])
    if a1v:
        x(a1[0])
    if b0v:
        x(b0[0])
    if b1v:
        x(b1[0])

    dual_zz_controlled_x(a0[0], a1[0], b0[0], b1[0], tgt[0])

    result = multi_measurement([a0, a1, b0, b1, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (a0v, a1v, b0v, b1v, int(expected_tgt))
    assert got == expected, f"expected {expected}, got {got}"


@pytest.mark.parametrize("op_name, a0v, a1v, b0v, b1v, expected_tgt", DUAL_ZZ_WRAPPER_CASES)
def test_dual_zz_wrappers_static(op_name, a0v, a1v, b0v, b1v, expected_tgt):
    """Verify the dual-ZZ Z and generic-gate wrappers on representative inputs."""
    a0 = QuantumVariable(1)
    a1 = QuantumVariable(1)
    b0 = QuantumVariable(1)
    b1 = QuantumVariable(1)
    tgt = QuantumVariable(1)

    if a0v:
        x(a0[0])
    if a1v:
        x(a1[0])
    if b0v:
        x(b0[0])
    if b1v:
        x(b1[0])

    if op_name == "z":
        h(tgt[0])
        dual_zz_controlled_z(a0[0], a1[0], b0[0], b1[0], tgt[0])
        h(tgt[0])
    else:
        dual_zz_controlled_gate(a0[0], a1[0], b0[0], b1[0], lambda: x(tgt[0]))

    result = multi_measurement([a0, a1, b0, b1, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (a0v, a1v, b0v, b1v, int(expected_tgt))
    assert got == expected, f"expected {expected}, got {got}"


def test_dual_zz_controlled_x_static_inputs_restored():
    """Verify dual-ZZ controlled X does not disturb the input controls."""
    a0 = QuantumVariable(1)
    a1 = QuantumVariable(1)
    b0 = QuantumVariable(1)
    b1 = QuantumVariable(1)
    tgt = QuantumVariable(1)

    x(a0[0])
    x(b1[0])
    dual_zz_controlled_x(a0[0], a1[0], b0[0], b1[0], tgt[0])

    result = multi_measurement([a0, a1, b0, b1, tgt])
    assert len(result) == 1
    got = tuple(int(v) for v in next(iter(result.keys())))
    expected = (1, 0, 0, 1, 0)
    assert got == expected, f"expected {expected}, got {got}"


@pytest.mark.parametrize("op_name, a0v, a1v, b0v, b1v, expected_tgt", DUAL_ZZ_DYNAMIC_CASES)
def test_dual_zz_controlled_dynamic(op_name, a0v, a1v, b0v, b1v, expected_tgt):
    """Smoke-test dual-ZZ helpers in dynamic mode."""

    @jaspify
    def run():
        a0 = QuantumVariable(1)
        a1 = QuantumVariable(1)
        b0 = QuantumVariable(1)
        b1 = QuantumVariable(1)
        tgt = QuantumVariable(1)

        if a0v:
            x(a0[0])
        if a1v:
            x(a1[0])
        if b0v:
            x(b0[0])
        if b1v:
            x(b1[0])

        if op_name == "x":
            dual_zz_controlled_x(a0[0], a1[0], b0[0], b1[0], tgt[0])
        elif op_name == "z":
            h(tgt[0])
            dual_zz_controlled_z(a0[0], a1[0], b0[0], b1[0], tgt[0])
            h(tgt[0])
        else:
            dual_zz_controlled_gate(a0[0], a1[0], b0[0], b1[0], lambda: x(tgt[0]))

        return measure(a0), measure(a1), measure(b0), measure(b1), measure(tgt)

    measured = run()
    assert tuple(int(value) for value in measured) == (
        a0v,
        a1v,
        b0v,
        b1v,
        int(expected_tgt),
    )