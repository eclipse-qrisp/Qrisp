# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for the gate application functions in Qrisp."""

import pytest

from qrisp import QuantumVariable
from qrisp.core.gate_application_functions import (
    barrier,
    cp,
    crz,
    cx,
    cy,
    cz,
    gphase,
    h,
    mcp,
    mcx,
    mcz,
    p,
    rx,
    rxx,
    ry,
    ryy,
    rz,
    rzz,
    s,
    s_dg,
    sx,
    sx_dg,
    swap,
    t,
    t_dg,
    u3,
    x,
    xxyy,
    y,
    z,
)

# Every gate-application function that was changed to return None instead of
# echoing back its qubit arguments. Each entry applies the gate to fresh
# qubits of a 6-qubit QuantumVariable, which is enough for every arity here.
GATE_CALLS = {
    "cx": lambda qv: cx(qv[0], qv[1]),
    "cy": lambda qv: cy(qv[0], qv[1]),
    "cz": lambda qv: cz(qv[0], qv[1]),
    "h": lambda qv: h(qv[0]),
    "x": lambda qv: x(qv[0]),
    "y": lambda qv: y(qv[0]),
    "z": lambda qv: z(qv[0]),
    "mcx": lambda qv: mcx(qv[:3], qv[3]),
    "mcz": lambda qv: mcz(qv[:4]),
    "mcp": lambda qv: mcp(0.3, qv[:4]),
    "p": lambda qv: p(0.3, qv[0]),
    "cp": lambda qv: cp(0.3, qv[0], qv[1]),
    "rx": lambda qv: rx(0.3, qv[0]),
    "ry": lambda qv: ry(0.3, qv[0]),
    "rz": lambda qv: rz(0.3, qv[0]),
    "crz": lambda qv: crz(0.3, qv[0], qv[1]),
    "s": lambda qv: s(qv[0]),
    "t": lambda qv: t(qv[0]),
    "s_dg": lambda qv: s_dg(qv[0]),
    "t_dg": lambda qv: t_dg(qv[0]),
    "sx": lambda qv: sx(qv[0]),
    "sx_dg": lambda qv: sx_dg(qv[0]),
    "gphase": lambda qv: gphase(0.3, qv[0]),
    "xxyy": lambda qv: xxyy(0.3, 0.2, qv[0], qv[1]),
    "rzz": lambda qv: rzz(0.3, qv[0], qv[1]),
    "rxx": lambda qv: rxx(0.3, qv[0], qv[1]),
    "ryy": lambda qv: ryy(0.3, qv[0], qv[1]),
    "u3": lambda qv: u3(0.1, 0.2, 0.3, qv[0]),
    "barrier": lambda qv: barrier([qv[0], qv[1]]),
    "swap": lambda qv: swap(qv[0], qv[1]),
}


@pytest.mark.parametrize("name", GATE_CALLS.keys())
def test_gate_application_functions_return_none(name):
    """Every gate-application function applies its gate by side effect and
    returns None instead of echoing back its qubit arguments."""
    qv = QuantumVariable(6)
    assert GATE_CALLS[name](qv) is None


@pytest.mark.parametrize("controls_amount", [0, 1, 2, 3])
def test_mcx_returns_none_for_every_control_count(controls_amount):
    """mcx used to special-case 0 and 1 controls with their own early
    ``return``; each branch must still return None."""
    qv = QuantumVariable(controls_amount + 1)
    assert mcx(qv[:controls_amount], qv[controls_amount]) is None


@pytest.mark.parametrize("method", ["gray", "auto", "balauca"])
def test_mcz_applies_the_gate_exactly_once(method):
    """Regression test for a bug where removing mcz's ``return qubits``
    also removed the early exit it doubled as: the "gray"/"auto" branch fell
    through into also calling ``mcz_inner``, applying the multi-controlled Z
    gate twice. Since Z*Z = I, this silently cancelled the phase tag instead
    of raising an error, so it is checked via the resulting phase rather than
    just the return value (which is None either way).
    """
    qv = QuantumVariable(3)
    x(qv)
    result = mcz(list(qv.reg), method=method)
    assert result is None
    assert str(qv.qs.statevector()) == "-|111>"
