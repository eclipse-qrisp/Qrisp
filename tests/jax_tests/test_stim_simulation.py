"""********************************************************************************
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

from qrisp import *
from qrisp.jasp import *


def test_stim_simulation():

    @stimulate
    def main():

        qbl = QuantumBool()
        qf = QuantumFloat(4)

        # Bring qbl into superposition
        h(qbl)

        # Perform a measure
        cl_bl = measure(qbl)

        # Perform a conditional operation based on the measurement outcome
        with control(cl_bl):
            qf[:] = 1
            h(qf[2])

        return measure(qf), measure(qbl)

    assert main() in [(1.0, True), (5.0, True), (0.0, False)]

    @stimulate
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return measure(qf), measure(a), measure(qbl)

    for i in range(3):
        for j in range(3):
            assert main(i, j) in [(0.0, 0.0, False), (2**i, 2**j, True)]


def test_stim_dispatch_single_qubit_gates():
    """Exercise every single-qubit entry in BufferedQuantumState's stim dispatch table.

    The rest of this file only exercises h via the stim backend; y, z, s, s_dg
    are added here so a typo in _STIM_GATE_METHODS can't hide behind an
    untested dict entry. Each check is a small, deterministic Clifford
    identity, so a wrong dispatch flips the measured outcome rather than
    merely raising:
      - y|0> = i|1> (global phase invisible to measurement) -> True
      - H Z H == X, so h,z,h must match a plain x (True); h,h alone (no z)
        is the identity (False), confirming z actually contributes.
      - S.S == Z (up to phase), so h,s,s,h must also give True.
      - S.S_dg == Identity, so h,s,s_dg,h must give False, like h,h alone.
    """

    @stimulate
    def test_y():
        qv = QuantumVariable(1)
        y(qv[0])
        return measure(qv[0])

    assert test_y() == True

    @stimulate
    def test_hh():
        qv = QuantumVariable(1)
        h(qv[0])
        h(qv[0])
        return measure(qv[0])

    assert test_hh() == False

    @stimulate
    def test_hzh():
        qv = QuantumVariable(1)
        h(qv[0])
        z(qv[0])
        h(qv[0])
        return measure(qv[0])

    assert test_hzh() == True

    @stimulate
    def test_h_ss_h():
        qv = QuantumVariable(1)
        h(qv[0])
        s(qv[0])
        s(qv[0])
        h(qv[0])
        return measure(qv[0])

    assert test_h_ss_h() == True

    @stimulate
    def test_h_s_sdg_h():
        qv = QuantumVariable(1)
        h(qv[0])
        s(qv[0])
        s_dg(qv[0])
        h(qv[0])
        return measure(qv[0])

    assert test_h_s_sdg_h() == False


def test_stim_dispatch_controlled_gates():
    """Exercise cx/cy/cz in BufferedQuantumState's stim dispatch table.

    cx is already exercised (as part of GHZ states) elsewhere in this file;
    it's repeated here so all three controlled gates are verified together.
    cy and cz use a phase-kickback circuit rather than a plain population
    check, since a population-only check (control=1 via prior x, then look
    at whether the target flipped) can't distinguish a Y or Z dispatch from
    an X dispatch -- X, Y both flip a |0> target's population, and a lone
    computational-basis Z application is invisible to a Z-basis measurement.
    Instead: put the control in a Y (resp. Z) eigenstate with eigenvalue -1,
    apply the controlled gate, then apply H to the control again. A correctly
    dispatched controlled-Y/Z kicks the -1 phase back onto the control,
    flipping it from |+> to |1> -- deterministically measurable, and
    different from what a wrong dispatch (e.g. accidentally wired to cx)
    would produce.
    """

    @stimulate
    def test_cx():
        qv = QuantumVariable(2)
        x(qv[0])
        cx(qv[0], qv[1])
        return measure(qv[0]), measure(qv[1])

    assert test_cx() == (True, True)

    # cz phase kickback: control=H|0>, target=X|0>=|1> (Z's -1-eigenstate).
    @stimulate
    def test_cz():
        qv = QuantumVariable(2)
        h(qv[0])
        x(qv[1])
        cz(qv[0], qv[1])
        h(qv[0])
        return measure(qv[0])

    assert test_cz() == True

    @stimulate
    def test_no_cz():
        qv = QuantumVariable(2)
        h(qv[0])
        x(qv[1])
        h(qv[0])
        return measure(qv[0])

    assert test_no_cz() == False

    # cy phase kickback: control=H|0>, target=S_dg(H|0>)) (Y's -1-eigenstate).
    @stimulate
    def test_cy():
        qv = QuantumVariable(2)
        h(qv[0])
        h(qv[1])
        s_dg(qv[1])
        cy(qv[0], qv[1])
        h(qv[0])
        return measure(qv[0])

    assert test_cy() == True

    @stimulate
    def test_no_cy():
        qv = QuantumVariable(2)
        h(qv[0])
        h(qv[1])
        s_dg(qv[1])
        h(qv[0])
        return measure(qv[0])

    assert test_no_cy() == False


def test_stimulate_raises_when_returning_quantum_variable():
    """Stimulate must reject a function that returns a QuantumVariable, not silently accept it."""

    @stimulate
    def main():
        return QuantumFloat(2)

    with pytest.raises(Exception, match="Tried to simulate function returning a QuantumVariable"):
        main()
