"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

import time

from qrisp import *
from qrisp.jasp import *


def test_control_compilation():

    def test_f_0(a):

        b = QuantumFloat(4)

        b[:] = a

        c = QuantumBool()

        with control([b[0], b[1]], 3):
            with control([b[2], b[3]], ctrl_state="10"):
                x(c[0])

        return measure(c)

    jaspr = make_jaspr(test_f_0)(1)

    for i in range(2**4):
        assert jaspr(i) == (i == 7)

    ###############
    @custom_control
    def swap(qv, qb, ctrl=None):

        cx(qv[0], qv[1])

        if ctrl is not None:
            with control(ctrl):
                cx(qv[1], qv[0])
        else:
            cx(qv[1], qv[0])

        h(qb)
        cx(qv[0], qv[1])

    def test_f_1():

        ctrl_qv = QuantumVariable(4)
        qv = QuantumFloat(3)

        qb = qv[2]

        x(ctrl_qv[0])
        x(ctrl_qv[1])

        x(qv[0])

        with invert():
            with control([ctrl_qv[2], ctrl_qv[3]]):
                x(ctrl_qv[0])
                with control([ctrl_qv[0], ctrl_qv[1]], ctrl_state="01"):
                    swap(qv, qb)

    jaspr = make_jaspr(test_f_1)()
    qc = jaspr.to_qc()
    assert qc.cnot_count() < 50

    @terminal_sampling
    def main(phi, i):

        qv = QuantumFloat(i)

        x(qv[: qv.size - 1])

        qbl = QuantumBool()
        h(qbl)

        with control(qbl[0]):
            with conjugate(h)(qv[qv.size - 1]):
                mcp(phi, qv)

        return qv

    assert main(np.pi, 5) == {15.0: 0.5, 31.0: 0.5}
