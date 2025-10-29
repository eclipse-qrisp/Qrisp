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

from qrisp import *
from qrisp.jasp import *


def test_custom_control():

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

    def test_f():

        ctrl_qv = QuantumVariable(3)
        qv = QuantumFloat(2)

        qb = ctrl_qv[2]

        x(ctrl_qv[0])
        x(ctrl_qv[1])

        x(qv[0])

        with control([ctrl_qv[0], ctrl_qv[1]]):
            swap(qv, qb)

        return measure(qv)

    jaspr = make_jaspr(test_f)()

    assert jaspr() == 2
