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

from qrisp import QuantumVariable, h, x, LCU
from qrisp.jasp import terminal_sampling
from qrisp.operators import X,Y,Z
import numpy as np

# A = I + X
def test_LCU():

    def U0(operand):
        pass

    def U1(operand):
        x(operand)

    unitaries = [U0, U1]

    def state_prep(case):
        h(case)

    def operand_prep():
        operand = QuantumVariable(1)
        return operand

    @terminal_sampling
    def main():

        qv = LCU(operand_prep, state_prep, unitaries)
        return qv

    meas_res = main()
    assert np.round(meas_res[0],3)==0.5 and np.round(meas_res[1],3)==0.5
    #{0: 0.5, 1: 0.5}


# A = cos(H)
def test_LCU_cos():

    H = Z(0)*Z(1) + X(0)*X(1)

    def U0(operand):
        H.trotterization(forward_evolution=False)(operand)

    def U1(operand):
        H.trotterization(forward_evolution=True)(operand)

    unitaries = [U0, U1]

    def state_prep(case):
        h(case)

    def operand_prep():
        operand = QuantumVariable(2)
        return operand

    @terminal_sampling
    def main():

        qv = LCU(operand_prep, state_prep, unitaries)
        return qv

    meas_res = main()
    assert np.round(meas_res[0],3)==0.145 and np.round(meas_res[3],3)==0.855
    #{3: 0.85471756539818, 0: 0.14528243460182003}