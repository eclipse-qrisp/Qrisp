"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

# Created by ann81984 at 06.05.2022
from qrisp.arithmetic import QuantumFloat, q_divmod
from qrisp.misc import multi_measurement


def test_quantum_divison():
    numerator = QuantumFloat(3, -2, signed=True)
    divisor = QuantumFloat(3, -2, signed=True)

    n = 4 / 8
    d = -5 / 4
    numerator.encode(n)
    divisor.encode(d)

    prec = 4
    quotient, remainder = q_divmod(numerator, divisor, prec=prec, adder="thapliyal")

    q, r = list(multi_measurement([quotient, remainder]))[0]

    print("Q: ", q)
    print("delta_Q: ", abs(q - n / d))
    print("R: ", r)
    assert q == -0.375 and r == 0.03125 and abs(q - n / d) < 2**-prec
