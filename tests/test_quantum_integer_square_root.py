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

from qrisp.alg_primitives.arithmetic.isqrt import q_isqrt
from qrisp import QuantumFloat, multi_measurement
import math

def test_quantum_square_root():
    for a in range(6, 26):
        expected_root = int(math.sqrt(a))
        expected_remainder = a - expected_root**2

        n = math.ceil(math.log2(a + 1))
        R = QuantumFloat(n)
        R[:] = a

        F = q_isqrt(R)

        r, f, = list(multi_measurement([R, F]))[0]

        print("R:", a)
        print("Root:", f)
        print("Remainder", r)
        assert f == expected_root and r == expected_remainder

    R = QuantumFloat(8)
    R[:] = {131: 1, 81: 1}
    F = q_isqrt(R)
    
    mes_res = multi_measurement([R, F])
    
    expected_results = {(10, 11): 0.5, (0, 9): 0.5}
    
    assert mes_res == expected_results

