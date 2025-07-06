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
from qrisp import *
import math
import pytest

def test_quantum_square_root():
    for a in range(6, 26):
        expected_root = int(math.sqrt(a))
        expected_remainder = a - expected_root**2

        n = math.ceil(math.log2(a + 1))
        if(n % 2 == 0):
            n += 1
        R = QuantumFloat(n, signed=True)
        R[:] = a

        F = q_isqrt(R)

        r, f, = list(multi_measurement([R, F]))[0]

        print("R:", a)
        print("Root:", f)
        print("Remainder", r)
        assert f == expected_root and r == expected_remainder

    R = QuantumFloat(4, signed=True)
    R[:] = 15

    print("Number of qubits:", R.size)
    with pytest.raises(ValueError, match="Input variable should have even number of qubits."):
        q_isqrt(R)

    R = QuantumFloat(4, signed=False)
    R[:] = 15
    
    print("Signed:", R.signed)
    with pytest.raises(ValueError, match="Input variable should be signed."):
        q_isqrt(R)

    

