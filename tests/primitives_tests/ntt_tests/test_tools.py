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

import numpy as np
import jax.numpy as jnp
import pytest
from qrisp import boolean_simulation, measure, QuantumArray, QuantumModulus
from qrisp.alg_primitives.ntt import qntt, qntt_inv, multiply_qntts, multiply_ntts


@pytest.mark.parametrize(
    "a, n, q, root",
    [
        # 1. Small test case (n=4, q=13)
        (jnp.array([3, 1, 4, 9]), 4, 13, 5),
        # 2. Small test case with the other primitive 4th root of unity mod 13
        (jnp.array([12, 0, 7, 2]), 4, 13, 8),
        # 3. Medium test case (n=8, q=17)
        (jnp.array([1, 15, 8, 4, 12, 3, 9, 0]), 8, 17, 9),
    ],
)
def test_ntt(a, n, q, root):
    """
    Tests that taking the incomplete NTT and then the INTT
    returns the original polynomial modulo q.
    """

    @boolean_simulation
    def main():

        qa = QuantumArray(QuantumModulus(q), shape=(n,))
        qa[:] = a

        # 1. Forward Transform
        qntt(qa, root)

        # 2. Inverse Transform
        qntt_inv(qa, root)

        return measure(qa)

    a_new = np.array(main())

    # 3. Check for exact equality modulo q
    assert np.array_equal(a_new % q, a % q)


@pytest.mark.parametrize(
    "a, b, n, q, root",
    [
        # 1. Small test case (n=4, q=13)
        (jnp.array([3, 1, 4, 9]), jnp.array([1, 1, 5, 3]), 4, 13, 5),
        # 2. Small test case with the other primitive 4th root of unity mod 13
        (jnp.array([12, 0, 7, 2]), jnp.array([3, 1, 4, 9]), 4, 13, 8),
        # 3. Medium test case (n=8, q=17)
        (jnp.array([1, 15, 8, 4, 12, 3, 9, 0]), jnp.array([1, 1, 5, 15, 13, 3, 4, 1]), 8, 17, 9),
    ],
)
def test_qc_multiply_ntts(a, b, n, q, root):
    """
    Tests that quantum-classical multiplication of NTTs matches classical multiplication of NTTs.
    """

    @boolean_simulation
    def main():

        qa = QuantumArray(QuantumModulus(q), shape=(n,))
        qa[:] = a

        res = multiply_qntts(qa, b, root)

        return measure(res)

    res = np.array(main())
    expected = multiply_ntts(a, b, n, q, root)

    # 3. Check for exact equality modulo q
    assert np.array_equal(res % q, expected % q)
