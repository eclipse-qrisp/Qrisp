"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import *
from qrisp.jasp import *
import pytest

def test_L1_ladder():
    def main(N):
        qf = QuantumFloat(N)
        ladder1_synth_jax(qf)

    for N in range(2, 10):
        circ1 = QuantumFloat(N)
        with invert():
            for i in range(N - 1):
                cx(circ1[i], circ1[i + 1])
        jsp = make_jaspr(main, garbage_collection="none")(1)
        circ2 = jsp.to_qc(N)

        assert circ2.compare_unitary(circ1.qs)

@pytest.mark.parametrize("N", range(4, 6))
def test_l2_ladder(N):
    def main(N, j, k):
        x2 = QuantumFloat(N)
        y2 = QuantumFloat(N)
        anc2 = QuantumFloat(1)
        x2[:] = j
        y2[:] = k
        ladder2_synth_jax(x2[:] + anc2[:], y2[:], method="khattar")
        return x2, y2, anc2

    for k in range(2**N):
        for j in range(2**N):
            x1 = QuantumFloat(N)
            y1 = QuantumFloat(N)
            anc1 = QuantumFloat(1)
            x1[:] = j
            y1[:] = k
            mcx([x1[N - 1], y1[N - 1]], anc1[0])
            with invert():
                for i in range(N - 1):
                    mcx([x1[i], y1[i]], x1[i + 1])
            res1 = multi_measurement([x1, y1, anc1])
            print(dict(res1))

            res2 = terminal_sampling(main)(N, j, k)
            print(dict(res2))
            assert (
                res1.keys() == res2.keys()
            ), f"Key mismatch:\nres1: {res1.keys()}\nres2: {res2.keys()}"
            res1_vals = np.array([res1[k] for k in sorted(res1)])
            res2_vals = np.array([res2[k] for k in sorted(res2)])

            assert np.allclose(
                res1_vals, res2_vals, rtol=1e-12, atol=1e-12
            ), f"Value mismatch:\nres1: {res1_vals}\nres2: {res2_vals}"
            print("Test passed for N =", N, "j =", j, " k =", k)


def test_remaud_adder():
    def main(N, j, k):
        
        A = QuantumFloat(N)
        B = QuantumFloat(N)
        A[:] = j
        B[:] = k

        Z = QuantumFloat(1)
        remaud_adder(A, B, Z)
        return A, B
    # for N in range(4, 7):
    N = 4
    for k in range(2**N):
        for j in range(2**N):
            x1 = QuantumModulus(2**N)
            y1 = QuantumModulus(2**N)
            # anc1 = QuantumFloat(1)
            x1[:] = j
            y1[:] = k
            y1+=x1
                    
            res1 = multi_measurement([x1, y1])
            print(dict(res1))

            res2 = terminal_sampling(main)(N, j, k)
            print(dict(res2))
            assert (
                res1.keys() == res2.keys()
            ), f"Key mismatch:\nres1: {res1.keys()}\nres2: {res2.keys()}"
            res1_vals = np.array([res1[k] for k in sorted(res1)])
            res2_vals = np.array([res2[k] for k in sorted(res2)])

            assert np.allclose(
                res1_vals, res2_vals, rtol=1e-12, atol=1e-12
            ), f"Value mismatch:\nres1: {res1_vals}\nres2: {res2_vals}"
            print("Test passed for N =", N, "j =", j, " k =", k)
