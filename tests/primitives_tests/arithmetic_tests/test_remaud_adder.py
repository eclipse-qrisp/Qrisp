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
# ********************************************************************************/

"""Tests remaud_adder correctness and its ladder synthesis helpers (_ladder1_synth_jax, _ladder2_synth_jax)."""

from qrisp import (
    QuantumFloat,
    boolean_simulation,
    cx,
    invert,
    make_jaspr,
    mcx,
    measure,
    multi_measurement,
    remaud_adder,
)
from qrisp.alg_primitives.arithmetic.adders.remaud import _ladder1_synth_jax, _ladder2_synth_jax


def test_L1_ladder():
    """_ladder1_synth_jax matches an inverted CNOT ladder, including the edge cases N=1 and N=2."""

    def main(N):
        qf = QuantumFloat(N)
        _ladder1_synth_jax(qf)

    for N in range(1, 10):
        circ1 = QuantumFloat(N)
        with invert():
            for i in range(N - 1):
                cx(circ1[i], circ1[i + 1])
        jsp = make_jaspr(main)(1)
        circ2 = jsp.to_qc(N)

        assert circ2.compare_unitary(circ1.qs)


def test_L1_ladder_static():
    """_ladder1_synth_jax in static mode, including the edge cases N=1 and N=2."""
    for N in range(1, 6):
        circ1 = QuantumFloat(N)
        with invert():
            for i in range(N - 1):
                cx(circ1[i], circ1[i + 1])

        qf = QuantumFloat(N)
        _ladder1_synth_jax(qf)

        assert qf.qs.compare_unitary(circ1.qs)


def test_l2_ladder(exhaustive=False):
    """_ladder2_synth_jax matches an inverted Toffoli ladder, including the edge case of a 2-qubit x register."""

    @boolean_simulation
    def main(N, j, k):
        x2 = QuantumFloat(N)
        y2 = QuantumFloat(N)
        anc2 = QuantumFloat(1)
        x2[:] = j
        y2[:] = k
        _ladder2_synth_jax(x2[:] + anc2[:], y2[:], method="khattar")
        return measure(x2), measure(y2), measure(anc2)

    if exhaustive:
        up_bound = 8
    else:
        up_bound = 5

    for N in range(1, up_bound):
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
                x1, y1, anc1 = next(iter(multi_measurement([x1, y1, anc1])))

                x2, y2, anc2 = main(N, j, k)

                assert x2 == x1
                assert y2 == y1
                assert anc2 == anc1


def test_remaud_adder():
    """Exhaustive dynamic test of sum and carry out, including sizes below 4."""

    @boolean_simulation
    def main(N, j, k):

        A = QuantumFloat(N)
        B = QuantumFloat(N)
        A[:] = j
        B[:] = k

        Z = QuantumFloat(1)
        remaud_adder(A, B, Z)
        return measure(A), measure(B), measure(Z)

    for N in range(1, 8):
        for k in range(2**N):
            for j in range(2**N):
                A, B, Z = main(N, j, k)
                assert A == j
                assert B == (k + j) % (2**N)
                assert Z == (k + j) >> N


def test_remaud_adder_standard(exhaustive=False):
    """Exhaustive static test of sum and carry out, including sizes below 4."""
    if exhaustive:
        up_bound = 8
    else:
        up_bound = 5

    for N in range(1, up_bound):
        for k in range(2**N):
            for j in range(2**N):
                A = QuantumFloat(N)
                B = QuantumFloat(N)
                A[:] = j
                B[:] = k

                Z = QuantumFloat(1)
                remaud_adder(A, B, Z)

                assert multi_measurement([A, B, Z]) == {(j, (k + j) % (2**N), (k + j) >> N): 1.0}


def test_remaud_adder_static_large():
    """Static mode with N >= 9, where _ladder2_synth_jax reaches its inner CL/CR loops (regression test)."""
    for N in [9, 12]:
        A = QuantumFloat(N)
        B = QuantumFloat(N)
        Z = QuantumFloat(1)
        A[:] = 2**N - 3
        B[:] = 5

        remaud_adder(A, B, Z)

        assert multi_measurement([A, B, Z]) == {(2**N - 3, 2, 1): 1.0}
