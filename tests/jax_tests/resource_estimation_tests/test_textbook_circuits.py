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
# ********************************************************************************

"""Resource estimation of textbook circuits whose resources are known in closed form."""

# These tests are a source of truth for count_ops, depth and num_qubits. The
# circuits only use elementary gates, so the expected resources follow from
# the circuits themselves and not from other parts of Qrisp. Each expected
# value is written out as a number, and its closed form is given next to it.
# The expected qubit counts are written as (allocated, deallocated, peak, final),
# the four values returned by num_qubits.
#
# The depth is the length of the longest chain of gates sharing a qubit, where
# measurements and resets take no time, as in the depth metric. The expected
# depths were cross-checked with an independent scheduler that starts each
# gate as soon as all its qubits are free.
#
# Together, the circuits use every quantum primitive: allocation and deletion,
# qubit access with positive and negative indices, sizes, slices, fusing,
# gates, measurements of qubits and registers, resets and parities.

import pytest

from qrisp import (
    QuantumFloat,
    QuantumVariable,
    control,
    cx,
    cz,
    h,
    measure,
    parity,
    reset,
    t,
    t_dg,
    x,
    z,
)
from qrisp.jasp import jlen, jrange


def _expected(ops, depth, qubits):
    """Build the expected results of count_ops, depth and num_qubits (qubits as allocated, deallocated, peak, final)."""
    allocated, deallocated, peak, final = qubits
    return {
        "count_ops": ops,
        "depth": depth,
        "num_qubits": {
            "total_allocated": allocated,
            "total_deallocated": deallocated,
            "peak_allocations": peak,
            "finally_allocated": final,
        },
    }


def test_bell_pair(estimate_resources):
    """Bell pair: one H and one CNOT in sequence on two qubits."""

    def main():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])

    assert estimate_resources(main) == _expected({"h": 1, "cx": 1}, depth=2, qubits=(2, 0, 2, 2))


def _ghz_forward(n):
    """GHZ chain on a register with a dynamic size, from the first to the last qubit."""
    qv = QuantumFloat(n)
    h(qv[0])
    for i in jrange(jlen(qv) - 1):
        cx(qv[i], qv[i + 1])


def _ghz_backward(n):
    """GHZ chain from the last to the first qubit, addressed with negative indices."""
    qv = QuantumFloat(n)
    h(qv[-1])
    for i in jrange(1, n):
        cx(qv[-i], qv[-i - 1])


@pytest.mark.parametrize("program", [_ghz_forward, _ghz_backward], ids=["forward", "backward"])
@pytest.mark.parametrize(
    "n, expected",
    [
        (2, _expected({"h": 1, "cx": 1}, depth=2, qubits=(2, 0, 2, 2))),
        (3, _expected({"h": 1, "cx": 2}, depth=3, qubits=(3, 0, 3, 3))),
        (6, _expected({"h": 1, "cx": 5}, depth=6, qubits=(6, 0, 6, 6))),
    ],
    ids=["n=2", "n=3", "n=6"],
)
def test_ghz_chain(estimate_resources, program, n, expected):
    """GHZ state prepared with a chain of CNOTs: 1 H and n - 1 CNOTs, depth n."""
    assert estimate_resources(program, n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (2, _expected({"h": 1, "cx": 1}, depth=2, qubits=(4, 0, 4, 4))),
        (5, _expected({"h": 1, "cx": 4}, depth=5, qubits=(7, 0, 7, 7))),
    ],
    ids=["n=2", "n=5"],
)
def test_ghz_chain_on_slice(estimate_resources, n, expected):
    """GHZ chain on ``qv[1:-1]`` of an (n + 2)-qubit register, the two outer qubits staying idle."""

    def main(n):
        qv = QuantumFloat(n + 2)
        ghz = qv[1:-1]
        h(ghz[0])
        for i in jrange(jlen(ghz) - 1):
            cx(ghz[i], ghz[i + 1])

    assert estimate_resources(main, n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (4, _expected({"h": 1, "cx": 3}, depth=3, qubits=(4, 0, 4, 4))),
        (8, _expected({"h": 1, "cx": 7}, depth=4, qubits=(8, 0, 8, 8))),
        (16, _expected({"h": 1, "cx": 15}, depth=5, qubits=(16, 0, 16, 16))),
    ],
    ids=["n=4", "n=8", "n=16"],
)
def test_ghz_log_depth(estimate_resources, n, expected):
    """GHZ state with a CNOT fan-out doubling the entangled qubits in every layer: depth 1 + log2(n)."""

    def main():
        qv = QuantumVariable(n)
        h(qv[0])
        width = 1
        while width < n:
            for j in range(width):
                cx(qv[j], qv[j + width])
            width *= 2

    assert estimate_resources(main) == expected


def test_grover_two_qubits(estimate_resources):
    """One Grover iteration on 2 qubits marking |11>: 6 H, 4 X and 2 CZ in 7 layers."""

    def main():
        qv = QuantumVariable(2)
        h(qv)
        cz(qv[0], qv[1])  # oracle
        h(qv)  # diffusion operator
        x(qv)
        cz(qv[0], qv[1])
        x(qv)
        h(qv)

    assert estimate_resources(main) == _expected({"h": 6, "x": 4, "cz": 2}, depth=7, qubits=(2, 0, 2, 2))


def test_toffoli_decomposition(estimate_resources):
    """Toffoli gate decomposed as in Nielsen & Chuang, Fig. 4.9: 7 T and T-dagger, 6 CNOT, 2 H, depth 11."""

    def main():
        qv = QuantumVariable(3)
        c1, c2, tgt = qv[0], qv[1], qv[2]
        h(tgt)
        cx(c2, tgt)
        t_dg(tgt)
        cx(c1, tgt)
        t(tgt)
        cx(c2, tgt)
        t_dg(tgt)
        cx(c1, tgt)
        t(c2)
        t(tgt)
        h(tgt)
        cx(c1, c2)
        t(c1)
        t_dg(c2)
        cx(c1, c2)

    expected = _expected({"h": 2, "cx": 6, "t": 4, "t_dg": 3}, depth=11, qubits=(3, 0, 3, 3))
    assert estimate_resources(main) == expected


@pytest.mark.parametrize(
    "n, secret, expected",
    [
        (3, 0b001, _expected({"x": 1, "h": 7, "cx": 1, "measure": 3}, depth=4, qubits=(4, 0, 4, 4))),
        (4, 0b1011, _expected({"x": 1, "h": 9, "cx": 3, "measure": 4}, depth=6, qubits=(5, 0, 5, 5))),
        (5, 0b11111, _expected({"x": 1, "h": 11, "cx": 5, "measure": 5}, depth=8, qubits=(6, 0, 6, 6))),
    ],
    ids=["s=001", "s=1011", "s=11111"],
)
def test_bernstein_vazirani(estimate_resources, n, secret, expected):
    """Bernstein-Vazirani for an n-bit secret s: 2n + 1 H, |s| CNOTs, n measurements, depth 3 + |s|.

    The first Hadamard layer acts on the data register fused with the ancilla qubit.
    """

    def main():
        data = QuantumVariable(n)
        anc = QuantumVariable(1)
        x(anc[0])
        h(data[:] + [anc[0]])
        for i in range(n):
            if secret >> i & 1:
                cx(data[i], anc[0])
        h(data)
        return measure(data)

    assert estimate_resources(main) == expected


@pytest.mark.parametrize(
    "meas_behavior, n, outcome, expected",
    [
        ("0", 1, 0, _expected({"h": 1, "measure": 1, "x": 1}, depth=1, qubits=(2, 0, 2, 2))),
        ("0", 3, 0, _expected({"h": 3, "measure": 3, "x": 1}, depth=1, qubits=(4, 0, 4, 4))),
        ("1", 1, 1, _expected({"h": 1, "measure": 1, "x": 1}, depth=1, qubits=(2, 0, 2, 2))),
        ("1", 3, 7, _expected({"h": 3, "measure": 3, "x": 1}, depth=1, qubits=(4, 0, 4, 4))),
    ],
    ids=["all 0, n=1", "all 0, n=3", "all 1, n=1", "all 1, n=3"],
)
def test_register_measurement_value(estimate_resources, meas_behavior, n, outcome, expected):
    """Measuring an n-qubit register gives 0 if every qubit gives 0, and 2**n - 1 if every qubit gives 1.

    The X gate on the flag is applied only if the measured value equals the expected outcome.
    """

    def main(n):
        qv = QuantumFloat(n)
        h(qv)
        flag = QuantumVariable(1)
        with control(measure(qv) == outcome):
            x(flag[0])

    assert estimate_resources(main, n, meas_behavior=meas_behavior) == expected


@pytest.mark.parametrize(
    "meas_behavior, expected",
    [
        ("0", _expected({"h": 3, "cx": 2, "measure": 2}, depth=4, qubits=(3, 0, 3, 3))),
        ("1", _expected({"h": 3, "cx": 2, "measure": 2, "x": 1, "z": 1}, depth=4, qubits=(3, 0, 3, 3))),
    ],
    ids=["outcomes 0", "outcomes 1"],
)
def test_teleportation(estimate_resources, meas_behavior, expected):
    """Quantum teleportation: the X and Z corrections are applied only when their outcomes are 1."""

    def main():
        q = QuantumVariable(3)
        h(q[0])  # the state to teleport
        h(q[1])  # Bell pair between q[1] and q[2]
        cx(q[1], q[2])
        cx(q[0], q[1])  # Bell measurement of q[0] and q[1]
        h(q[0])
        m0 = measure(q[0])
        m1 = measure(q[1])
        with control(m1):
            x(q[2])
        with control(m0):
            z(q[2])

    assert estimate_resources(main, meas_behavior=meas_behavior) == expected


@pytest.mark.parametrize(
    "d, rounds, expected",
    [
        (3, 1, _expected({"cx": 4, "measure": 2}, depth=2, qubits=(5, 2, 5, 3))),
        (4, 3, _expected({"cx": 18, "measure": 9}, depth=6, qubits=(13, 9, 7, 4))),
        (5, 2, _expected({"cx": 16, "measure": 8}, depth=4, qubits=(13, 8, 9, 5))),
    ],
    ids=["d=3, 1 round", "d=4, 3 rounds", "d=5, 2 rounds"],
)
def test_repetition_code_fresh_ancillas(estimate_resources, d, rounds, expected):
    """Syndrome extraction of a distance-d repetition code, with fresh ancillas in every round.

    Each round allocates d - 1 ancillas, applies two layers of disjoint CNOTs
    (from ``data[:-1]`` and from ``data[1:]``), measures and deletes the ancillas:
    2 rounds (d - 1) CNOTs, rounds (d - 1) measurements, depth 2 rounds, and a
    peak of 2d - 1 qubits.
    """

    def main(d, rounds):
        data = QuantumFloat(d)
        for _ in jrange(rounds):
            anc = QuantumVariable(d - 1)
            for i in jrange(d - 1):
                cx(data[:-1][i], anc[i])
            for i in jrange(d - 1):
                cx(data[1:][i], anc[i])
            measure(anc)
            anc.delete()

    assert estimate_resources(main, d, rounds) == expected


@pytest.mark.parametrize(
    "d, rounds, expected",
    [
        (3, 1, _expected({"cx": 4, "measure": 2}, depth=2, qubits=(5, 0, 5, 5))),
        (4, 3, _expected({"cx": 18, "measure": 9}, depth=6, qubits=(7, 0, 7, 7))),
    ],
    ids=["d=3, 1 round", "d=4, 3 rounds"],
)
def test_repetition_code_reset_ancillas(estimate_resources, d, rounds, expected):
    """The same syndrome extraction, reusing the d - 1 ancillas with a reset after every round.

    count_ops does not count resets, and depth ignores them.
    """

    def main(d, rounds):
        data = QuantumFloat(d)
        anc = QuantumVariable(d - 1)
        for _ in jrange(rounds):
            for i in jrange(d - 1):
                cx(data[:-1][i], anc[i])
            for i in jrange(d - 1):
                cx(data[1:][i], anc[i])
            measure(anc)
            reset(anc)

    assert estimate_resources(main, d, rounds) == expected


@pytest.mark.parametrize(
    "meas_behavior, expected",
    [
        ("0", _expected({"cx": 4, "measure": 2}, depth=4, qubits=(6, 0, 6, 6))),
        ("1", _expected({"cx": 4, "measure": 2, "x": 1}, depth=4, qubits=(6, 0, 6, 6))),
    ],
    ids=["syndrome 00", "syndrome 11"],
)
def test_repetition_code_decoding(estimate_resources, meas_behavior, expected):
    """Decoding a distance-3 repetition code from its two syndrome bits.

    An odd syndrome weight (their parity) means an error on an outer data qubit,
    both bits set mean an error on the middle one. When both outcomes are 1, only
    the middle qubit is corrected.
    """

    def main():
        data = QuantumVariable(3)
        anc = QuantumVariable(2)
        flag = QuantumVariable(1)
        cx(data[0], anc[0])
        cx(data[1], anc[0])
        cx(data[1], anc[1])
        cx(data[2], anc[1])
        s0 = measure(anc[0])
        s1 = measure(anc[1])
        with control(parity(s0, s1)):
            x(flag[0])
        with control(s0):
            with control(s1):
                x(data[1])

    assert estimate_resources(main, meas_behavior=meas_behavior) == expected
