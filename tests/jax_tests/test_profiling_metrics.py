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

"""Tests for the behavior shared by the count_ops, depth and num_qubits profiling metrics."""

import itertools
import math
from typing import NamedTuple

import pytest

from qrisp import (
    QuantumBool,
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
from qrisp.jasp import count_ops, depth, jlen, jrange, num_qubits, qache
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import normalize_slice_bounds
from qrisp.jasp.interpreter_tools.interpreters.utilities import meas_rng


def test_normalize_slice_bounds_matches_python():
    """``normalize_slice_bounds`` follows Python slicing semantics for every combination of bounds."""
    for size, start, stop in itertools.product(range(5), range(-7, 8), range(-7, 8)):
        expected = range(size)[start:stop]
        norm_start, norm_stop = normalize_slice_bounds(size, start, stop)

        assert int(norm_stop) - int(norm_start) == len(expected), (size, start, stop)
        if expected:
            assert int(norm_start) == expected.start, (size, start, stop)


def _measure_and_flip(target):
    """Measure a fresh qubit in superposition, and if the outcome is 1, flip ``target``.

    The outcome 1 also allocates (and uses) an extra qubit, so every metric can
    tell how many outcomes were 1: count_ops counts the X gates, num_qubits the
    extra qubits, and depth the sequential X gates on ``target``.
    """
    qb = QuantumBool()
    h(qb)
    with control(measure(qb)):
        extra = QuantumBool()
        h(extra)
        x(target)


@qache
def _measure_and_flip_subroutine(target):
    """Call ``_measure_and_flip`` through a qache'd subroutine."""
    _measure_and_flip(target)


def _loop_program(num_iterations):
    """Measure inside a jrange loop."""
    target = QuantumBool()
    for _ in jrange(num_iterations):
        _measure_and_flip(target)


def _subroutine_program():
    """Measure inside 16 calls of the same qache'd subroutine."""
    target = QuantumBool()
    for _ in range(16):
        _measure_and_flip_subroutine(target)


def _branch_program(num_iterations):
    """Measure inside a classical branch, then again after it."""
    target = QuantumBool()
    with control(num_iterations > 0):
        for _ in jrange(num_iterations):
            _measure_and_flip(target)
    for _ in jrange(num_iterations):
        _measure_and_flip(target)


@pytest.mark.parametrize(
    "program,args,num_measurements",
    [
        (_loop_program, (16,), 16),
        (_subroutine_program, (), 16),
        (_branch_program, (8,), 16),
    ],
    ids=["jrange loop", "qache subroutine", "branch"],
)
def test_metrics_see_the_same_random_outcomes(program, args, num_measurements):
    """With a random measurement behavior, all metrics see the same outcomes, and the outcomes vary.

    The k-th measured qubit is sampled with the key ``jax.random.key(k)``.

    Regression test: depth and num_qubits numbered the measurements with a counter
    that was not carried through loops, branches and subroutine calls. They reused
    the same random key, so every measurement of a loop had the same outcome, and
    the metrics took different branches than count_ops for the same program.
    """
    ops = count_ops(meas_behavior=meas_rng)(program)(*args)
    num_ones = ops.get("x", 0)

    assert ops["measure"] == num_measurements
    # Neither all 0 nor all 1, so a repeated key would change the counts below
    assert 0 < num_ones < num_measurements

    # The target, one measured qubit per measurement, and one extra qubit per outcome 1
    qubits = num_qubits(meas_behavior=meas_rng)(program)(*args)
    assert qubits["total_allocated"] == 1 + num_measurements + num_ones

    # All other gates act on fresh qubits in parallel, only the X gates on the target are sequential
    assert depth(meas_behavior=meas_rng)(program)(*args) == num_ones


def _qubits(total_allocated, total_deallocated=0, peak=None, final=None):
    """Build the expected result of num_qubits; by default nothing is deallocated."""
    return {
        "total_allocated": total_allocated,
        "total_deallocated": total_deallocated,
        "peak_allocations": total_allocated if peak is None else peak,
        "finally_allocated": total_allocated - total_deallocated if final is None else final,
    }


class Expected(NamedTuple):
    """The expected results of count_ops, depth and num_qubits for a program."""

    ops: dict
    depth: int
    qubits: dict


def _check_metrics(program, args, expected, meas_behavior="0"):
    """Check the three metrics of ``program(*args)`` against ``expected``."""
    assert count_ops(meas_behavior=meas_behavior)(program)(*args) == expected.ops
    assert depth(meas_behavior=meas_behavior)(program)(*args) == expected.depth
    assert num_qubits(meas_behavior=meas_behavior)(program)(*args) == expected.qubits


class TestTextbookCircuits:
    """Check all three metrics on textbook circuits whose resources are known in closed form.

    The circuits only use elementary gates, so the expected gate counts, depths and
    qubit numbers follow from the circuits themselves, not from other parts of Qrisp.
    The depth is the length of the longest chain of gates sharing a qubit, where
    measurements take no time, as in the depth metric. Together, the circuits use
    every quantum primitive: allocation and deletion, qubit access with positive and
    negative indices, sizes, slices, fusing, gates, measurements, resets and parities.
    """

    def test_bell_pair(self):
        """Bell pair: one H and one CNOT in sequence on two qubits."""

        def main():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])

        _check_metrics(main, (), Expected({"h": 1, "cx": 1}, 2, _qubits(2)))

    @pytest.mark.parametrize("n", [2, 3, 6])
    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_ghz_chain(self, n, direction):
        """GHZ state with a chain of CNOTs: 1 H and n - 1 CNOTs, depth n.

        The size is dynamic; the backward chain addresses the qubits with negative indices.
        """

        def forward(n):
            qv = QuantumFloat(n)
            h(qv[0])
            for i in jrange(jlen(qv) - 1):
                cx(qv[i], qv[i + 1])

        def backward(n):
            qv = QuantumFloat(n)
            h(qv[-1])
            for i in jrange(1, n):
                cx(qv[-i], qv[-i - 1])

        program = forward if direction == "forward" else backward
        _check_metrics(program, (n,), Expected({"h": 1, "cx": n - 1}, n, _qubits(n)))

    @pytest.mark.parametrize("n", [2, 5])
    def test_ghz_chain_on_slice(self, n):
        """GHZ state on ``qv[1:-1]`` of an (n + 2)-qubit register: the two outer qubits stay idle."""

        def main(n):
            qv = QuantumFloat(n + 2)
            ghz = qv[1:-1]
            h(ghz[0])
            for i in jrange(jlen(ghz) - 1):
                cx(ghz[i], ghz[i + 1])

        _check_metrics(main, (n,), Expected({"h": 1, "cx": n - 1}, n, _qubits(n + 2)))

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_ghz_log_depth(self, n):
        """GHZ state with a CNOT fan-out that doubles the entangled qubits per layer: depth 1 + log2(n)."""

        def main():
            qv = QuantumVariable(n)
            h(qv[0])
            width = 1
            while width < n:
                for j in range(width):
                    cx(qv[j], qv[j + width])
                width *= 2

        _check_metrics(main, (), Expected({"h": 1, "cx": n - 1}, 1 + int(math.log2(n)), _qubits(n)))

    def test_grover_two_qubits(self):
        """One Grover iteration on 2 qubits marking |11>: 6 H, 4 X, 2 CZ in 7 layers."""

        def main():
            qv = QuantumVariable(2)
            h(qv)
            cz(qv[0], qv[1])  # oracle
            h(qv)  # diffusion operator
            x(qv)
            cz(qv[0], qv[1])
            x(qv)
            h(qv)

        _check_metrics(main, (), Expected({"h": 6, "x": 4, "cz": 2}, 7, _qubits(2)))

    def test_toffoli_decomposition(self):
        """Toffoli gate decomposed as in Nielsen & Chuang (Fig. 4.9): 7 T/T-dagger, 6 CNOT, 2 H, depth 11."""

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

        _check_metrics(main, (), Expected({"h": 2, "cx": 6, "t": 4, "t_dg": 3}, 11, _qubits(3)))

    @pytest.mark.parametrize("n,secret", [(3, 0b111), (4, 0b1011), (5, 0b10110)])
    def test_bernstein_vazirani(self, n, secret):
        """Bernstein-Vazirani for an n-bit secret s: 2n + 1 H, |s| CNOTs, n measurements, depth 3 + |s|.

        The first Hadamard layer acts on the data register fused with the ancilla qubit.
        """
        weight = bin(secret).count("1")

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

        expected_ops = {"x": 1, "h": 2 * n + 1, "cx": weight, "measure": n}
        _check_metrics(main, (), Expected(expected_ops, 3 + weight, _qubits(n + 1)))

    @pytest.mark.parametrize("meas_behavior", ["0", "1"])
    @pytest.mark.parametrize("n", [1, 3])
    def test_register_measurement_value(self, meas_behavior, n):
        """Measuring an n-qubit register gives 0 if every qubit gives 0, and 2**n - 1 if every qubit gives 1."""
        expected_value = 0 if meas_behavior == "0" else 2**n - 1

        def main(n):
            qv = QuantumFloat(n)
            h(qv)
            flag = QuantumVariable(1)
            with control(measure(qv) == expected_value):
                x(flag[0])

        _check_metrics(main, (n,), Expected({"h": n, "measure": n, "x": 1}, 1, _qubits(n + 1)), meas_behavior)

    @pytest.mark.parametrize("meas_behavior", ["0", "1"])
    def test_teleportation(self, meas_behavior):
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

        expected_ops = {"h": 3, "cx": 2, "measure": 2}
        if meas_behavior == "1":
            expected_ops.update({"x": 1, "z": 1})
        _check_metrics(main, (), Expected(expected_ops, 4, _qubits(3)), meas_behavior)

    @pytest.mark.parametrize("d,rounds", [(3, 1), (4, 3), (5, 2)])
    def test_repetition_code_fresh_ancillas(self, d, rounds):
        """Syndrome extraction of a distance-d repetition code with fresh ancillas in every round.

        Each round allocates d - 1 ancillas, applies two layers of disjoint CNOTs
        (from ``data[:-1]`` and from ``data[1:]``), measures and deletes the ancillas.
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

        num_ancillas = rounds * (d - 1)
        expected_ops = {"cx": 2 * num_ancillas, "measure": num_ancillas}
        expected_qubits = _qubits(d + num_ancillas, num_ancillas, peak=2 * d - 1, final=d)
        _check_metrics(main, (d, rounds), Expected(expected_ops, 2 * rounds, expected_qubits))

    @pytest.mark.parametrize("d,rounds", [(3, 1), (4, 3)])
    def test_repetition_code_reset_ancillas(self, d, rounds):
        """The same syndrome extraction, reusing the ancillas with a reset after every round.

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

        num_measurements = rounds * (d - 1)
        expected_ops = {"cx": 2 * num_measurements, "measure": num_measurements}
        _check_metrics(main, (d, rounds), Expected(expected_ops, 2 * rounds, _qubits(2 * d - 1)))

    @pytest.mark.parametrize("meas_behavior", ["0", "1"])
    def test_repetition_code_decoding(self, meas_behavior):
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
            with control(s0 & s1):
                x(data[1])

        expected_ops = {"cx": 4, "measure": 2}
        if meas_behavior == "1":
            expected_ops["x"] = 1
        _check_metrics(main, (), Expected(expected_ops, 4, _qubits(6)), meas_behavior)
