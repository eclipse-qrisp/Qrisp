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

"""Tests the semiclassical order finding against exact results that do not depend on Qrisp."""

import math
import random
from collections import Counter

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    QFT,
    BigInteger,
    QuantumFloat,
    QuantumModulus,
    QuantumVariable,
    control,
    h,
    measure,
    measure_to_big_integer,
    p,
    x,
)
from qrisp.alg_primitives.iterative_qpe import _semiclassical_phase_estimation
from qrisp.algorithms.shor import semiclassical_order_finding
from qrisp.jasp import count_ops, jaspify, jaspr_to_static_register_jaspr, jrange, make_jaspr, q_fori_loop

# Largest total variation distance accepted between sampled and exact distributions
TOTAL_VARIATION_TOLERANCE = 0.1


def qpe_outcome_distribution(phases, precision):
    """Return the exact distribution of the outcome of the phase estimation of an equal mixture of eigenstates.

    For an eigenstate with phase ``phase`` (a fraction of a turn), the outcome ``y`` of
    the phase estimation with ``precision`` bits has the probability
    ``|sum_x exp(2 pi i x (phase - y / 2**t))|**2 / 4**t``.
    """
    outcomes = np.arange(2**precision)
    controls = np.arange(2**precision)
    distribution = np.zeros(2**precision)
    for phase in phases:
        amplitudes = np.exp(2j * np.pi * np.outer(phase - outcomes / 2**precision, controls)).sum(axis=1) / 2**precision
        distribution += np.abs(amplitudes) ** 2 / len(phases)
    return distribution


def make_phase_estimation(precision):
    """Return the semiclassical phase estimation of the phase gate p(2 pi phase), on its eigenstate |1>."""

    @jaspify
    def estimate(phase):
        qv = QuantumVariable(1)
        x(qv)

        def apply_power(qv, k):
            p(2 * np.pi * phase * 2.0**k, qv[0])

        return BigInteger(_semiclassical_phase_estimation(qv, apply_power, precision, outcome_limbs=True)[1])

    return estimate


@pytest.fixture
def seed_numpy():
    """Return np.random.seed, with which the simulator samples the measurements, and restore the state afterwards."""
    state = np.random.get_state()
    yield np.random.seed
    np.random.set_state(state)


@pytest.mark.parametrize("precision", [1, 3, 5])
def test_phase_estimation_exact_phases(precision):
    """A phase y / 2**t with t bits is estimated exactly, as y, for every y."""
    estimate = make_phase_estimation(precision)
    for y in range(2**precision):
        assert estimate(y / 2**precision)() == y, y


def test_phase_estimation_exact_phases_beyond_one_limb():
    """Outcomes of more than 32 bits are assembled correctly from their limbs, as for large moduli."""
    precision = 60

    @jaspify
    def estimate(numerator):
        qv = QuantumVariable(1)
        x(qv)

        def apply_power(qv, k):
            # The phase of U**(2**k) is (numerator * 2**k mod 2**t) / 2**t, computed exactly
            shifted = jnp.left_shift(jnp.asarray(numerator, dtype=jnp.uint64), jnp.asarray(k, dtype=jnp.uint64))
            phase = (shifted & jnp.uint64(2**precision - 1)).astype(jnp.float64) / 2**precision
            p(2 * np.pi * phase, qv[0])

        return BigInteger(_semiclassical_phase_estimation(qv, apply_power, precision, outcome_limbs=True)[1])

    rng = random.Random(precision)
    for y in (
        0,
        2**precision - 1,
        2**31,
        2**32,
        2**32 + 1,
        2**59 + 2**31,
        rng.getrandbits(precision),
        rng.getrandbits(precision),
    ):
        assert estimate(y)() == y, y


def test_phase_estimation_distribution_of_an_inexact_phase(seed_numpy):
    """For a phase that t bits cannot represent, the outcomes follow the distribution of the phase estimation."""
    precision, phase, shots = 4, 1 / 3, 400

    @jaspify
    def sample_counts():
        def shot(_, counts):
            qv = QuantumVariable(1)
            x(qv)

            def apply_power(qv, k):
                p(2 * np.pi * phase * 2.0**k, qv[0])

            outcome = _semiclassical_phase_estimation(qv, apply_power, precision, outcome_limbs=True)[1][0]
            x(qv)  # the eigenstate |1> is unchanged, so the qubit is released in |0>
            qv.delete()
            return counts.at[outcome].add(1)

        # One program for all shots, which is much faster than one program per shot
        return q_fori_loop(0, shots, shot, jnp.zeros(2**precision, dtype=jnp.int64))

    seed_numpy(2026)
    observed = np.asarray(sample_counts()) / shots
    expected = qpe_outcome_distribution([phase], precision)
    # The expected total variation distance of 400 samples is about 0.04
    assert 0.5 * np.abs(observed - expected).sum() < TOTAL_VARIATION_TOLERANCE


@pytest.mark.parametrize(
    "a, N, order",
    [(7, 15, 4), (4, 15, 2)],
    ids=["a=7, N=15", "a=4, N=15"],
)
def test_order_finding_outcomes(a, N, order, seed_numpy):
    """If the order r divides 2**t, every outcome is a multiple of 2**t / r.

    Every other outcome has probability 0, so a wrong multiplier, a wrong
    correction or a wrong bit order would show up as an outcome off the multiples.
    """
    precision = 2 * (N - 1).bit_length()
    assert pow(a, order, N) == 1 and (2**precision) % order == 0

    def main():
        return semiclassical_order_finding(a, N)

    run = jaspify(main)
    seed_numpy(a * N)
    # Each shot gives 0 with probability 1 / r, and a circuit that never applied the
    # multiplications would always give 0, so sample until another outcome appears
    max_shots = 12
    outcomes = [run()() for _ in range(3)]
    while not any(outcomes) and len(outcomes) < max_shots:
        outcomes.append(run()())
    assert any(outcomes), f"no outcome other than 0 in {outcomes}"
    assert all(outcome % (2**precision // order) == 0 for outcome in outcomes)


def qpe_order_finding(a, N, num_limbs):
    """Return the order-finding circuit with a phase register of 2n qubits and the inverse QFT."""
    precision = 2 * (N - 1).bit_length()

    def main():
        modulus = BigInteger.create_static(N, num_limbs)
        multiplier = BigInteger.create_static(a, num_limbs)
        qm = QuantumModulus(modulus)
        qm[:] = 1
        qpe_res = QuantumFloat(precision)
        h(qpe_res)
        for i in jrange(precision):
            with control(qpe_res[i]):
                qm *= multiplier
            multiplier = (multiplier * multiplier) % modulus
        QFT(qpe_res, inv=True)
        return measure_to_big_integer(qpe_res, num_limbs)

    return main


def random_inputs(bits, seed):
    """Return (a, N) with N odd of exactly ``bits`` bits and 1 < a < N coprime to N."""
    rng = random.Random(seed)
    while True:
        N = rng.randrange(2 ** (bits - 1) + 1, 2**bits, 2)
        a = rng.randrange(2, N)
        if math.gcd(a, N) == 1:
            return a, N


def test_resources_match_the_phase_register_circuit():
    """The semiclassical circuit differs from the circuit with a phase register only by its readout.

    Both apply the same 2n controlled multiplications. With t = 2n phase bits, the
    circuit with a phase register adds 2t Hadamard gates, the t (t - 1) / 2
    controlled phase gates (3 P and 2 CX each) and t / 2 swaps (3 CX each) of the
    inverse QFT, and t measurements. The semiclassical circuit adds two Hadamard
    gates, a rotation and a measurement per step. The resets of the control qubits
    and of the QuantumModulus are not counted. It needs 2n - 1 fewer qubits. At
    34 bits, the modulus and the multipliers need two limbs and the outcome three,
    as for large moduli. The resource estimators are compiled once per circuit and
    metric, whatever the size, so a single size keeps the test quick.
    """
    bits = 34
    a, N = random_inputs(bits, seed=bits)
    n, m = bits, math.ceil(math.log2(bits))
    t, num_limbs = 2 * n, -(-2 * n // 32)
    # Each circuit is traced once for both resource metrics
    semiclassical = make_jaspr(lambda: semiclassical_order_finding(a, N))()
    qpe = make_jaspr(qpe_order_finding(a, N, num_limbs))()

    expected = Counter(qpe.count_ops(meas_behavior="1"))
    expected.subtract({"h": 2 * t, "p": 3 * t * (t - 1) // 2, "cx": t * (t - 1) + 3 * (t // 2), "measure": t})
    expected.update({"h": 2 * t, "rz": t, "measure": t})
    assert semiclassical.count_ops(meas_behavior="1") == {name: count for name, count in expected.items() if count}

    semiclassical_qubits = semiclassical.num_qubits(meas_behavior="1")
    qpe_qubits = qpe.num_qubits(meas_behavior="1")
    assert semiclassical_qubits["peak_allocations"] == 3 * n + 2 * m + 2
    assert semiclassical_qubits["peak_allocations"] == qpe_qubits["peak_allocations"] - (t - 1)
    # The control qubits are allocated one after the other instead of all at once
    assert semiclassical_qubits["total_allocated"] == qpe_qubits["total_allocated"]
    # The semiclassical order finding also releases its QuantumModulus
    assert semiclassical_qubits["finally_allocated"] == 0
    assert qpe_qubits["finally_allocated"] == n + t


def test_custom_precision():
    """The precision sets the number of steps, not the number of qubits."""
    precision = 11
    n, m = 4, 2  # N = 15 has 4 bits

    jaspr = make_jaspr(lambda: semiclassical_order_finding(7, 15, precision=precision))()
    assert jaspr.count_ops(meas_behavior="1")["rz"] == precision
    assert jaspr.num_qubits(meas_behavior="1")["peak_allocations"] == 3 * n + 2 * m + 2


def test_qubits_are_reset_before_release():
    """The QuantumModulus is reset before it is released, so that later allocations can reuse its qubits.

    A static register of three qubits, the peak for a = 1 and N = 3 (whose trivial
    multiplications allocate nothing), gives the released qubits to the next allocation
    without resetting them. The phase estimation leaves the QuantumModulus in |1> here.
    """

    def main():
        semiclassical_order_finding(1, 3)
        return measure(QuantumFloat(3))

    assert jaspr_to_static_register_jaspr(make_jaspr(main)(), 3)() == 0


def test_custom_adder():
    """The adder is passed on to the modular multiplications."""

    class AdderCalled(Exception):
        """Raised by the adder, which shows that the multiplications use it."""

    def adder(*_args, **_kwargs):
        raise AdderCalled

    with pytest.raises(AdderCalled):
        make_jaspr(lambda: semiclassical_order_finding(7, 15, inpl_adder=adder))()


@pytest.mark.parametrize(
    "a, N, precision, error",
    [
        (2, 16, None, ValueError),
        (2, 1, None, ValueError),
        (3, 15, None, ValueError),
        (2, 15, 0, ValueError),
        (2.0, 15, None, TypeError),
        (2, 15, 2.0, TypeError),
    ],
    ids=["even N", "N < 3", "a not coprime", "precision 0", "float a", "float precision"],
)
def test_invalid_inputs(a, N, precision, error):
    """Invalid inputs raise an error before any circuit is built."""
    with pytest.raises(error):
        count_ops(meas_behavior="1")(lambda: semiclassical_order_finding(a, N, precision))()


def test_requires_jasp_mode():
    """Outside of Jasp mode, the function raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="Jasp mode"):
        semiclassical_order_finding(7, 15)
