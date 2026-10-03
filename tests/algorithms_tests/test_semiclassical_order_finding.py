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
    measure_to_big_integer,
    p,
    x,
)
from qrisp.alg_primitives.iterative_qpe import _semiclassical_phase_estimation
from qrisp.algorithms.shor import semiclassical_order_finding
from qrisp.algorithms.shor.order_finding import _to_limbs
from qrisp.jasp import count_ops, jaspify, jrange, num_qubits

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

        return _semiclassical_phase_estimation(qv, apply_power, precision, num_limbs=-(-precision // 32))[1]

    return estimate


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

        return _semiclassical_phase_estimation(qv, apply_power, precision, num_limbs=-(-precision // 32))[1]

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


def test_phase_estimation_distribution_of_an_inexact_phase():
    """For a phase that t bits cannot represent, the outcomes follow the distribution of the phase estimation."""
    precision, phase, shots = 4, 1 / 3, 400
    estimate = make_phase_estimation(precision)
    np.random.seed(2026)  # the simulator samples the measurements with NumPy
    counts = Counter(estimate(phase)() for _ in range(shots))

    expected = qpe_outcome_distribution([phase], precision)
    observed = np.array([counts[y] for y in range(2**precision)]) / shots
    # The expected total variation distance of 400 samples is about 0.04
    assert 0.5 * np.abs(observed - expected).sum() < TOTAL_VARIATION_TOLERANCE


@pytest.mark.parametrize(
    "a, N, order",
    [(2, 5, 4), (7, 15, 4), (4, 15, 2)],
    ids=["a=2, N=5", "a=7, N=15", "a=4, N=15"],
)
def test_order_finding_outcomes(a, N, order):
    """If the order r divides 2**t, every outcome is a multiple of 2**t / r.

    Every other outcome has probability 0, so a wrong multiplier, a wrong
    correction or a wrong bit order would show up as an outcome off the multiples.
    """
    precision = 2 * (N - 1).bit_length()
    assert pow(a, order, N) == 1 and (2**precision) % order == 0

    def main():
        return semiclassical_order_finding(a, N)

    run = jaspify(main)
    np.random.seed(a * N)
    for _ in range(3):
        assert run()() % (2**precision // order) == 0


def qpe_order_finding(a, N, num_limbs):
    """Return the order-finding circuit with a phase register of 2n qubits and the inverse QFT."""
    precision = 2 * (N - 1).bit_length()

    def main():
        modulus = BigInteger.create_static(N, num_limbs)
        multiplier = BigInteger.create_static(a, num_limbs)
        register = QuantumModulus(modulus)
        register[:] = 1
        phase_register = QuantumFloat(precision)
        h(phase_register)
        for i in jrange(precision):
            with control(phase_register[i]):
                register *= multiplier
            multiplier = (multiplier * multiplier) % modulus
        QFT(phase_register, inv=True)
        return measure_to_big_integer(phase_register, num_limbs)

    return main


def qpe_readout(precision, num_limbs):
    """Return the readout of the circuit with a phase register: Hadamard gates, inverse QFT and measurement."""

    def main():
        phase_register = QuantumFloat(precision)
        h(phase_register)
        QFT(phase_register, inv=True)
        return measure_to_big_integer(phase_register, num_limbs)

    return main


def semiclassical_readout(precision):
    """Return the readout of the semiclassical circuit: its steps without the controlled multiplications."""

    def main():
        def apply_power(_qv, _k):
            pass

        return _semiclassical_phase_estimation(
            QuantumVariable(1), apply_power, precision, num_limbs=-(-precision // 32)
        )[1]

    return main


def random_inputs(bits, seed):
    """Return (a, N) with N odd of exactly ``bits`` bits and 1 < a < N coprime to N."""
    rng = random.Random(seed)
    while True:
        N = rng.randrange(2 ** (bits - 1) + 1, 2**bits, 2)
        a = rng.randrange(2, N)
        if math.gcd(a, N) == 1:
            return a, N


@pytest.mark.parametrize("bits", [4, 16])
def test_resources_match_the_phase_register_circuit(bits):
    """The semiclassical circuit differs from the circuit with a phase register only by its readout.

    Both apply the same 2n controlled multiplications. The circuit with a phase
    register adds the Hadamard gates of the register, the inverse QFT and the
    measurements, the semiclassical circuit two Hadamard gates, a rotation and a
    measurement per step. It needs 2n - 1 fewer qubits.
    """
    a, N = random_inputs(bits, seed=bits)
    n, m = bits, math.ceil(math.log2(bits))
    precision, num_limbs = 2 * n, -(-2 * n // 32)

    def semiclassical():
        return semiclassical_order_finding(a, N)

    qpe = qpe_order_finding(a, N, num_limbs)
    gates = count_ops(meas_behavior="1")
    readout = gates(semiclassical_readout(precision))()
    assert readout == {"h": 2 * precision, "rz": precision, "measure": precision}
    expected = Counter(gates(qpe)())
    expected.subtract(gates(qpe_readout(precision, num_limbs))())
    expected.update(readout)
    assert gates(semiclassical)() == {name: count for name, count in expected.items() if count}

    qubits = num_qubits(meas_behavior="1")
    semiclassical_qubits, qpe_qubits = qubits(semiclassical)(), qubits(qpe)()
    assert semiclassical_qubits["peak_allocations"] == 3 * n + 2 * m + 2
    assert semiclassical_qubits["peak_allocations"] == qpe_qubits["peak_allocations"] - (precision - 1)
    # The control qubits are allocated one after the other instead of all at once
    assert semiclassical_qubits["total_allocated"] == qpe_qubits["total_allocated"]
    assert semiclassical_qubits["finally_allocated"] == n
    assert qpe_qubits["finally_allocated"] == n + precision


def test_custom_precision():
    """The precision sets the number of steps, not the number of qubits."""
    precision = 11
    n, m = 4, 2  # N = 15 has 4 bits

    def main():
        return semiclassical_order_finding(7, 15, precision=precision)

    assert count_ops(meas_behavior="1")(main)()["rz"] == precision
    assert num_qubits(meas_behavior="1")(main)()["peak_allocations"] == 3 * n + 2 * m + 2


def test_to_limbs_matches_biginteger():
    """The limbs of the multipliers are those of a BigInteger."""
    rng = random.Random(0)
    for num_limbs in (1, 3, 8):
        for value in (0, 1, 2 ** (32 * num_limbs) - 1, rng.getrandbits(32 * num_limbs)):
            assert np.array_equal(
                _to_limbs(value, num_limbs), np.asarray(BigInteger.create_static(value, num_limbs).digits)
            )


@pytest.mark.parametrize(
    "a, N, precision, error",
    [
        (2, 16, None, ValueError),
        (2, 1, None, ValueError),
        (3, 15, None, ValueError),
        (2, 15, 0, ValueError),
        (2.0, 15, None, TypeError),
    ],
    ids=["even N", "N < 3", "a not coprime", "precision 0", "float a"],
)
def test_invalid_inputs(a, N, precision, error):
    """Invalid inputs raise an error before any circuit is built."""
    with pytest.raises(error):
        count_ops(meas_behavior="1")(lambda: semiclassical_order_finding(a, N, precision))()


def test_requires_jasp_mode():
    """Outside of Jasp mode, the function raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="Jasp mode"):
        semiclassical_order_finding(7, 15)
