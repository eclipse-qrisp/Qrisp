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

"""Tests for Jasp Montgomery modular multiplication and order-finding via QPE."""

import math
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    QFT,
    BigInteger,
    QuantumArray,
    QuantumBool,
    QuantumFloat,
    QuantumModulus,
    best_montgomery_shift,
    boolean_simulation,
    control,
    fourier_adder,
    gidney_adder,
    h,
    jasp_fourier_adder,
    jrange,
    measure,
    modinv,
    multi_measurement,
    terminal_sampling,
    x,
)
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
    bi_pow2mod,
    egcd,
    montgomery_decoder,
    montgomery_encoder,
    new_montgomery_decoder,
    pow2_mod_N,
    smallest_power_of_two,
)
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import modinv as jasp_modinv
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import (
    cq_montgomery_multiply,
    cq_montgomery_multiply_inplace,
    qq_montgomery_multiply,
    qq_montgomery_multiply_modulus,
)


@pytest.mark.parametrize(
    "n, N, shift",
    [(10, 97, 4), (5, 3, 2), (32, 3221225473, 5), (62, 2**62 - 57, 6)],
    ids=["n=10, N=97", "n=5, N=3", "32-bit N", "62-bit N"],
)
def test_best_montgomery_shift(n, N, shift):
    """The shift is ceil(log2(ceil(n (N - 1) / N))) in Python and under tracing, also for moduli close to 2**63.

    Regression test: under tracing, n * (N - 1) was computed in int64 and
    overflowed for large moduli, which gave a wrong shift.
    """
    assert best_montgomery_shift(n, N) == shift
    assert int(boolean_simulation(best_montgomery_shift)(n, N)) == shift


def test_montgomery_jasp_qq():
    @boolean_simulation
    def qq(a, b, n, N):
        qa = QuantumFloat(n)
        qa[:] = a
        qb = QuantumFloat(n)
        qb[:] = b
        m = best_montgomery_shift(N)
        res = qq_montgomery_multiply(qa, qb, N, m, gidney_adder)
        return measure(qa), measure(qb), measure(res)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        q = modinv(2**n, N)
        for a in range(1, 50, 3):
            for b in range(1, 50, 5):
                if a % N != 0 and b % N != 0:
                    ar, br, rr = qq(a % N, b % N, n, N)
                    assert ar == a % N
                    assert br == b % N
                    assert rr == (ar * br * q) % N


def test_montgomery_not_jasp_qq():
    X = 29
    y = 21
    N = 31
    n = 5

    def test_qq_g():
        qx = QuantumFloat(n)
        qx[:] = X
        qy = QuantumFloat(n)
        qy[:] = y
        m = best_montgomery_shift(N)
        res = qq_montgomery_multiply(qx, qy, N, m, gidney_adder)
        return multi_measurement([res])

    m = best_montgomery_shift(N)

    assert test_qq_g()[((X * y * modinv(2**m, N)) % N,)] == 1.0


def test_montgomery_jasp_cq():
    @boolean_simulation
    def cq(a, b, n, N):
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(n, N)  # one partial product per qubit of qb
        res = cq_montgomery_multiply(a, qb, N, shift, gidney_adder)
        return measure(qb), measure(res)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                if a % N != 0 and b % N != 0:
                    br, rr = cq(a % N, b % N, n, N)
                    assert br == b % N
                    assert rr == ((a % N) * br) % N


def test_montgomery_jasp_cq_inplace():
    @boolean_simulation
    def icq(a, b, n, N):
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(n, N)  # one partial product per qubit of qb
        cq_montgomery_multiply_inplace(a, qb, N, shift, gidney_adder)
        return measure(qb)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        q = jasp_modinv(2**n, N)
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                if a % N != 0 and b % N != 0 and np.gcd(a, N) == 1:
                    br = icq(a % N, b % N, n, N)
                    assert br == ((a % N) * (b % N)) % N


def test_montgomery_jasp_cq_inplace_controlled():
    @boolean_simulation
    def cicq(a, b, n, N, c):
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(n, N)  # one partial product per qubit of qb
        qc = QuantumBool()
        qc[:] = c
        with control(qc[0]):
            cq_montgomery_multiply_inplace(a, qb, N, shift, gidney_adder)
        return measure(qb)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        q = jasp_modinv(2**n, N)
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                for c in [0, 1]:
                    if a % N != 0 and b % N != 0 and np.gcd(a, N) == 1:
                        br = cicq(a % N, b % N, n, N, c)
                        assert br == (((a % N) ** c) * (b % N)) % N


def test_montgomery_jasp_cq_inplace_bi():
    @boolean_simulation
    def bicicq(a, b, n, N, c):
        a = BigInteger.create(a, 3)
        N = BigInteger.create(N, 3)
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(n, N)  # one partial product per qubit of qb
        qc = QuantumBool()
        qc[:] = c
        with control(qc[0]):
            cq_montgomery_multiply_inplace(a, qb, N, shift, gidney_adder)
        return measure(qb)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                for c in [0, 1]:
                    if a % N != 0 and b % N != 0 and np.gcd(a, N) == 1:
                        br = bicicq(a % N, b % N, n, N, c)
                        assert br == (((a % N) ** c) * (b % N)) % N


def test_montgomery_find_order():
    def find_order(a, N, inpl_adder):
        qg = QuantumModulus(N, inpl_adder)
        qg[:] = 1
        qpe_res = QuantumFloat(2 * qg.size + 1, exponent=-(2 * qg.size + 1))
        h(qpe_res)
        for i in range(len(qpe_res)):
            with control(qpe_res[i]):
                qg *= a
                a = (a * a) % N
        QFT(qpe_res, inv=True)
        return qpe_res.get_measurement()

    dict_norm_fourier = find_order(4, 13, fourier_adder)
    dict_norm_gidney = find_order(4, 13, gidney_adder)

    @terminal_sampling
    def find_order(a, N, inpl_adder):
        qg = QuantumModulus(N, inpl_adder=inpl_adder)
        x(qg[0])
        qpe_res = QuantumFloat(2 * qg.size + 1, exponent=-(2 * qg.size + 1))
        h(qpe_res)
        for i in jrange(qpe_res.size):
            with control(qpe_res[i]):
                qg *= a
            a = (a * a) % N
        QFT(qpe_res, inv=True)
        return qpe_res

    dict_jasp_fourier = find_order(4, 13, jasp_fourier_adder)
    dict_jasp_gidney = find_order(4, 13, gidney_adder)

    dict_bim_fourier = find_order(BigInteger.create(4, 1), BigInteger.create(13, 1), jasp_fourier_adder)
    dict_bim_gidney = find_order(BigInteger.create(4, 1), BigInteger.create(13, 1), gidney_adder)

    def check_dict_equality(a, b):
        for key in a.keys():
            if not np.allclose(a[key], b.get(key, -1), rtol=0.001, atol=0.001):
                return False
        return True

    assert check_dict_equality(dict_norm_fourier, dict_norm_gidney)
    assert check_dict_equality(dict_jasp_fourier, dict_jasp_gidney)
    assert check_dict_equality(dict_bim_fourier, dict_bim_gidney)

    assert check_dict_equality(dict_norm_gidney, dict_jasp_gidney)
    assert check_dict_equality(dict_jasp_gidney, dict_bim_gidney)


def test_egcd_bezout_identity():
    """`egcd` must return the gcd and Bézout coefficients satisfying a*x + b*y = gcd(a, b)."""
    for a, b in [(35, 15), (240, 46), (17, 5), (100, 1)]:
        g, x, y = egcd(a, b)
        g, x, y = int(g), int(x), int(y)
        assert g == math.gcd(a, b)
        assert a * x + b * y == g


def test_bi_pow2mod_matches_python_pow():
    """`bi_pow2mod` must compute 2**exp mod m as a BigInteger, matching Python's pow."""
    for exp, mod in [(10, 97), (0, 97), (1, 3), (37, 1009)]:
        mod_bi = BigInteger.create_static(mod, 4)
        result = bi_pow2mod(exp, mod_bi)
        assert result() == pow(2, exp, mod)


def test_pow2_mod_n_traced_matches_python_pow():
    """`pow2_mod_N` must compute 2**exp mod N under jax.jit tracing, matching Python's pow."""
    traced = jax.jit(pow2_mod_N)
    for exp, mod in [(10, 97), (0, 97), (37, 1009)]:
        assert int(traced(exp, mod)) == pow(2, exp, mod)


def test_pow2_mod_n_large_moduli():
    """`pow2_mod_N` matches Python's pow for moduli of 33 to 63 bits.

    Regression test: the products were computed in int64, which overflowed for
    moduli above 2**31.5.
    """
    rng = random.Random(0)
    cases = [
        (rng.randrange(200), rng.randrange(2 ** (bits - 1) + 1, 2**bits, 2)) for bits in range(33, 64) for _ in range(8)
    ]
    results = jax.vmap(pow2_mod_N)(jnp.array([e for e, _ in cases]), jnp.array([n for _, n in cases]))
    assert results.tolist() == [pow(2, e, n) for e, n in cases]


@pytest.mark.parametrize("bits", [32, 40, 50, 62])
def test_traced_montgomery_encoder_and_decoder_large_moduli(bits):
    """Under tracing, `montgomery_encoder` and `montgomery_decoder` match the Python results for large moduli.

    Regression test: x * R was computed in int64, which overflowed for moduli above 2**31.5.
    """
    rng = random.Random(bits)
    N = rng.randrange(2 ** (bits - 1) + 1, 2**bits, 2)
    encode = boolean_simulation(montgomery_encoder)
    decode = boolean_simulation(montgomery_decoder)
    for _ in range(5):
        x, R = rng.randrange(N), rng.randrange(1, N)
        assert int(encode(x, R, N)) == x * R % N
        assert int(decode(x, 8, N)) == montgomery_decoder(x, 8, N)


def test_traced_smallest_power_of_two_exact():
    """Under tracing, `smallest_power_of_two` gives ceil(log2(n)) exactly around every power of two below 2**63.

    Regression test: a float log2 gave one bit too few for n = 2**k + 1 with k >= 49.
    """
    traced = boolean_simulation(smallest_power_of_two)
    for k in range(1, 63):
        for n in (2**k - 1, 2**k, 2**k + 1):
            if n < 2**63:
                assert int(traced(n)) == smallest_power_of_two(n), n
    assert int(traced(0)) == 0
    assert int(traced(1)) == 0


def test_smallest_power_of_two_bigint_matches_int_at_powers_of_two():
    """`smallest_power_of_two` must agree between the int and BigInteger paths, including at exact powers of two."""  # {1, 2, 4, 8, 16, 1024} in addition to non-power-of-two values and n=0.
    for n in [0, 1, 2, 3, 4, 7, 8, 15, 16, 100, 1023, 1024]:
        expected = smallest_power_of_two(n)
        assert int(smallest_power_of_two(BigInteger.create(n, 4))) == expected


def test_montgomery_encoder_decoder_mixed_bigint_roundtrip():
    """`montgomery_encoder`/`montgomery_decoder` round-trip a BigInteger with plain-int args."""
    radix, modulus, x = 1024, 97, 42
    x_bi = BigInteger.create(x, 4)
    encoded = montgomery_encoder(x_bi, radix, modulus)
    assert isinstance(encoded, BigInteger)
    assert encoded() == (x * radix) % modulus
    decoded = montgomery_decoder(encoded, radix, modulus)
    assert isinstance(decoded, BigInteger)
    assert decoded() == x


def test_montgomery_encoder_rejects_mismatched_bigint_widths():
    """`montgomery_encoder` must reject BigIntegers with different limb widths, not silently return a wrong result."""
    with pytest.raises(ValueError):
        montgomery_encoder(BigInteger.create(42, 4), BigInteger.create(1024, 8), 97)


def test_new_montgomery_decoder_positive_and_negative_shift():
    """`new_montgomery_decoder` must decode both positive (inverse) and non-positive shifts."""
    modulus, x, m = 97, 55, 10
    encoded = (pow(2, m, modulus) * x) % modulus
    assert new_montgomery_decoder(encoded, m, modulus) == x
    # Non-positive shift decodes with 2**abs(m) mod N instead of its inverse
    assert new_montgomery_decoder(x, -3, modulus) == (x * pow(2, 3, modulus)) % modulus


def test_qq_montgomery_multiply_modulus():
    """`qq_montgomery_multiply_modulus` must compute the montgomery product of two QuantumModuli."""
    qx = QuantumModulus(97, inpl_adder=gidney_adder)
    qy = QuantumModulus(97, inpl_adder=gidney_adder)
    qx[:] = 12
    qy[:] = 7
    res = qq_montgomery_multiply_modulus(qx, qy)
    assert multi_measurement([qx, qy, res]) == {(12, 7, 84): 1.0}


def test_cq_montgomery_mat_multiply():
    """`QuantumArray @ np.ndarray` must work for a standard numpy integer matrix."""
    modulus = 7
    a_array = QuantumArray(qtype=QuantumModulus(modulus, inpl_adder=gidney_adder), shape=(2, 2))
    a_array[:] = np.array([[1, 2], [3, 4]])
    b_array = np.array([[1, 2], [3, 4]])
    r_array = a_array @ b_array
    (outcome,) = list(multi_measurement([r_array]).keys())
    assert outcome[0].tolist() == [[0, 3], [1, 1]]
