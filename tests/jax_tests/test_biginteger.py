"""
********************************************************************************
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

import random
import math
import numpy as np
import pytest
import jax.numpy as jnp

from qrisp.alg_primitives.arithmetic.jasp_arithmetic import (
    BigInteger,
    bi_modinv,
    bi_montgomery_encode,
    bi_montgomery_decode,
    bi_extended_euclidean
)

from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BASE

# ----------------- Global parameter lists -----------------
SEEDS = [0, 1, 2]

CREATE_STATIC_SIZES = [4, 8]
CREATE_STATIC_SEEDS = [0, 1, 2]

CREATE_DYNAMIC_SIZES = [4, 8]
CREATE_DYNAMIC_SEEDS = [3, 4, 5]

FLOAT_CREATE_SIZE = [4] # Must remain small
FLOAT_CREATE_SEEDS = [6, 7, 8]

ADD_SIZES = [2, 4, 8]
ADD_SEEDS = [20, 21, 22, 23]

SUB_SIZES = [2, 4, 8]
SUB_SEEDS = [30, 31, 32, 33]

MUL_SIZES = [2, 4, 8]
MUL_SEEDS = [40, 41, 42, 43]

LSHIFT_SIZES = [1, 2, 4]
LSHIFT_SHIFTS = [0, 1, 7, 31, 32, 33, 63, 64, 95]
LSHIFT_SEEDS = [50, 51]

RSHIFT_SIZES = [1, 2, 4]
RSHIFT_SHIFTS = [0, 1, 7, 31, 32, 33, 63, 64, 95]
RSHIFT_SEEDS = [52, 53]

BITWISE_SIZES = [2, 4, 8]
BITWISE_SEEDS = [60, 61]

CMP_SIZES = [2, 4]
CMP_SEEDS = [70, 71, 72]

POW_SIZES = [2, 4]
POW_EXPS = [0, 1, 5]
POW_SEEDS = [75, 76, 77]

BITLEVEL_SIZES = [2, 3]
BITLEVEL_SEEDS = [80, 81]

BIT_SIZE_SIZES = [2, 4]
BIT_SIZE_SEEDS = [90, 91, 92]

DIVMOD_SIZES_A = [5, 25, 30]
DIVMOD_SIZES_B = [5, 23, 27]
DIVMOD_SEEDS = [100, 102]

MODINV_SIZE = [10, 11, 12]
MODINV_SEEDS = [110, 111, 112]

MONT_SIZE = [2]
MONT_SEEDS = [120, 121]

CALL_SIZE = [2] # Must remain small
CALL_SEEDS = [130, 131]

EEA_SIZES = [2, 3]
EEA_SEEDS = [160, 161, 162]


# ----------------- Helpers -----------------

def mask_for_size(size: int) -> int:
    return (1 << (32 * size)) - 1


def _digits_array(maybe_bi):
    """
    Return the digits array whether input is a BigInteger or a raw JAX/Numpy array.
    """
    if isinstance(maybe_bi, (jnp.ndarray, np.ndarray)):
        return maybe_bi
    if hasattr(maybe_bi, "digits"):
        return maybe_bi.digits
    raise TypeError(f"Unexpected type for digits extraction: {type(maybe_bi)}")


def to_int(maybe_bi) -> int:
    """
    Convert a BigInteger (or its digits array) to a Python int exactly.
    """
    ds = np.asarray(_digits_array(maybe_bi), dtype=np.uint64).tolist()
    acc = 0
    for i, d in enumerate(ds):
        acc |= int(d) << (32 * i)
    return acc


def limbs_to_int(limbs) -> int:
    """
    Convert a list of uint32 limbs (little-endian) to a Python int.
    """
    acc = 0
    for i, d in enumerate(limbs):
        acc |= (int(d) & 0xFFFFFFFF) << (32 * i)
    return acc


def random_limbs(size):
    return [random.getrandbits(32) for _ in range(size)]


def mk_bigint_dynamic_by_base_digits(digits_le, size):
    """
    Dynamically build a BigInteger from base-2^32 digits using only
    small create(...) calls and multiplication by BASE, avoiding passing
    huge Python ints to create(...).

    x = 0
    for d in reversed(digits_le):
        x = x * BASE
        x = x + d
    """
    x = BigInteger.create(0, size)
    base = BigInteger.create(BASE, size)  # small, safe for dynamic create
    for d in reversed(digits_le):
        x = x * base
        x = x + BigInteger.create(int(d), size)
    return x


# ----------------- Creation tests -----------------

@pytest.mark.parametrize("size", CREATE_STATIC_SIZES)
@pytest.mark.parametrize("seed", CREATE_STATIC_SEEDS)
def test_create_static_large_exact(size, seed):
    random.seed(seed)
    limbs = random_limbs(size)
    iv = limbs_to_int(limbs)
    bi_static = BigInteger.create_static(iv, size)
    assert to_int(bi_static) == (iv & mask_for_size(size))


@pytest.mark.parametrize("size", CREATE_DYNAMIC_SIZES)
@pytest.mark.parametrize("seed", CREATE_DYNAMIC_SEEDS)
def test_create_dynamic_by_multiplication_large_exact(size, seed):
    random.seed(seed)
    limbs = random_limbs(size)
    iv = limbs_to_int(limbs)
    bi_dyn = mk_bigint_dynamic_by_base_digits(limbs, size)
    assert to_int(bi_dyn) == (iv & mask_for_size(size))


@pytest.mark.parametrize("size", FLOAT_CREATE_SIZE)
@pytest.mark.parametrize("seed", FLOAT_CREATE_SEEDS)
def test_create_from_float_small_exact(size, seed):
    random.seed(seed)
    iv = random.randrange(0, 1 << 50)
    fv = float(iv)
    bi = BigInteger.create(fv, size=size)
    assert to_int(bi) == iv


# ----------------- Basic arithmetic -----------------

@pytest.mark.parametrize("size", ADD_SIZES)
@pytest.mark.parametrize("seed", ADD_SEEDS)
def test_add_random(size, seed):
    random.seed(seed)
    M = mask_for_size(size)
    limbs_a = random_limbs(size)
    limbs_b = random_limbs(size)
    ia, ib = limbs_to_int(limbs_a), limbs_to_int(limbs_b)
    A = BigInteger.create_static(ia, size)
    B = BigInteger.create_static(ib, size)
    res = A + B
    assert to_int(res) == ((ia + ib) & M)


@pytest.mark.parametrize("size", SUB_SIZES)
@pytest.mark.parametrize("seed", SUB_SEEDS)
def test_sub_random_wraparound(size, seed):
    random.seed(seed)
    M = mask_for_size(size)
    limbs_a = random_limbs(size)
    limbs_b = random_limbs(size)
    ia, ib = limbs_to_int(limbs_a), limbs_to_int(limbs_b)
    A = BigInteger.create_static(ia, size)
    B = BigInteger.create_static(ib, size)
    res = A - B
    assert to_int(res) == ((ia - ib) & M)


@pytest.mark.parametrize("size", MUL_SIZES)
@pytest.mark.parametrize("seed", MUL_SEEDS)
def test_mul_random(size, seed):
    random.seed(seed)
    M = mask_for_size(size)
    # Use full-width random limbs for both operands now that mul is fixed
    limbs_a = random_limbs(size)
    limbs_b = random_limbs(size)
    ia, ib = limbs_to_int(limbs_a), limbs_to_int(limbs_b)
    A = BigInteger.create_static(ia, size)
    B = BigInteger.create_static(ib, size)
    res = A * B
    assert to_int(res) == ((ia * ib) & M)


# ----------------- Shifts -----------------

@pytest.mark.parametrize("size", LSHIFT_SIZES)
@pytest.mark.parametrize("shift", LSHIFT_SHIFTS)
@pytest.mark.parametrize("seed", LSHIFT_SEEDS)
def test_lshift(size, shift, seed):
    random.seed(seed)
    total_bits = 32 * size
    M = mask_for_size(size)
    # Build a value with lower half bits to observe shifting effect
    limbs = random_limbs(size)
    for i in range(size // 2, size):
        limbs[i] = 0
    iv = limbs_to_int(limbs)
    A = BigInteger.create_static(iv, size)
    res = A << shift
    expected = ((iv << shift) & M) if shift < total_bits else 0
    assert to_int(res) == expected


@pytest.mark.parametrize("size", RSHIFT_SIZES)
@pytest.mark.parametrize("shift", RSHIFT_SHIFTS)
@pytest.mark.parametrize("seed", RSHIFT_SEEDS)
def test_rshift(size, shift, seed):
    random.seed(seed)
    total_bits = 32 * size
    M = mask_for_size(size)
    limbs = random_limbs(size)
    iv = limbs_to_int(limbs)
    A = BigInteger.create_static(iv, size)
    res = A >> shift
    expected = (iv >> shift) if shift < total_bits else 0
    assert to_int(res) == (expected & M)


# ----------------- Bitwise ops -----------------

@pytest.mark.parametrize("size", BITWISE_SIZES)
@pytest.mark.parametrize("seed", BITWISE_SEEDS)
def test_bitwise_ops(size, seed):
    random.seed(seed)
    M = mask_for_size(size)
    limbs_a = random_limbs(size)
    limbs_b = random_limbs(size)
    ia, ib = limbs_to_int(limbs_a), limbs_to_int(limbs_b)
    A = BigInteger.create_static(ia, size)
    B = BigInteger.create_static(ib, size)
    assert to_int(A & B) == ((ia & ib) & M)
    assert to_int(A | B) == ((ia | ib) & M)
    assert to_int(A ^ B) == ((ia ^ ib) & M)
    assert to_int(~A) == ((~ia) & M)


# ----------------- Comparisons -----------------

@pytest.mark.parametrize("size", CMP_SIZES)
@pytest.mark.parametrize("seed", CMP_SEEDS)
def test_comparisons(size, seed):
    random.seed(seed)
    limbs_a = random_limbs(size)
    limbs_b = random_limbs(size)
    ia, ib = limbs_to_int(limbs_a), limbs_to_int(limbs_b)
    if ia == ib:
        ib ^= 1
    A = BigInteger.create_static(ia, size)
    B = BigInteger.create_static(ib, size)
    assert bool(A < B) == (ia < ib)
    assert bool(A <= B) == (ia <= ib)
    assert bool(A > B) == (ia > ib)
    assert bool(A >= B) == (ia >= ib)
    assert bool(A == B) == (ia == ib)
    assert bool(A != B) == (ia != ib)


# ----------------- Power -----------------

@pytest.mark.parametrize("size", POW_SIZES)
@pytest.mark.parametrize("exp", POW_EXPS)
@pytest.mark.parametrize("seed", POW_SEEDS)
def test_pow_edges_and_small(size, exp, seed):
    random.seed(seed)
    M = mask_for_size(size)
    limbs = random_limbs(size)
    ia = limbs_to_int(limbs)
    A = BigInteger.create_static(ia, size)
    res = A ** exp
    # exact modulo 2^(32*size)
    expected = pow(ia, exp, 1 << (32 * size))
    assert to_int(res) == (expected & M)


# ----------------- Bit-level helpers -----------------

@pytest.mark.parametrize("size", BITLEVEL_SIZES)
@pytest.mark.parametrize("seed", BITLEVEL_SEEDS)
def test_get_bit_and_flip_bit(size, seed):
    random.seed(seed)
    total_bits = 32 * size
    # Choose a sparse set of bit positions including boundaries
    bits = [0, 1, 31, 32, total_bits - 2, total_bits - 1]
    x = BigInteger.create_static(0, size)
    for i in bits:
        x = x.flip_bit(i)
    reads = [int(x.get_bit(i)) for i in bits]
    assert all(v == 1 for v in reads)
    expected = 0
    for i in bits:
        expected |= (1 << i)
    assert to_int(x) == expected


@pytest.mark.parametrize("size", BIT_SIZE_SIZES)
@pytest.mark.parametrize("seed", BIT_SIZE_SEEDS)
def test_bit_size_zero_and_random(size, seed):
    random.seed(seed)
    assert int(BigInteger.create_static(0, size).bit_size()) == 0
    # Random non-zero values
    limbs = random_limbs(size)
    if all(l == 0 for l in limbs):
        limbs[0] = 1
    iv = limbs_to_int(limbs)
    A = BigInteger.create_static(iv, size)
    bs = int(A.bit_size())
    assert bs == iv.bit_length()


# ----------------- Division and modulo (limited due to float approximation) -----------------

@pytest.mark.parametrize("size_a", DIVMOD_SIZES_A)
@pytest.mark.parametrize("size_b", DIVMOD_SIZES_B)
@pytest.mark.parametrize("seed", DIVMOD_SEEDS)
def test_div_mod(size_a, size_b, seed):
    random.seed(seed)
    size = max(size_a, size_b)
    M = (1 << (32 * size)) - 1
    # Full-width random operands (ensure divisor != 0)
    a = random.getrandbits(32 * size_a - 1) | 1
    b = random.getrandbits(32 * size_b)

    A = BigInteger.create_static(a, size)
    B = BigInteger.create_static(b, size)

    q_bi = B // A
    r_bi = B % A
    q = to_int(q_bi)
    r = to_int(r_bi)

    # Reference with Python ints
    exp_q = b // a
    exp_r = b % a

    assert q == exp_q
    assert r == exp_r
    assert ((b & M) == ((a * q + r) & M))


# ----------------- Modular inverse -----------------

@pytest.mark.parametrize("size", MODINV_SIZE)
@pytest.mark.parametrize("seed", MODINV_SEEDS)
def test_modinv_small_prime_modulus(size, seed):
    random.seed(seed)
    m = 1_000_003  # prime
    a = random.randrange(1, m)
    A = BigInteger.create_static(a, size)
    M = BigInteger.create_static(m, size)
    inv_bi = bi_modinv(A, M)
    inv = to_int(inv_bi)
    assert (a * inv) % m == 1


# ----------------- Montgomery encode/decode -----------------

@pytest.mark.parametrize("size", MONT_SIZE)
@pytest.mark.parametrize("seed", MONT_SEEDS)
def test_montgomery_encode_decode_small(size, seed):
    random.seed(seed)
    m = 1_000_003  # small prime modulus
    R_value = pow(2, 32 * size, m)
    x = random.randrange(0, m)
    X = BigInteger.create_static(x, size)
    R = BigInteger.create_static(R_value, size)
    M = BigInteger.create_static(m, size)
    x_mon = bi_montgomery_encode(X, R, M)
    x_dec = bi_montgomery_decode(x_mon, R, M)
    # Decoding should give x mod m
    assert to_int(x_dec) % m == (x % m)


# ----------------- __call__ approximation (small exact range) -----------------

@pytest.mark.parametrize("size", CALL_SIZE)
@pytest.mark.parametrize("seed", CALL_SEEDS)
def test_call_float_approx_small_exact(size, seed):
    random.seed(seed)
    iv = random.randrange(0, 1 << 50)
    b = BigInteger.create(iv, size)  # safe range for dynamic create
    assert float(iv) == (b)()


# ----------------- Extended euclidean algorithm -----------------

@pytest.mark.parametrize("size", EEA_SIZES)
@pytest.mark.parametrize("seed", EEA_SEEDS)
def test_bi_extended_euclidean_small_range(size, seed):
    # Keep numbers small so that // and % used inside the EEA remain accurate
    random.seed(seed)
    a = random.randrange(1, 1 << 40)
    b = random.randrange(1, 1 << 40)

    A = BigInteger.create_static(a, size)
    B = BigInteger.create_static(b, size)

    g_bi, x_bi, y_bi = bi_extended_euclidean(A, B)

    g = to_int(g_bi)
    x = to_int(x_bi)
    y = to_int(y_bi)

    # Verify gcd
    expected_g = math.gcd(a, b)
    assert g == expected_g

    # Verify Bézout identity modulo 2^(32*size)
    M = 1 << (32 * size)
    lhs = (a * x + b * y) % M
    assert lhs == (g % M)

    # Additional divisibility checks
    assert a % g == 0
    assert b % g == 0