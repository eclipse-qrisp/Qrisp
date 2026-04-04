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
import pytest
import math
import jax
import jax.numpy as jnp
import numpy as np
from qrisp import *

CREATE_MEASURE_SEEDS = [0, 1, 2]
CREATE_MEASURE_SIZE = [2, 4, 8]

QC_INPLACE_SEEDS = [0, 1]
QC_INPLACE_SIZE = [1, 2]


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


# ----------------- Tests -----------------

@pytest.mark.parametrize("size", CREATE_MEASURE_SIZE)
@pytest.mark.parametrize("seed", CREATE_MEASURE_SEEDS)
def test_modulus_biginteger_create_measure(seed, size):
    random.seed(seed)
    limbs_N = random_limbs(size)
    limbs_V = random_limbs(size)
    N_int = limbs_to_int(limbs_N)
    V_int = limbs_to_int(limbs_V) % N_int

    @boolean_simulation
    def main():
        N = BigInteger.create_static(N_int, size)
        V = BigInteger.create_static(V_int, size)
        qm = QuantumModulus(N)
        qm[:] = V
        return measure(qm)
    assert to_int(main()) == V_int


def test_measure_to_big_integer():
    @boolean_simulation
    def main():
        qv = QuantumVariable(40)
        x(qv[0])
        x(qv[5])
        x(qv[33])
        return measure_to_big_integer(qv, 2)

    assert to_int(main()) == (1 << 0) + (1 << 5) + (1 << 33)


@pytest.mark.parametrize("size", QC_INPLACE_SIZE)
@pytest.mark.parametrize("seed", QC_INPLACE_SEEDS)
def test_modulus_biginteger_qc_inplace(seed, size):
    random.seed(seed)
    limbs_N = random_limbs(size)
    N_int = limbs_to_int(limbs_N)
    if N_int % 2 == 0:
        N_int += 1

    limbs_A = random_limbs(size)
    A_int = limbs_to_int(limbs_A) % N_int
    while True:
        limbs_B = random_limbs(size)
        B_int = limbs_to_int(limbs_B) % N_int
        if math.gcd(B_int, N_int) == 1:
            break

    @boolean_simulation
    def main():
        N = BigInteger.create_static(N_int, size+2)
        A = BigInteger.create_static(A_int, size+2)
        B = BigInteger.create_static(B_int, size+2)
        qm = QuantumModulus(N)
        qm[:] = A
        qm *= B
        return measure(qm)
    
    assert to_int(main()) == (A_int * B_int) % N_int


def test_modulus_create_measure():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        return measure(a)

    assert test() == 7

    @jaspify
    def test_m():
        a = QuantumModulus(13)
        a.m = 4
        a[:] = 7
        return measure(a)

    assert test_m() == 7


def test_modulus_qc_inplace_multiply():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        a *= 11
        return measure(a)

    assert test() == (7*11) % 13


def test_modulus_qc_multiply():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test_l():
        a = QuantumModulus(13)
        a[:] = 7
        b = a * 3
        return measure(a), measure(b)

    assert test_l() == (7, (7*3) % 13)

    @jaspify
    def test_r():
        a = QuantumModulus(13)
        a[:] = 7
        b = 3 * a
        return measure(a), measure(b)

    assert test_r() == (7, (7*3) % 13)


def test_modulus_qq_multiply():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a.m = 4
        a[:] = 7
        b = QuantumModulus(13)
        b.m = 4
        b[:] = 12
        c = a * b
        return measure(a), measure(b), measure(c)

    assert test() == (7, 12, (7*12) % 13)


def test_modulus_qq_multiply_standard_form():
    """Regression test: qq_montgomery_multiply_modulus must work when both
    inputs are in standard form (m=0).  Before the fix, the function used x.m
    directly as the reduction shift, which was 0 for standard-form inputs and
    produced wrong results."""
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        b = QuantumModulus(13)
        b[:] = 12
        c = a * b
        return measure(a), measure(b), measure(c)

    assert test() == (7, 12, (7 * 12) % 13)

    # Also test with a different modulus to cover more bit-width cases.
    @jaspify
    def test2():
        a = QuantumModulus(31)
        a[:] = 17
        b = QuantumModulus(31)
        b[:] = 23
        c = a * b
        return measure(a), measure(b), measure(c)

    assert test2() == (17, 23, (17 * 23) % 31)


# ----- Traced BigInteger modulus tests (PR 495 fixes) -----

# Small primes used across the traced-BigInteger tests.  They are small enough
# that BigInteger needs only 1 limb, keeping the tests fast.
_TRACED_BI_PRIMES = [7, 13, 31, 97]


@pytest.mark.parametrize("p", _TRACED_BI_PRIMES)
def test_qq_multiply_traced_biginteger_zeros(p):
    """qq multiply with traced BigInteger modulus — both inputs default (0)."""

    @jaspify
    def run(N):
        a = QuantumModulus(N)
        b = QuantumModulus(N)
        c = a * b
        return measure(c)

    result = to_int(run(BigInteger.create_static(p, 1)))
    assert result == 0, f"p={p}: expected 0, got {result}"


@pytest.mark.parametrize(
    "p,a_val,b_val",
    [
        (13, 5, 10),
        (13, 7, 12),
        (13, 1, 9),
        (31, 17, 23),
        (97, 50, 80),
    ],
)
def test_qq_multiply_traced_biginteger(p, a_val, b_val):
    """qq multiply with traced BigInteger modulus — non-trivial values."""

    @jaspify
    def run(N):
        a = QuantumModulus(N)
        b = QuantumModulus(N)
        a[:] = BigInteger.create(a_val, N.digits.shape[0])
        b[:] = BigInteger.create(b_val, N.digits.shape[0])
        c = a * b
        return measure(c)

    result = to_int(run(BigInteger.create_static(p, 1)))
    expected = (a_val * b_val) % p
    assert result == expected, f"p={p}: {a_val}*{b_val} mod {p} = {result}, expected {expected}"


def test_qq_multiply_traced_biginteger_identity():
    """Multiplying by 1 returns the other factor."""

    @jaspify
    def run(N):
        a = QuantumModulus(N)
        b = QuantumModulus(N)
        a[:] = BigInteger.create(1, N.digits.shape[0])
        b[:] = BigInteger.create(11, N.digits.shape[0])
        c = a * b
        return measure(c)

    result = to_int(run(BigInteger.create_static(13, 1)))
    assert result == 11


def test_qq_multiply_traced_biginteger_larger_prime():
    """Traced BigInteger with a larger-than-8-bit prime (1213 fits in 1 limb)."""

    @jaspify
    def run(N):
        a = QuantumModulus(N)
        b = QuantumModulus(N)
        a[:] = BigInteger.create(100, N.digits.shape[0])
        b[:] = BigInteger.create(200, N.digits.shape[0])
        c = a * b
        return measure(c)

    result = to_int(run(BigInteger.create_static(1213, 1)))
    assert result == (100 * 200) % 1213


def test_bi_pow2mod_basic():
    """Sanity-check bi_pow2mod against Python pow(2, e, N)."""
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import bi_pow2mod

    for p in [7, 13, 31, 97, 1213]:
        N_bi = BigInteger.create_static(p, 1)
        for e in [0, 1, 2, 5, 10, 20, 32]:
            result = to_int(bi_pow2mod(e, N_bi))
            expected = pow(2, e, p)
            assert result == expected, f"bi_pow2mod({e}, {p}) = {result}, expected {expected}"


def test_jdecoder_roundtrip_traced_biginteger():
    """Encode a value into Montgomery form, then verify jdecoder recovers it."""

    @jaspify
    def run(N):
        a = QuantumModulus(N)
        a[:] = BigInteger.create(9, N.digits.shape[0])
        # Multiply by 1 to trigger Montgomery encoding+decoding
        b = QuantumModulus(N)
        b[:] = BigInteger.create(1, N.digits.shape[0])
        c = a * b
        return measure(c)

    result = to_int(run(BigInteger.create_static(13, 1)))
    assert result == 9


# ----- _moduli_neq tests -----

def test_moduli_neq_static_equal():
    """Static BigInteger moduli with equal values should return False."""
    from qrisp.qtypes.quantum_modulus import _moduli_neq
    a = BigInteger.create_static(13, 1)
    b = BigInteger.create_static(13, 1)
    assert _moduli_neq(a, b) is False


def test_moduli_neq_static_different_value():
    """Static BigInteger moduli with different values should return True."""
    from qrisp.qtypes.quantum_modulus import _moduli_neq
    a = BigInteger.create_static(13, 1)
    b = BigInteger.create_static(17, 1)
    assert _moduli_neq(a, b) is True


def test_moduli_neq_static_different_shape():
    """Static BigInteger moduli with different limb counts should return True."""
    from qrisp.qtypes.quantum_modulus import _moduli_neq
    a = BigInteger.create_static(13, 1)
    b = BigInteger.create_static(13, 2)
    assert _moduli_neq(a, b) is True


def test_moduli_neq_mixed_types():
    """Comparing BigInteger with int should return True."""
    from qrisp.qtypes.quantum_modulus import _moduli_neq
    a = BigInteger.create_static(13, 1)
    assert _moduli_neq(a, 13) is True
    assert _moduli_neq(13, a) is True


def test_moduli_neq_plain_ints():
    """Plain int moduli should compare normally."""
    from qrisp.qtypes.quantum_modulus import _moduli_neq
    assert _moduli_neq(13, 13) is False
    assert _moduli_neq(13, 17) is True


def test_moduli_neq_raises_on_traced():
    """Traced BigInteger moduli must raise RuntimeError, not silently pass."""
    from qrisp.qtypes.quantum_modulus import _moduli_neq

    # Create traced digits by running inside jax.jit
    @jax.jit
    def inner(digits):
        a = BigInteger(digits)
        b = BigInteger(digits)
        try:
            _moduli_neq(a, b)
            return jnp.int32(0)  # should not reach here
        except RuntimeError:
            return jnp.int32(1)  # expected

    result = inner(jnp.array([13], dtype=jnp.uint32))
    assert int(result) == 1


def test_moduli_neq_not_called_during_tracing():
    """qq_multiply with traced BigInteger should not trigger _moduli_neq."""
    # If the guard is correct, this runs without RuntimeError
    @jaspify
    def run(N):
        a = QuantumModulus(N)
        b = QuantumModulus(N)
        c = a * b
        return measure(c)

    result = to_int(run(BigInteger.create_static(7, 1)))
    assert result == 0


# ----- BigInteger.coerce tests -----

def test_coerce_from_int():
    """coerce(int, size) should produce correct BigInteger."""
    bi = BigInteger.coerce(42, 2)
    assert bi.digits.shape[0] == 2
    assert to_int(bi) == 42


def test_coerce_from_biginteger_same_size():
    """coerce(BigInteger, same_size) should return the same object."""
    original = BigInteger.create_static(99, 2)
    result = BigInteger.coerce(original, 2)
    assert result is original


def test_coerce_from_biginteger_smaller_padded():
    """coerce(BigInteger, larger_size) should zero-pad."""
    small = BigInteger.create_static(42, 1)
    result = BigInteger.coerce(small, 3)
    assert result.digits.shape[0] == 3
    assert to_int(result) == 42


def test_coerce_from_biginteger_larger_raises():
    """coerce(BigInteger, smaller_size) should raise ValueError."""
    big = BigInteger.create_static(42, 4)
    with pytest.raises(ValueError, match="truncation"):
        BigInteger.coerce(big, 2)


# ----- _coerce_bigint_operand tests -----

def test_coerce_bigint_operand_from_int():
    """_coerce_bigint_operand should convert int to BigInteger with correct limbs."""
    from qrisp.qtypes.quantum_modulus import _coerce_bigint_operand
    modulus = BigInteger.create_static(13, 2)
    result = _coerce_bigint_operand(5, modulus)
    assert isinstance(result, BigInteger)
    assert result.digits.shape[0] == 2
    assert to_int(result) == 5


def test_coerce_bigint_operand_pads_smaller():
    """_coerce_bigint_operand should zero-pad a smaller BigInteger."""
    from qrisp.qtypes.quantum_modulus import _coerce_bigint_operand
    modulus = BigInteger.create_static(13, 3)
    value = BigInteger.create_static(7, 1)
    result = _coerce_bigint_operand(value, modulus)
    assert result.digits.shape[0] == 3
    assert to_int(result) == 7


def test_coerce_bigint_operand_same_size():
    """_coerce_bigint_operand with matching limbs returns same object."""
    from qrisp.qtypes.quantum_modulus import _coerce_bigint_operand
    modulus = BigInteger.create_static(13, 2)
    value = BigInteger.create_static(7, 2)
    result = _coerce_bigint_operand(value, modulus)
    assert result is value


def test_coerce_bigint_operand_larger_raises():
    """_coerce_bigint_operand should raise if BigInteger has too many limbs."""
    from qrisp.qtypes.quantum_modulus import _coerce_bigint_operand
    modulus = BigInteger.create_static(13, 1)
    value = BigInteger.create_static(7, 3)
    with pytest.raises(ValueError, match="truncation"):
        _coerce_bigint_operand(value, modulus)
