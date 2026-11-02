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

"""Tests QuantumModulus and BigInteger arithmetic, including Montgomery multiplication, under Jasp tracing."""

import math
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    BigInteger,
    QuantumArray,
    QuantumBool,
    QuantumFloat,
    QuantumModulus,
    QuantumVariable,
    boolean_simulation,
    conjugate,
    control,
    count_ops,
    custom_inversion,
    h,
    jaspify,
    measure,
    measure_to_big_integer,
    num_qubits,
    qache,
    x,
)
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import bi_pow2mod
from qrisp.qtypes.quantum_modulus import _coerce_bigint_operand, _moduli_neq

CREATE_MEASURE_SEEDS = [0, 1, 2]
CREATE_MEASURE_SIZE = [2, 4, 8]

QC_INPLACE_SEEDS = [0, 1]
QC_INPLACE_SIZE = [1, 2]


# ----------------- Helpers -----------------


def mask_for_size(size: int) -> int:
    return (1 << (32 * size)) - 1


def _digits_array(maybe_bi):
    """Return the digits array whether input is a BigInteger or a raw JAX/Numpy array."""
    if isinstance(maybe_bi, (jnp.ndarray, np.ndarray)):
        return maybe_bi
    if hasattr(maybe_bi, "digits"):
        return maybe_bi.digits
    raise TypeError(f"Unexpected type for digits extraction: {type(maybe_bi)}")


def to_int(maybe_bi) -> int:
    """Convert a BigInteger (or its digits array) to a Python int exactly."""
    ds = np.asarray(_digits_array(maybe_bi), dtype=np.uint64).tolist()
    acc = 0
    for i, d in enumerate(ds):
        acc |= int(d) << (32 * i)
    return acc


def limbs_to_int(limbs) -> int:
    """Convert a list of uint32 limbs (little-endian) to a Python int."""
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
        N = BigInteger.create_static(N_int, size + 2)
        A = BigInteger.create_static(A_int, size + 2)
        B = BigInteger.create_static(B_int, size + 2)
        qm = QuantumModulus(N)
        qm[:] = A
        qm *= B
        return measure(qm)

    assert to_int(main()) == (A_int * B_int) % N_int


def test_modulus_create_measure():
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
    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        a *= 11
        return measure(a)

    assert test() == (7 * 11) % 13


def test_modulus_qc_multiply():
    @jaspify
    def test_l():
        a = QuantumModulus(13)
        a[:] = 7
        b = a * 3
        return measure(a), measure(b)

    assert test_l() == (7, (7 * 3) % 13)

    @jaspify
    def test_r():
        a = QuantumModulus(13)
        a[:] = 7
        b = 3 * a
        return measure(a), measure(b)

    assert test_r() == (7, (7 * 3) % 13)


def test_modulus_numpy_scalar_add_with_nonzero_shift():
    @jaspify
    def test_add():
        a = QuantumModulus(13)
        a.m = 3
        a[:] = 9
        b = a + np.int64(10)
        return measure(a), measure(b)

    assert test_add() == (9, (9 + 10) % 13)


def test_modulus_scalar_sub_with_nonzero_shift():
    @jaspify
    def test_sub():
        a = QuantumModulus(13)
        a.m = 3
        a[:] = 9
        b = a - 10
        return measure(a), measure(b)

    assert test_sub() == (9, (9 - 10) % 13)


def test_modulus_qq_multiply():
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

    assert test() == (7, 12, (7 * 12) % 13)


def test_modulus_qq_multiply_standard_form():
    """Regression test: qq_montgomery_multiply_modulus must work when both
    inputs are in standard form (m=0).  Before the fix, the function used x.m
    directly as the reduction shift, which was 0 for standard-form inputs and
    produced wrong results.
    """

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
    """Qq multiply with traced BigInteger modulus — both inputs default (0)."""

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
    """Qq multiply with traced BigInteger modulus — non-trivial values."""

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


def test_measure_traced_scalar_modulus_static_shift():
    """Regression: traced scalar modulus with non-zero static shift must decode."""

    @jaspify
    def run():
        qf = QuantumFloat(4)
        qf[:] = 7
        modulus = jnp.int32(measure(qf))

        qm = QuantumModulus(modulus)
        qm[:] = 3
        qm.m = 1
        return measure(qm)

    assert int(run()) == (3 * pow(2, -1, 7)) % 7


def test_measure_traced_scalar_modulus_traced_shift():
    """Regression: traced scalar modulus with traced shift must decode."""

    @jaspify
    def run():
        qf = QuantumFloat(4)
        qf[:] = 7
        modulus = jnp.int32(measure(qf))

        qf_shift = QuantumFloat(2)
        qf_shift[:] = 2
        shift = jnp.int32(measure(qf_shift))

        qm = QuantumModulus(modulus)
        qm[:] = 3
        qm.m = shift
        return measure(qm)

    assert int(run()) == (3 * pow(2, -2, 7)) % 7


# ----- _moduli_neq tests -----


def test_moduli_neq_static_equal():
    """Static BigInteger moduli with equal values should return False."""
    a = BigInteger.create_static(13, 1)
    b = BigInteger.create_static(13, 1)
    assert _moduli_neq(a, b) is False


def test_moduli_neq_static_different_value():
    """Static BigInteger moduli with different values should return True."""
    a = BigInteger.create_static(13, 1)
    b = BigInteger.create_static(17, 1)
    assert _moduli_neq(a, b) is True


def test_moduli_neq_static_different_shape():
    """Static BigInteger moduli with different limb counts should return True."""
    a = BigInteger.create_static(13, 1)
    b = BigInteger.create_static(13, 2)
    assert _moduli_neq(a, b) is True


def test_moduli_neq_mixed_types():
    """Comparing BigInteger(13) with int 13 should return False (they are equal)."""
    a = BigInteger.create_static(13, 1)
    assert _moduli_neq(a, 13) is False
    assert _moduli_neq(13, a) is False


def test_moduli_neq_mixed_types_different():
    """Comparing BigInteger(13) with int 17 should return True."""
    a = BigInteger.create_static(13, 1)
    assert _moduli_neq(a, 17) is True
    assert _moduli_neq(17, a) is True


def test_moduli_neq_plain_ints():
    """Plain int moduli should compare normally."""
    assert _moduli_neq(13, 13) is False
    assert _moduli_neq(13, 17) is True


def test_moduli_neq_raises_on_traced():
    """Traced BigInteger moduli must raise RuntimeError, not silently pass."""

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
    modulus = BigInteger.create_static(13, 2)
    result = _coerce_bigint_operand(5, modulus)
    assert isinstance(result, BigInteger)
    assert result.digits.shape[0] == 2
    assert to_int(result) == 5


def test_coerce_bigint_operand_pads_smaller():
    """_coerce_bigint_operand should zero-pad a smaller BigInteger."""
    modulus = BigInteger.create_static(13, 3)
    value = BigInteger.create_static(7, 1)
    result = _coerce_bigint_operand(value, modulus)
    assert result.digits.shape[0] == 3
    assert to_int(result) == 7


def test_coerce_bigint_operand_same_size():
    """_coerce_bigint_operand with matching limbs returns same object."""
    modulus = BigInteger.create_static(13, 2)
    value = BigInteger.create_static(7, 2)
    result = _coerce_bigint_operand(value, modulus)
    assert result is value


def test_coerce_bigint_operand_larger_raises():
    """_coerce_bigint_operand should raise if BigInteger has too many limbs."""
    modulus = BigInteger.create_static(13, 1)
    value = BigInteger.create_static(7, 3)
    with pytest.raises(ValueError, match="truncation"):
        _coerce_bigint_operand(value, modulus)


# ----- comparison_wrapper tests (Montgomery shift handling) -----


def test_comparison_same_nonzero_shift():
    """Two QuantumModuli with the same non-zero Montgomery shift can be compared."""
    a = QuantumModulus(13)
    b = QuantumModulus(13)
    a[:] = 5
    b[:] = 5
    a.m = 3
    b.m = 3
    # Same shift → comparison should work, not raise
    res = a == b
    assert measure(res)[0]


def test_comparison_same_nonzero_shift_neq():
    """Two QuantumModuli with the same non-zero shift but different values."""
    a = QuantumModulus(13)
    b = QuantumModulus(13)
    a[:] = 5
    b[:] = 7
    a.m = 2
    b.m = 2
    res = a != b
    assert measure(res)[0]


def test_comparison_different_shifts_raises():
    """Two QuantumModuli with different non-zero shifts must raise."""
    a = QuantumModulus(13)
    b = QuantumModulus(13)
    a[:] = 5
    b[:] = 5
    a.m = 2
    b.m = 3
    with pytest.raises(Exception, match="differing Montgomery shifts"):
        a == b


def test_comparison_nonzero_shift_vs_non_modulus_raises():
    """Comparing QuantumModulus(m!=0) with a QuantumFloat must raise."""
    a = QuantumModulus(13)
    a[:] = 5
    a.m = 2
    b = QuantumFloat(4)
    b[:] = 5
    with pytest.raises(Exception, match="non-zero Montgomery shift"):
        a == b


# ----- Traced-modulus tracer-leak regression tests -----
# Regression for: repeated, independent @jaspify calls with a QuantumModulus
# built from a traced modulus argument used to raise jax.errors.UnexpectedTracerError
# on the 2nd call, because `modulus` wasn't threaded through the pytree as a
# proper traced leaf (it got smuggled into static aux_data instead).


def test_traced_modulus_qache_repeated_calls():
    """A @qache'd subroutine touching a traced-modulus QuantumModulus must not
    leak a tracer across independent @jaspify calls."""

    @qache
    def op(f, root):
        f[0] += root

    @jaspify
    def solve(q, root):
        f = QuantumArray(QuantumModulus(q), shape=(1,))
        op(f, root)
        return measure(f)

    for _ in range(3):
        assert list(solve(13, 5)) == [5.0]


def test_traced_modulus_custom_inversion_repeated_calls():
    """A @custom_inversion function used inside conjugate() on a traced-modulus
    QuantumModulus must not leak a tracer across independent @jaspify calls."""

    @custom_inversion
    def op(f, root, inv=False):
        if inv:
            f[0] -= root
        else:
            f[0] += root

    @jaspify
    def solve(q, root):
        f = QuantumArray(QuantumModulus(q), shape=(1,))
        with conjugate(op)(f, root):
            pass
        return measure(f)

    for _ in range(3):
        assert list(solve(13, 5)) == [0.0]


def test_traced_modulus_jnp_array_repeated_calls():
    """A modulus passed as a concrete jnp.array (not a function argument) must
    still yield consistent, correct results across repeated calls."""

    @jaspify
    def main():
        q = jnp.array(13)
        a = QuantumModulus(q)
        b = QuantumModulus(q)
        a[:] = 3
        b[:] = 5
        res = a + b
        return measure(res)

    for _ in range(3):
        assert int(main()) == 8


def test_static_modulus_hardcoded_literal_repeated_calls():
    """A hardcoded, static modulus inside @jaspify must still work across repeated calls."""

    @jaspify
    def solve():
        f = QuantumModulus(13)
        f[:] = 3
        f += 5
        return measure(f)

    for _ in range(3):
        assert int(solve()) == 8


# ----- Montgomery shift handling under Jasp (issue #925) -----

SHIFT_N = 13

SHIFT_PAIRS = [(0, 0), (2, 2), (2, 0), (0, -3), (-5, 1)]

SHIFT_OPS = {
    "add": lambda x, y: (x + y) % SHIFT_N,
    "sub": lambda x, y: (x - y) % SHIFT_N,
    "iadd": lambda x, y: (x + y) % SHIFT_N,
    "isub": lambda x, y: (x - y) % SHIFT_N,
}


def _apply_shift_op(op, a, b):
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "iadd":
        a += b
    else:
        a -= b
    return a


def test_shift_qache_multiplication_no_tracer_leak():
    """The shift of a product computed inside qache must not leak a tracer."""

    @qache
    def mul(a, b):
        return a * b

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        b = QuantumModulus(SHIFT_N)
        a[:] = 3
        b[:] = 5
        return measure(mul(a, b))

    assert int(main()) == 2


def test_shift_part_of_qache_key():
    """A qached function called with differing shifts must not reuse a stale trace."""

    @qache
    def measure_inside(a):
        return measure(a)

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a[:] = 3
        b = QuantumModulus(SHIFT_N)
        b[:] = 3
        b.m = 2
        return measure(a), measure(b), measure_inside(a), measure_inside(b)

    assert tuple(int(x) for x in main()) == (3, 4, 3, 4)


def test_shift_inplace_addition_after_multiplication():
    """In-place addition of operands with differing shifts (product has a negative shift)."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        b = QuantumModulus(SHIFT_N)
        a[:] = 3
        b[:] = 5
        c = a * b
        c += a
        return measure(c)

    assert int(main()) == 5


def test_shift_negative_classical_addition():
    """Encoding a classical operand into a QuantumModulus with negative shift."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        b = QuantumModulus(SHIFT_N)
        a[:] = 3
        b[:] = 5
        c = a * b
        c += 1
        return measure(c)

    assert int(main()) == 3


@pytest.mark.parametrize("am, bm", SHIFT_PAIRS)
@pytest.mark.parametrize("op", list(SHIFT_OPS))
def test_shift_aware_arithmetic_static_shift(op, am, bm):
    """Addition/subtraction with mixed shifts known at trace time."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a.m = am
        a[:] = 9
        b = QuantumModulus(SHIFT_N)
        b.m = bm
        b[:] = 7
        return measure(_apply_shift_op(op, a, b))

    assert int(main()) == SHIFT_OPS[op](9, 7)


@pytest.mark.parametrize("am, bm", SHIFT_PAIRS)
@pytest.mark.parametrize("op", list(SHIFT_OPS))
def test_shift_aware_arithmetic_traced_shift(op, am, bm):
    """Addition/subtraction with mixed shifts that are traced (crossing a qache boundary)."""

    @qache
    def kernel(a, b):
        if op in ("iadd", "isub"):
            _apply_shift_op(op, a, b)
            return
        return _apply_shift_op(op, a, b)

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a.m = am
        a[:] = 9
        b = QuantumModulus(SHIFT_N)
        b.m = bm
        b[:] = 7
        res = kernel(a, b)
        return measure(a if res is None else res)

    assert int(main()) == SHIFT_OPS[op](9, 7)


@pytest.mark.parametrize("am, bm", SHIFT_PAIRS)
@pytest.mark.parametrize("op", list(SHIFT_OPS))
def test_shift_aware_arithmetic_non_jasp(op, am, bm):
    """Addition/subtraction with mixed shifts outside of Jasp."""
    a = QuantumModulus(SHIFT_N)
    a.m = am
    a[:] = 9
    b = QuantumModulus(SHIFT_N)
    b.m = bm
    b[:] = 7
    res = _apply_shift_op(op, a, b)
    assert list(res.get_measurement().keys()) == [SHIFT_OPS[op](9, 7)]


def test_shift_aware_addition_quantum_float_operand():
    """A QuantumFloat operand is treated as having shift 0."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a.m = 2
        a[:] = 9
        b = QuantumFloat(4)
        b[:] = 7
        a += b
        return measure(a)

    assert int(main()) == 3


@pytest.mark.parametrize("qm_m", [0, -1, 3])
def test_shift_aware_addition_quantum_float_exponent(qm_m):
    """A QuantumFloat operand with exponent e (value = raw * 2^e) acts as shift -e."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a.m = qm_m
        a[:] = 9
        b = QuantumFloat(3, exponent=1)
        b[:] = 6
        a += b
        return measure(a)

    assert int(main()) == (9 + 6) % SHIFT_N


def test_shift_aware_addition_quantum_float_operand_qache():
    """A QuantumFloat operand crossing a qache boundary (traced exponent)."""

    @qache
    def kernel(a, b):
        a += b

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a.m = 2
        a[:] = 9
        b = QuantumFloat(3, exponent=1)
        b[:] = 6
        kernel(a, b)
        return measure(a)

    assert int(main()) == (9 + 6) % SHIFT_N


@pytest.mark.parametrize("qm_m", [0, 2])
@pytest.mark.parametrize("exponent", [0, 1])
def test_shift_aware_addition_quantum_float_operand_non_jasp(qm_m, exponent):
    """A QuantumFloat operand outside of Jasp."""
    a = QuantumModulus(SHIFT_N)
    a.m = qm_m
    a[:] = 9
    b = QuantumFloat(3, exponent=exponent)
    b[:] = 6
    a += b
    assert list(a.get_measurement().keys()) == [(9 + 6) % SHIFT_N]


@pytest.mark.parametrize("op", list(SHIFT_OPS))
def test_shift_aware_arithmetic_signed_quantum_float_raises(op):
    """Signed QuantumFloat operands are rejected, since the sign bit has negative weight."""
    a = QuantumModulus(SHIFT_N)
    a[:] = 9
    b = QuantumFloat(3, signed=True)
    b[:] = -1
    with pytest.raises(TypeError, match="signed QuantumFloat"):
        _apply_shift_op(op, a, b)


def test_shift_aware_arithmetic_signed_quantum_float_raises_jasp():
    """Signed QuantumFloat operands are rejected in Jasp mode (signed is static)."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a[:] = 9
        b = QuantumFloat(3, signed=True)
        b[:] = -1
        a += b
        return measure(a)

    with pytest.raises(TypeError, match="signed QuantumFloat"):
        main()


@pytest.mark.parametrize("am", [0, 2, -3])
def test_shift_aware_rsub(am):
    """Classical minus QuantumModulus with a non-zero shift."""

    @jaspify
    def main():
        a = QuantumModulus(SHIFT_N)
        a.m = am
        a[:] = 9
        return measure(5 - a)

    assert int(main()) == (5 - 9) % SHIFT_N

    a = QuantumModulus(SHIFT_N)
    a.m = am
    a[:] = 9
    res = 5 - a
    assert list(res.get_measurement().keys()) == [(5 - 9) % SHIFT_N]


@pytest.mark.parametrize("flag_on", [True, False])
def test_shift_aware_addition_controlled(flag_on):
    """Controlled ``b += a`` with differing shifts adds a only if the control is set.

    Regression test: in static mode, ``custom_control`` replaced the per-bit
    ``ctrl=a[i]`` of ``mod_adder`` with the enclosing control qubit, so every
    bit's constant was added regardless of a.
    """
    N = 5

    @jaspify
    def main(a_val):
        a = QuantumModulus(N)
        b = QuantumModulus(N)
        b.m = 1
        a[:] = a_val
        b[:] = 2
        flag = QuantumBool()
        if flag_on:
            x(flag)
        with control(flag):
            b += a
        return measure(b)

    for a_val in range(N):
        expected = (2 + a_val) % N if flag_on else 2
        assert int(main(a_val)) == expected, a_val

        a = QuantumModulus(N)
        b = QuantumModulus(N)
        b.m = 1
        a[:] = a_val
        b[:] = 2
        flag = QuantumBool()
        if flag_on:
            x(flag)
        with control(flag):
            b += a
        assert list(b.get_measurement().keys()) == [expected], a_val


@pytest.mark.parametrize("N", [12, np.int64(12)])
def test_shift_aware_addition_even_modulus_raises(N):
    """Differing shifts need 2 to be invertible modulo N."""
    a = QuantumModulus(N)
    a[:] = 1
    b = QuantumModulus(N)
    b[:] = 1
    b.m = 1
    with pytest.raises(ValueError, match="odd modulus"):
        a += b


def test_shift_aware_addition_even_biginteger_modulus_raises():
    """A BigInteger modulus with concrete digits is checked for parity as well."""
    N = BigInteger.create_static(12, 1)

    @jaspify
    def main():
        a = QuantumModulus(N)
        a[:] = BigInteger.create(1, 1)
        b = QuantumModulus(N)
        b[:] = BigInteger.create(1, 1)
        b.m = 1
        a += b
        return measure(a)

    with pytest.raises(ValueError, match="odd modulus"):
        main()


def test_shift_aware_addition_odd_biginteger_modulus_closure():
    """A concrete odd BigInteger modulus passes the parity check."""
    N = BigInteger.create_static(SHIFT_N, 1)

    @jaspify
    def main():
        a = QuantumModulus(N)
        a.m = 2
        a[:] = BigInteger.create(9, 1)
        b = QuantumModulus(N)
        b[:] = BigInteger.create(7, 1)
        a += b
        return measure(a)

    assert to_int(main()) == (9 + 7) % SHIFT_N


# 2^31 - 1 fits in one 32-bit limb, but products of two residues do not.
LARGE_BI_PRIME = 2**31 - 1


@pytest.mark.parametrize("am, bm", [(0, 3), (5, -2)])
def test_shift_aware_addition_large_biginteger_modulus(am, bm):
    """Mixed-shift addition with a BigInteger modulus whose residue products overflow a limb."""
    a_val, b_val = 123456789, 2000000000

    @jaspify
    def main(N):
        a = QuantumModulus(N)
        a.m = am
        a[:] = BigInteger.create(a_val, N.digits.shape[0])
        b = QuantumModulus(N)
        b.m = bm
        b[:] = BigInteger.create(b_val, N.digits.shape[0])
        a += b
        return measure(a)

    result = to_int(main(BigInteger.create_static(LARGE_BI_PRIME, 1)))
    assert result == (a_val + b_val) % LARGE_BI_PRIME


# ----------------- Montgomery shift of the multiplication by a classical factor -----------------

# The multiplication by a classical factor X uses a Montgomery reduction with a
# shift m that must depend on the size n of the quantum register: the reduction
# sums one partial product (reduced modulo N) per qubit, so m = ceil(log2(n))
# is enough. The shift used to be computed from the value of X instead, which
# was close to n for large factors and could overflow for 32-bit moduli.
# The factors below cover both regimes: small (2, 3) and close to N.


@pytest.mark.parametrize("N", [13, 17])
def test_inplace_multiplication_by_classical_factor_all_inputs(N):
    """``qm *= X`` computes X * y mod N for every input y, with and without a control qubit."""

    @boolean_simulation
    def multiply(factor, y, c):
        qm = QuantumModulus(N)
        qm[:] = y
        ctrl = QuantumBool()
        ctrl[:] = c
        with control(ctrl[0]):
            qm *= factor
        return measure(qm)

    for factor in (2, 3, N - 2, N - 1):
        for y in range(N):
            for c in (0, 1):
                assert int(multiply(factor, y, c)) == factor**c * y % N, (factor, y, c)


@pytest.mark.parametrize("N", [13, 17])
def test_multiplication_by_classical_factor_all_inputs(N):
    """``qm * X`` computes X * y mod N for every input y and leaves ``qm`` unchanged."""

    @boolean_simulation
    def multiply(factor, y):
        qm = QuantumModulus(N)
        qm[:] = y
        res = qm * factor
        return measure(qm), measure(res)

    for factor in (2, 3, N - 2, N - 1):
        for y in range(N):
            assert tuple(int(v) for v in multiply(factor, y)) == (y, factor * y % N), (factor, y)


def test_inplace_multiplication_with_32_bit_modulus():
    """``qm *= X`` computes X * y mod N for a 32-bit integer modulus.

    Regression test: the shift was computed from X * (N - 1), which overflowed
    int64 for a 32-bit factor and modulus. The shift became 0 and the result was wrong.
    """
    N = 3221225473

    @boolean_simulation
    def multiply(factor, y):
        qm = QuantumModulus(N)
        qm[:] = y
        qm *= factor
        return measure(qm)

    for factor, y in [(N - 2, 12345), (N - 2, N - 1), (3, 2**31 + 5), (123456789, 987654321)]:
        assert int(multiply(factor, y)) == factor * y % N, (factor, y)


@pytest.mark.parametrize(
    "N, factor, peak",
    [(40961, 3, 58), (40961, 40959, 58), (3221225473, 3, 108), (3221225473, 3221225471, 108)],
    ids=["16-bit N, factor 3", "16-bit N, factor N - 2", "32-bit N, factor 3", "32-bit N, factor N - 2"],
)
def test_multiplication_shift_depends_on_register_size(N, factor, peak):
    """A controlled ``qm *= X`` needs the same number of qubits for every factor X.

    It allocates the control qubit, the n-qubit register, a temporary copy of it,
    m + 1 auxiliary qubits and the n + m ancillas of the widest Gidney adder:
    3n + 2m + 2 qubits with m = ceil(log2(n)), that is 58 for n = 16 and 108 for n = 32.

    Regression test: with the shift computed from the value of the factor, the
    qubit count changed with the factor (54 and 82 for n = 16).
    """

    def main():
        qm = QuantumModulus(N)
        ctrl = QuantumBool()
        h(ctrl)
        with control(ctrl):
            qm *= factor

    assert num_qubits(meas_behavior="1")(main)()["peak_allocations"] == peak


@pytest.mark.parametrize("factor", [3, np.int64(3), jnp.int64(3)], ids=["int", "np.int64", "jnp.int64"])
def test_multiplication_accepts_the_same_factors_as_inplace_multiplication(factor):
    """``qm * X`` accepts the same classical factors as ``qm *= X``, including JAX integers.

    Regression test: ``qm * jnp.int64(3)`` raised a TypeError, while ``qm *= jnp.int64(3)`` worked.
    """

    @jaspify
    def multiply():
        qm = QuantumModulus(13)
        qm[:] = 5
        return measure(qm * factor)

    @jaspify
    def multiply_inplace():
        qm = QuantumModulus(13)
        qm[:] = 5
        qm *= factor
        return measure(qm)

    assert int(multiply()) == 5 * 3 % 13
    assert int(multiply_inplace()) == 5 * 3 % 13


def test_montgomery_shift_roundtrip_with_40_bit_modulus():
    """A QuantumModulus with a non-zero Montgomery shift measures the value it was set to, for a 40-bit modulus.

    Regression test: encoding and decoding multiplied in int64, which overflowed
    for moduli above 2**31.5.
    """
    N = 1000000000039

    @boolean_simulation
    def roundtrip(value):
        qm = QuantumModulus(N)
        qm.m = 3
        qm[:] = value
        return measure(qm)

    for value in (1, 12345, 987654321123, N - 1):
        assert int(roundtrip(value)) == value, value


@pytest.mark.parametrize("N", [13, 17])
def test_inpl_mult_with_inverse_matches_inplace_multiplication(N):
    """``qm.inpl_mult(X, inverse=X^-1)`` computes X * y mod N for every input y, with and without control."""

    @boolean_simulation
    def multiply(factor, inverse, y, c):
        qm = QuantumModulus(N)
        qm[:] = y
        ctrl = QuantumBool()
        ctrl[:] = c
        with control(ctrl[0]):
            qm.inpl_mult(factor, inverse=inverse)
        return measure(qm)

    for factor in (2, 3, N - 2, N - 1):
        for y in range(N):
            for c in (0, 1):
                assert int(multiply(factor, pow(factor, -1, N), y, c)) == factor**c * y % N, (factor, y, c)


def test_inpl_mult_with_inverse_biginteger_modulus():
    """``qm.inpl_mult`` with a BigInteger modulus and a precomputed BigInteger inverse."""
    rng = random.Random(40)
    N = rng.randrange(2**39 + 1, 2**40, 2)
    factor, y = rng.randrange(2, N), rng.randrange(1, N)
    while math.gcd(factor, N) != 1:
        factor = rng.randrange(2, N)

    @boolean_simulation
    def multiply(y):
        qm = QuantumModulus(BigInteger.create_static(N, 2))
        qm[:] = y
        qm.inpl_mult(BigInteger.create_static(factor, 2), inverse=BigInteger.create_static(pow(factor, -1, N), 2))
        return measure(qm)

    assert multiply(y)() == factor * y % N


def test_inpl_mult_with_inverse_gives_the_same_circuit():
    """Passing the inverse only skips its classical computation: the gates and qubits are those of ``qm *= X``."""
    rng = random.Random(64)
    N = rng.randrange(2**63 + 1, 2**64, 2)
    factor = rng.randrange(2, N)
    while math.gcd(factor, N) != 1:
        factor = rng.randrange(2, N)

    def program(use_inverse):
        def main():
            qm = QuantumModulus(BigInteger.create_static(N, 4))
            qm[:] = 1
            qb = QuantumBool()
            h(qb)
            with control(qb):
                if use_inverse:
                    qm.inpl_mult(
                        BigInteger.create_static(factor, 4), inverse=BigInteger.create_static(pow(factor, -1, N), 4)
                    )
                else:
                    qm *= BigInteger.create_static(factor, 4)

        return main

    for metric in (count_ops, num_qubits):
        assert metric(meas_behavior="1")(program(True))() == metric(meas_behavior="1")(program(False))()


def test_inpl_mult_rejects_quantum_operands():
    """``inpl_mult`` only multiplies by classical factors."""

    def main(operand_is_factor):
        qm = QuantumModulus(13)
        other = QuantumFloat(4)
        if operand_is_factor:
            qm.inpl_mult(other)
        else:
            qm.inpl_mult(3, inverse=other)

    for operand_is_factor in (True, False):
        with pytest.raises(TypeError):
            main(operand_is_factor)


@pytest.mark.parametrize("bits", [36, 40, 61, 63])
def test_inplace_multiplication_with_large_integer_modulus(bits):
    """``qm *= X`` computes X * y mod N for an integer modulus of up to 61 bits.

    Regression test: the partial products X * 2**j mod N were computed as
    (X << j) % N in int64, which overflowed for moduli above about 33 bits.
    """
    rng = random.Random(bits)
    N = rng.randrange(2 ** (bits - 1) + 1, 2**bits, 2)
    factor, y = rng.randrange(2, N), rng.randrange(1, N)

    @boolean_simulation
    def multiply(factor, y):
        qm = QuantumModulus(N)
        qm[:] = y
        qm *= factor
        return measure(qm)

    assert int(multiply(factor, y)) == factor * y % N


@pytest.mark.parametrize("operation", ["imul", "mul", "inpl_mult"])
def test_multiplication_rejects_float_factors(operation):
    """Multiplying by a float raises a TypeError instead of truncating the factor."""

    def main():
        qm = QuantumModulus(13)
        qm[:] = 5
        factor = jnp.float32(2.7)
        if operation == "imul":
            qm *= factor
        elif operation == "mul":
            qm * factor
        else:
            qm.inpl_mult(factor)

    with pytest.raises(TypeError):
        count_ops(meas_behavior="1")(main)()


@pytest.mark.parametrize("operation", ["imul", "mul", "inpl_mult"])
def test_multiplication_rejects_non_scalar_factors(operation):
    """Multiplying by a JAX array that is not a scalar raises a TypeError."""

    def main():
        qm = QuantumModulus(13)
        qm[:] = 5
        factor = jnp.array([3])
        if operation == "imul":
            qm *= factor
        elif operation == "mul":
            qm * factor
        else:
            qm.inpl_mult(factor)

    with pytest.raises(TypeError):
        count_ops(meas_behavior="1")(main)()


def test_inpl_mult_with_smaller_width_inverse():
    """``qm.inpl_mult`` accepts a BigInteger inverse with fewer limbs than the modulus.

    Regression test: the inverse was widened from its own width,
    and the uncomputation raised a ValueError for the mixed widths.
    """

    @boolean_simulation
    def multiply(y):
        qm = QuantumModulus(BigInteger.create_static(13, 2))
        qm[:] = y
        qm.inpl_mult(3, inverse=BigInteger.create_static(9, 1))
        return measure(qm)

    for y in (1, 5, 12):
        assert multiply(y)() == 3 * y % 13, y


@pytest.mark.parametrize(
    "make_factor",
    [
        lambda f: f,
        lambda f: BigInteger.create_static(f, 2),
        lambda f: BigInteger.create_static(f, 4),
    ],
    ids=["int", "2-limb BigInteger", "4-limb BigInteger"],
)
def test_multiplication_by_factor_wider_than_biginteger_modulus(make_factor):
    """A factor with more limbs than the BigInteger modulus is reduced modulo it, not truncated.

    Regression test: the factor was cut to the limbs of the
    modulus (a Python int) or rejected (a wider BigInteger).
    """
    N, factor = 4294967291, 2**40 + 3

    @boolean_simulation
    def multiply(y):
        qm = QuantumModulus(BigInteger.create_static(N, 1))
        qm[:] = y
        qm *= make_factor(factor)
        return measure(qm), measure(qm * make_factor(factor))

    for y in (1, 12345, N - 1):
        inplace, product = multiply(y)
        assert inplace() == factor * y % N, y
        assert product() == factor**2 * y % N, y


def test_multiplication_by_traced_factor_wider_than_biginteger_modulus():
    """A traced int64 factor with more bits than a one-limb BigInteger modulus is reduced modulo it.

    Regression test (review of #918): the factor was cut to the 32 bits of the modulus.
    """
    N, factor = 4294967291, 2**40 + 3

    @boolean_simulation
    def multiply(factor, y):
        qm = QuantumModulus(BigInteger.create_static(N, 1))
        qm[:] = y
        qm *= factor
        return measure(qm)

    for y in (1, 12345, N - 1):
        assert multiply(factor, y)() == factor * y % N, y


def test_multiplication_by_python_int_wider_than_64_bits():
    """A Python int factor of more than 64 bits is reduced modulo an integer modulus.

    Regression test: under tracing, only its low 64 bits were kept.
    """
    N, factor = 13, 2**70 + 3

    @boolean_simulation
    def multiply(y):
        qm = QuantumModulus(N)
        qm[:] = y
        qm *= factor
        return measure(qm)

    for y in range(1, N):
        assert int(multiply(y)) == factor * y % N, y


def test_inpl_mult_biginteger_inverse_with_integer_modulus():
    """``qm.inpl_mult`` with an integer modulus accepts a BigInteger inverse.

    Regression test (review of #918): the inverse stayed a one-limb BigInteger,
    so the uncomputation truncated the modulus 2**40 + 39 to that limb.
    """
    N = 2**40 + 39

    @boolean_simulation
    def multiply(y):
        qm = QuantumModulus(N)
        qm[:] = y
        qm.inpl_mult(pow(3, -1, N), inverse=BigInteger.create_static(3, 1))
        return measure(qm)

    for y in (1, 12345, N - 1):
        assert int(multiply(y)) == pow(3, -1, N) * y % N, y


@pytest.mark.parametrize("operation", ["imul", "inpl_mult"])
def test_inplace_multiplication_by_unsigned_factor(operation):
    """An unsigned JAX factor of 2**63 or more is reduced modulo N in unsigned arithmetic.

    Regression test (review of #918): the inverse computation cast the factor
    2**64 - 1 (2 modulo 13) to int64, saw -1 and returned 1 instead of 7.
    """
    N, factor = 13, 2**64 - 1

    @boolean_simulation
    def multiply(y):
        qm = QuantumModulus(N)
        qm[:] = y
        if operation == "imul":
            qm *= jnp.uint64(factor)
        else:
            qm.inpl_mult(jnp.uint64(factor))
        return measure(qm)

    for y in range(1, N):
        assert int(multiply(y)) == factor * y % N, y
