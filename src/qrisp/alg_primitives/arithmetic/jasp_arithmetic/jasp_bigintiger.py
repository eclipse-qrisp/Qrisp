"""
********************************************************************************
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
********************************************************************************
"""


import jax.numpy as jnp
import jax.lax as lax
import jax
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

BASE = 2**32
DTYPE = jnp.uint32
BASE_FL = float(BASE)


@register_pytree_node_class
@dataclass(frozen=True)
class BigInteger:
    """
    Fixed-width, little-endian base-2^32 big integer for JAX.

    This type represents non-negative integers using a fixed number of 32-bit
    limbs (dtype=uint32) in little-endian order (digits[0] is the least significant
    limb). It is compatible with JAX transformations (jit, vmap, pytrees).
    
    All arithmetic (addition, subtraction, multiplication, shifts, division,
    modulo, bitwise) is performed modulo 2^(32*n), where n = len(digits).
    Overflow and underflow beyond the most-significant limb are discarded.

    Notes
    -----
    - Width (number of limbs) is determined by `digits.shape[0]` and remains
      constant; operations do not change the number of limbs.
    - Operators assume both operands have the same number of limbs.

    Attributes
    ----------
    digits : jnp.ndarray
        Little-endian limbs (dtype=uint32) of length `n`.
    """
    digits: jnp.ndarray  # Little-endian base-2^32

    @jax.jit
    def __call__(self):
        """
        Return a float64 approximation of the integer value.

        Computes sum_i digits[i] * (2^32)^i as float64. Exact only for values
        that fit into float64; larger integers may lose precision.

        Returns
        -------
        jnp.float64
            Approximate numeric value (float64).
        """
        r = lax.fori_loop(
            0,
            self.digits.shape[0],
            lambda i, val: jnp.float64(
                self.digits[i]) * BASE_FL ** jnp.float64(i) + val,
            0.0,
        )
        return r

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        Returns
        -------
        tuple
            A pair `(children, aux_data)` where `children` is a tuple containing
            the digits array, and `aux_data` is `None`.
        """
        return (self.digits,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        PyTree unflatten for JAX.

        Parameters
        ----------
        aux_data : Any
            Auxiliary data (unused, expected `None`).
        children : tuple
            Tuple containing the digits array.

        Returns
        -------
        BigInteger
            Reconstructed instance.
        """
        return cls(*children)

    @staticmethod
    def create_static(n, size=2):
        """
        Create a BigInteger from Python using pure Python loops.

        This variant does not use JAX primitives and is suitable for static
        construction (e.g., outside `jit`). The result has exactly `size` limbs
        with wraparound modulo 2^(32*size).

        Parameters
        ----------
        n : int or float
            Non-negative number. Floats are truncated; very large floats
            (> 2**53) may lose precision before conversion.
        size : int, default=2
            Number of limbs (digits) to allocate.

        Returns
        -------
        BigInteger
            Fixed-width representation of `n` modulo 2^(32*size).
        """
        digits = []
        for i in range(size):
            digits.append(n % BASE)
            n //= BASE
        return BigInteger(jnp.array(digits, dtype=DTYPE))

    @staticmethod
    def create(n, size=2):
        """
        Create a BigInteger using JAX primitives.

        Constructs a fixed-width BigInteger with exactly `size` limbs, interpreting
        the input modulo 2^(32*size). JIT-friendly.

        Parameters
        ----------
        n : int or float or jnp.integer or jnp.floating
            Non-negative number. Floats are truncated; very large floats
            (> 2**53) may lose precision before conversion.
        size : int, default=2
            Number of limbs (digits) to allocate.

        Returns
        -------
        BigInteger
            Fixed-width representation of `n` modulo 2^(32*size).

        Notes
        -----
        When called with a Python literal outside `jit`, the value must fit into
        JAX's host integer range (typically up to 64 bits). For arbitrarily large
        Python integers, prefer `create_static`.
        """
        def body_fun(i, args):
            digits, num = args
            digits = digits.at[i].set(jnp.uint32(num % BASE))
            num //= BASE
            return digits, num
        digits, _ = lax.fori_loop(0, size, body_fun, (jnp.zeros(size, dtype=DTYPE), n))
        return BigInteger(digits)

    @staticmethod
    def create_dynamic(n, size=2):
        """
        Alias of `create`.

        Parameters
        ----------
        n : int or float or jnp.integer or jnp.floating
            Non-negative number.
        size : int, default=2
            Number of limbs (digits) to allocate.

        Returns
        -------
        BigInteger
            Fixed-width representation modulo 2^(32*size).

        See Also
        --------
        BigInteger.create : JAX-compatible constructor.
        """
        return BigInteger.create(n, size)

    @jax.jit
    def __add__(self, other: "BigInteger") -> "BigInteger":
        """
        Add two BigIntegers with wraparound modulo 2^(32*n).

        If `other` is a scalar, it is converted to the same width. The number
        of limbs `n` is taken from `self`.

        Parameters
        ----------
        other : BigInteger or int
            Addend.

        Returns
        -------
        BigInteger
            (self + other) mod 2^(32*n).
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        a, b = self.digits, other.digits
        result = jnp.zeros_like(a)
        carry = jnp.uint64(0)

        def add_step(i, state):
            carry, result = state
            s = jnp.uint64(a[i]) + jnp.uint64(b[i]) + carry
            digit = jnp.uint32(s & (BASE - 1))
            new_carry = jnp.uint64(s >> 32)
            result = result.at[i].set(jnp.uint32(digit))
            return new_carry, result

        carry, result = lax.fori_loop(0, a.shape[0], add_step, (carry, result))
        return BigInteger(result)

    @jax.jit
    def __sub__(self, other: "BigInteger") -> "BigInteger":
        """
        Subtract two BigIntegers with wraparound modulo 2^(32*n).

        If `other` is a scalar, it is converted to the same width. Computes
        (self - other) mod 2^(32*n) using borrow propagation.

        Parameters
        ----------
        other : BigInteger or int
            Subtrahend.

        Returns
        -------
        BigInteger
            (self - other) mod 2^(32*n).
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        a, b = self.digits, other.digits
        result = jnp.zeros_like(a)

        def add_step(i, state):
            carry, result = state
            s, new_carry = lax.cond(
                jnp.uint64(a[i]) >= jnp.uint64(b[i]) + carry,
                lambda: (jnp.uint32(a[i] - b[i] - carry), 0),
                lambda: (jnp.uint32(jnp.uint64(
                    a[i]) + jnp.uint64(BASE) - jnp.uint64(b[i]) - carry), 1),
            )
            result = result.at[i].set(jnp.uint32(s))
            return new_carry, result

        carry, result = lax.fori_loop(0, a.shape[0], add_step, (0, result))
        return BigInteger(result)

    @jax.jit
    def __sub_alt__(self, other: "BigInteger") -> "BigInteger":
        """
        Alternative subtraction using bitwise complement identity.

        Computes (self - other) as `~((~self) + other)` with wraparound
        modulo 2^(32*n).

        Parameters
        ----------
        other : BigInteger or int
            Subtrahend.

        Returns
        -------
        BigInteger
            (self - other) mod 2^(32*n).
        """
        return ~((~self) + other)

    @jax.jit
    def __mul__(self, other: "BigInteger") -> "BigInteger":
        """
        Multiply two BigIntegers with wraparound modulo 2^(32*n).

        Implements schoolbook multiplication and accumulates into `n` limbs,
        discarding overflow beyond the n-th limb (wraparound).

        Parameters
        ----------
        other : BigInteger or int
            Multiplier.

        Returns
        -------
        BigInteger
            (self * other) mod 2^(32*n).
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        a, b = self.digits, other.digits

        # result is uint32, but always promote to uint64 for arithmetic
        result = jnp.zeros(n, dtype=DTYPE)

        def outer_loop(i, result):
            carry0 = jnp.uint64(0)

            def inner_body(j, state):
                res, carry = state
                k = i + j
                tmp = (
                    jnp.uint64(res[k])
                    + jnp.uint64(a[i]) * jnp.uint64(b[j])
                    + carry
                )
                new_digit = jnp.uint32(tmp & jnp.uint64(0xFFFFFFFF))
                res = res.at[k].set(new_digit)
                carry = tmp >> jnp.uint64(32)
                return (res, carry)

            # j runs where k = i + j < n  -> j in [0, n - i)
            result, _ = lax.fori_loop(0, n - i, inner_body, (result, carry0))
            # Drop any remaining carry (mod base^n)
            return result

        result = lax.fori_loop(0, n, outer_loop, result)
        return BigInteger(result)

    @jax.jit
    def __pow__(self, other):
        """
        Integer exponentiation (square-and-multiply).

        Performs `self ** other` by binary exponentiation, modulo 2^(32*n).

        Parameters
        ----------
        other : int or jnp.integer
            Exponent (>= 0).

        Returns
        -------
        BigInteger
            self raised to power `other` modulo 2^(32*n).

        Notes
        -----
        For `other == 0`, returns 1 (the multiplicative identity) with the same width.
        """
        n = self.digits.shape[0]
        base = self
        exp = jnp.asarray(other, dtype=jnp.uint64)
        acc = BigInteger.create(1, n)

        def cond_fun(state):
            base, exp, acc = state
            return exp > 0

        def body_fun(state):
            base, exp, acc = state
            acc = lax.cond((exp & jnp.uint64(1)) == jnp.uint64(1),
                           lambda a: a * base,
                           lambda a: a,
                           acc)
            base = base * base
            exp = exp >> jnp.uint64(1)
            return base, exp, acc

        _, _, acc = lax.while_loop(cond_fun, body_fun, (base, exp, acc))
        return acc

    def __repr__(self):
        """
        String representation with limbs in little-endian order.

        Returns
        -------
        str
            String representation of digits (uint32 list).
        """
        return f"BigInteger(digits={self.digits.tolist()})"

    @jax.jit
    def __lt__(self, other: "BigInteger"):
        """
        Less-than comparison between two fixed-width BigIntegers.

        If `other` is a scalar, it is converted to the same width. Requires
        both operands to have the same number of limbs.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        jnp.bool_
            True if `self < other` (unsigned), else False.
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        m = other.digits.shape[0]
        assert n == m

        d0 = self.digits
        d1 = other.digits

        def body_fun(val):
            i, res_found = val
            i -= 1
            res_found = lax.cond(d0[i] < d1[i], lambda: 1, lambda: -5)
            res_found = lax.cond(d0[i] > d1[i], lambda: 0, lambda: res_found)
            return i, res_found

        def cond_fun(val):
            i, res_found = val
            return jnp.logical_and(res_found == -5, i > 0)

        _, res = lax.while_loop(cond_fun, body_fun, (n, -5))
        res = lax.cond(res == -5, lambda: 0, lambda: res)
        return (res != 0)

    @jax.jit
    def __eq__(self, other: "BigInteger"):
        """
        Equality comparison between two fixed-width BigIntegers.

        If `other` is a scalar, it is converted to the same width.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        jnp.bool_
            True if all limbs are equal, else False.
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        m = other.digits.shape[0]
        assert n == m
        d0 = self.digits
        d1 = other.digits
        return jnp.all(d0 == d1)

    @jax.jit
    def __ne__(self, other: "BigInteger"):
        """
        Inequality comparison between two fixed-width BigIntegers.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        jnp.bool_
            True if any limb differs, else False.
        """
        return jnp.logical_not(self == other)

    @jax.jit
    def __le__(self, other: "BigInteger"):
        """
        Less-or-equal comparison between two fixed-width BigIntegers.

        If `other` is a scalar, it is converted to the same width. Requires
        both operands to have the same number of limbs.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        jnp.bool_
            True if `self <= other` (unsigned), else False.
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        m = other.digits.shape[0]
        assert n == m

        d0 = self.digits
        d1 = other.digits

        def body_fun(val):
            i, res_found = val
            i -= 1
            res_found = lax.cond(d0[i] < d1[i], lambda: 1, lambda: -5)
            res_found = lax.cond(d0[i] > d1[i], lambda: 0, lambda: res_found)
            return i, res_found

        def cond_fun(val):
            i, res_found = val
            return jnp.logical_and(res_found == -5, i > 0)

        _, res = lax.while_loop(cond_fun, body_fun, (n, -5))
        res = lax.cond(res == -5, lambda: 1, lambda: res)
        return (res != 0)

    @jax.jit
    def __lshift__(self, shift):
        """
        Logical left shift by a non-negative number of bits.

        Shifts bits left by `shift` and fills with zeros, within fixed width.
        Bits shifted out of the most-significant end are discarded.

        Parameters
        ----------
        shift : int or jnp.integer
            Number of bits to shift (>= 0).

        Returns
        -------
        BigInteger
            (self << shift) mod 2^(32*n).
        """
        total_bits = self.digits.shape[0] * 32
        zeros = BigInteger(jnp.zeros_like(self.digits))

        def do_shift(_):
            def body_fun(i, x):
                return lax.cond(
                    self.get_bit(i) != 0,
                    lambda: x.flip_bit(i + shift),
                    lambda: x,
                )
            return lax.fori_loop(0, total_bits - shift, body_fun, zeros)

        return lax.cond(
            jnp.asarray(shift) >= total_bits,
            lambda _: zeros,
            do_shift,
            operand=None,
        )

    @jax.jit
    def __rshift__(self, shift):
        """
        Logical right shift by a non-negative number of bits.

        Shifts bits right by `shift` and fills with zeros, within fixed width.
        Bits shifted out of the least-significant end are discarded.

        Parameters
        ----------
        shift : int or jnp.integer
            Number of bits to shift (>= 0).

        Returns
        -------
        BigInteger
            (self >> shift) within fixed width.
        """
        total_bits = self.digits.shape[0] * 32
        zeros = BigInteger(jnp.zeros_like(self.digits))

        def do_shift(_):
            def body_fun(i, x):
                return lax.cond(
                    self.get_bit(i) != 0,
                    lambda: x.flip_bit(i - shift),
                    lambda: x,
                )
            return lax.fori_loop(shift, total_bits, body_fun, zeros)

        return lax.cond(
            jnp.asarray(shift) >= total_bits,
            lambda _: zeros,
            do_shift,
            operand=None,
        )

    @jax.jit
    def __and__(self, other: "BigInteger") -> "BigInteger":
        """
        Bitwise AND between two fixed-width BigIntegers.

        If `other` is a scalar, it is converted to the same width.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        BigInteger
            self & other (limb-wise).
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        return BigInteger(self.digits & other.digits)

    @jax.jit
    def __or__(self, other: "BigInteger") -> "BigInteger":
        """
        Bitwise OR between two fixed-width BigIntegers.

        If `other` is a scalar, it is converted to the same width.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        BigInteger
            self | other (limb-wise).
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        return BigInteger(self.digits | other.digits)

    @jax.jit
    def __xor__(self, other: "BigInteger") -> "BigInteger":
        """
        Bitwise XOR between two fixed-width BigIntegers.

        If `other` is a scalar, it is converted to the same width.

        Parameters
        ----------
        other : BigInteger or int
            Right-hand operand.

        Returns
        -------
        BigInteger
            self ^ other (limb-wise).
        """
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, n)
        return BigInteger(self.digits ^ other.digits)

    @jax.jit
    def __invert__(self) -> "BigInteger":
        """
        Bitwise NOT on a fixed-width BigInteger.

        Returns
        -------
        BigInteger
            Bitwise complement of `self`, limb-wise.
        """
        return BigInteger(~self.digits)

    @jax.jit
    def __mod__(self, other: "BigInteger"):
        """
        Modulo operation `self % other` (fixed-width).

        If `other` is a scalar, it is converted to the same width. Uses
        `remainder_division`.

        Parameters
        ----------
        other : BigInteger or int
            Modulus (must be non-zero).

        Returns
        -------
        BigInteger
            Remainder `r` with 0 <= r < other (when other != 0).
        """
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, self.digits.shape[0])
        r, q = self.remainder_division(other)
        return r

    @jax.jit
    def __floordiv__(self, other: "BigInteger"):
        """
        Floor division `self // other` (fixed-width).

        If `other` is a scalar, it is converted to the same width. Uses
        `remainder_division`.

        Parameters
        ----------
        other : BigInteger or int
            Divisor (must be non-zero).

        Returns
        -------
        BigInteger
            Quotient `q` such that self = other*q + r, 0 <= r < other.
        """
        if not isinstance(other, BigInteger):
            other = BigInteger.create(other, self.digits.shape[0])
        r, q = self.remainder_division(other)
        return q

    @jax.jit
    def get_bit(self, i: int):
        """
        Get the value of the i-th bit (0-based, LSB=bit 0).

        Parameters
        ----------
        i : int or jnp.integer
            Bit index (0 <= i < 32*n).

        Returns
        -------
        jnp.uint32
            Either 0 or 1.
        """
        pos = i // 32
        pos_in = i % 32
        return (self.digits[pos] >> pos_in) & 1

    @jax.jit
    def flip_bit(self, i: int):
        """
        Toggle the i-th bit (0-based, LSB=bit 0).

        Parameters
        ----------
        i : int or jnp.integer
            Bit index to flip (0 <= i < 32*n).

        Returns
        -------
        BigInteger
            Copy of `self` with the specified bit toggled.
        """
        pos = i // 32
        pos_in = i % 32
        ds = jnp.copy(self.digits)
        ds = ds.at[pos].set(jnp.uint32(ds[pos] ^ (1 << pos_in)))
        return BigInteger(ds)

    @jax.jit
    def bit_size(self):
        """
        Return the position of the most significant set bit plus one.

        Approximates the bit-length of the value: floor(log2(x)) + 1.
        Returns 0 for zero.

        Returns
        -------
        jnp.int64
            Bit-length of the value (0 for zero).
        """
        n = self.digits.shape[0]
        is_zero = jnp.all(self.digits == 0)

        def nonzero_len():
            # Find index ms of the most-significant non-zero limb (scan from top)
            def cond_fun(i):
                return jnp.logical_and(i > 0, self.digits[i] == jnp.uint32(0))

            def body_fun(i):
                return i - 1

            ms = lax.while_loop(cond_fun, body_fun, n - 1)
            limb = self.digits[ms]
            # bit length of a 32-bit limb
            limb_bits = (jnp.floor(jnp.log2(jnp.float64(limb))).astype(jnp.int64) + 1)
            return jnp.int64(32) * jnp.int64(ms) + limb_bits

        return lax.cond(is_zero, lambda: jnp.int64(0), nonzero_len)

    def remainder_division(self, other: "BigInteger"):
        """
        Exact division using Knuth long division (base 2^32).

        Computes quotient and remainder such that:
        `self = other * q + r`, with `0 <= r < other`.

        Parameters
        ----------
        other : BigInteger
            Divisor (must be non-zero), same width as `self`.

        Returns
        -------
        tuple of BigInteger
            `(r, q)` where `r` is the remainder and `q` is the quotient.

        Notes
        -----
        Uses a normalized Knuth division with limb base 2^32. Both `self`
        and `other` must have the same number of limbs.
        """
        n = self.digits.shape[0]
        m = other.digits.shape[0]
        assert n == m
        r_digits, q_digits = _remainder_division_knuth(self.digits, other.digits)
        return BigInteger(r_digits), BigInteger(q_digits)


@jax.jit
def _clz32(x):
    """
    Count leading zeros in a 32-bit word.

    Parameters
    ----------
    x : jnp.uint32
        Input 32-bit word.

    Returns
    -------
    jnp.int32
        Number of leading zeros in `x` (0..32).
    """
    # Count leading zeros in a 32-bit word (simple loop)
    def cond_fun(state):
        x, s = state
        return jnp.logical_and(s < jnp.int32(32), (x & jnp.uint32(0x80000000)) == jnp.uint32(0))

    def body_fun(state):
        x, s = state
        return jnp.uint32(x << jnp.uint32(1)), s + jnp.int32(1)
    _, s = lax.while_loop(cond_fun, body_fun, (x, jnp.int32(0)))
    return s


@jax.jit
def _ms_length(a: jnp.ndarray):
    """
    Effective limb length (highest non-zero index + 1).

    Parameters
    ----------
    a : jnp.ndarray
        Little-endian uint32 limb array.

    Returns
    -------
    jnp.int32
        Effective length in limbs (0 if all zero).
    """
    # Effective limb length: highest non-zero index + 1, or 0 if all zero.
    n = a.shape[0]
    all_zero = jnp.all(a == jnp.uint32(0))

    def find_ms():
        def cond_fun(i):
            return jnp.logical_and(i > 0, a[i] == jnp.uint32(0))

        def body_fun(i):
            return i - 1
        ms = lax.while_loop(cond_fun, body_fun, jnp.int32(n - 1))
        return ms + jnp.int32(1)
    return lax.cond(all_zero, lambda: jnp.int32(0), find_ms)


@jax.jit
def _shl_bits(arr: jnp.ndarray, s):
    """
    Shift-left by `s` bits across limbs (0 <= s < 32).

    Parameters
    ----------
    arr : jnp.ndarray
        Little-endian uint32 limb array to shift.
    s : jnp.int32
        Bit count (0..31).

    Returns
    -------
    tuple
        `(out, carry_out)` where `out` is the shifted array and `carry_out`
        is the carry from the most-significant limb (uint32).
    """
    # Shift-left by s bits (0<=s<32) across limbs (little endian).
    s_u = jnp.uint32(s)
    n = arr.shape[0]

    def no_shift():
        return jnp.array(arr, dtype=jnp.uint32), jnp.uint32(0)

    def do_shift():
        out = jnp.zeros_like(arr)
        carry = jnp.uint32(0)

        def body(i, state):
            carry, out = state
            ai = arr[i]
            low = jnp.uint32((jnp.uint64(ai) << jnp.uint64(s_u))
                             & jnp.uint64(0xFFFFFFFF))
            new_digit = jnp.uint32(low | carry)
            out = out.at[i].set(new_digit)
            # IMPORTANT: parentheses for precedence
            new_carry = jnp.uint32(jnp.uint64(ai) >> (jnp.uint64(32) - jnp.uint64(s_u)))
            return new_carry, out
        carry_out, out = lax.fori_loop(0, n, body, (carry, out))
        return out, carry_out
    return lax.cond(s == jnp.int32(0), no_shift, do_shift)


@jax.jit
def _shr_bits(arr: jnp.ndarray, s) -> jnp.ndarray:
    """
    Shift-right by `s` bits across limbs (0 <= s < 32).

    Parameters
    ----------
    arr : jnp.ndarray
        Little-endian uint32 limb array to shift.
    s : jnp.int32
        Bit count (0..31).

    Returns
    -------
    jnp.ndarray
        Shifted array (little-endian, uint32).
    """
    # Shift-right by s bits (0<=s<32) across limbs (little endian).
    s_u = jnp.uint32(s)
    n = arr.shape[0]

    def no_shift():
        return jnp.array(arr, dtype=jnp.uint32)

    def do_shift():
        out = jnp.zeros_like(arr)
        carry = jnp.uint32(0)

        def body(i, state):
            carry, out = state
            idx = n - 1 - i
            ai = arr[idx]
            high = jnp.uint32((jnp.uint64(carry) << (jnp.uint64(
                32) - jnp.uint64(s_u))) & jnp.uint64(0xFFFFFFFF))
            new_digit = jnp.uint32(
                (jnp.uint64(ai) >> jnp.uint64(s_u)) | jnp.uint64(high))
            out = out.at[idx].set(new_digit)
            new_carry = jnp.uint32(ai & (jnp.uint32(1) << s_u) - jnp.uint32(1))
            return new_carry, out
        _, out = lax.fori_loop(0, n, body, (carry, out))
        return out
    return lax.cond(s == jnp.int32(0), no_shift, do_shift)


@jax.jit
def _divmod_single_limb(u: jnp.ndarray, d):
    """
    Divide a multi-limb number by a single 32-bit limb.

    Parameters
    ----------
    u : jnp.ndarray
        Little-endian uint32 limb array (dividend).
    d : jnp.uint32
        Single-limb divisor (must be non-zero).

    Returns
    -------
    tuple
        `(q_digits, r_low)` where `q_digits` is the quotient array (uint32)
        and `r_low` is the remainder (uint32).
    """
    # Divide multi-limb u by single 32-bit d (d != 0). Returns (q_digits, r_low).
    n = u.shape[0]
    q = jnp.zeros_like(u)
    rem = jnp.uint64(0)

    def body(i, state):
        q, rem = state
        idx = n - 1 - i
        cur = (rem << jnp.uint64(32)) + jnp.uint64(u[idx])
        qi = jnp.uint32(cur // jnp.uint64(d))
        rem = jnp.uint64(cur % jnp.uint64(d))
        q = q.at[idx].set(qi)
        return q, rem
    q, rem = lax.fori_loop(0, n, body, (q, rem))
    return q, jnp.uint32(rem)


@jax.jit
def _remainder_division_knuth(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Knuth long division (base 2^32) for arrays of equal length.

    Parameters
    ----------
    u : jnp.ndarray
        Dividend limbs (little-endian uint32).
    v : jnp.ndarray
        Divisor limbs (little-endian uint32), not all zero.

    Returns
    -------
    tuple of jnp.ndarray
        `(r_digits, q_digits)` where both are little-endian uint32 arrays
        of the same length as `u`. The relation `u = v * q + r` holds with
        `0 <= r < v`.

    Notes
    -----
    Normalizes the divisor and uses a classic multi-precision division scheme.
    """
    # Knuth long division base 2^32. Inputs uint32 arrays (same length N).
    # Returns (r_digits, q_digits), both length N.
    N = u.shape[0]
    m = _ms_length(v)       # effective length of divisor
    n_eff = _ms_length(u)   # effective length of dividend

    # Divisor zero => return (0, 0)
    def div_by_zero():
        return jnp.zeros_like(u), jnp.zeros_like(u)

    def normal_path():
        # If u < v: q = 0, r = u
        def less_path():
            return u, jnp.zeros_like(u)

        def ge_path():
            # Single-limb fast path
            def single_limb_path():
                d = v[0]
                q, r_low = _divmod_single_limb(u, d)
                r = jnp.zeros_like(u).at[0].set(r_low)
                return r, q

            # Multi-limb path (m >= 2)
            def multi_limb_path():
                ms_idx = m - jnp.int32(1)
                v_msw = v[ms_idx]
                s = _clz32(v_msw)  # 0..31

                v_norm, _ = _shl_bits(v, s)           # length N
                u_norm_part, carry_u = _shl_bits(u, s)  # length N, carry
                u_norm = jnp.concatenate([u_norm_part, jnp.array(
                    [carry_u], dtype=jnp.uint32)], axis=0)  # N+1

                q = jnp.zeros_like(u)
                # how many quotient positions we fill
                j_count = jnp.maximum(n_eff - m, jnp.int32(0))

                BASE_MASK = jnp.uint64(0xFFFFFFFF)
                BASE64 = jnp.uint64(1) << jnp.uint64(32)

                def body(t, state):
                    u_norm, q = state
                    active = t <= j_count

                    def do_step(state):
                        u_norm, q = state
                        j = j_count - t          # 0..j_count
                        ujm = j + m              # index to top limb of current window

                        # Estimate qhat from top two limbs of u_norm and top limb of v_norm
                        u2 = jnp.uint64(u_norm[ujm])
                        u1 = jnp.uint64(u_norm[ujm - 1])
                        num = (u2 << jnp.uint64(32)) + u1
                        den = jnp.uint64(v_norm[m - 1])

                        qhat = jnp.minimum(num // den, jnp.uint64(0xFFFFFFFF))
                        rhat = num - qhat * den

                        # Refinement (at most twice)
                        vm2 = jnp.uint64(v_norm[m - 2])
                        u0 = jnp.uint64(u_norm[ujm - 2])

                        cond1 = jnp.logical_and(qhat == jnp.uint64(0xFFFFFFFF),
                                                qhat * vm2 > (rhat << jnp.uint64(32)) + u0)
                        qhat = jnp.where(cond1, qhat - jnp.uint64(1), qhat)
                        rhat = jnp.where(cond1, rhat + den, rhat)

                        cond2 = qhat * vm2 > (rhat << jnp.uint64(32)) + u0
                        qhat = jnp.where(cond2, qhat - jnp.uint64(1), qhat)
                        rhat = jnp.where(cond2, rhat + den, rhat)

                        # Subtract qhat * v_norm from u_norm[j ... j+m]
                        carry = jnp.uint64(0)

                        def sub_body(i, sstate):
                            u_norm, carry = sstate
                            vi = jnp.uint64(v_norm[i])
                            p = qhat * vi + carry
                            p_low = jnp.uint32(p & BASE_MASK)
                            p_high = p >> jnp.uint64(32)

                            idx = j + i
                            u_di = u_norm[idx]
                            borrow = u_di < p_low
                            new_u = jnp.uint32(
                                (jnp.uint64(u_di) + BASE64 - jnp.uint64(p_low)) & BASE_MASK)
                            u_norm = u_norm.at[idx].set(new_u)
                            carry = p_high + jnp.uint64(borrow)
                            return u_norm, carry

                        u_norm, carry = lax.fori_loop(0, m, sub_body, (u_norm, carry))

                        top_before = u_norm[ujm]
                        underflow = top_before < jnp.uint32(carry)
                        top_after = jnp.uint32(
                            (jnp.uint64(top_before) + BASE64 - jnp.uint64(carry)) & BASE_MASK)
                        u_norm = u_norm.at[ujm].set(top_after)

                        def fix_underflow(uq_state):
                            u_norm, q, qhat, j = uq_state
                            qhat = qhat - jnp.uint64(1)
                            carry2 = jnp.uint64(0)

                            def add_body(i, astate):
                                u_norm, carry2 = astate
                                idx = j + i
                                ssum = jnp.uint64(u_norm[idx]) + \
                                    jnp.uint64(v_norm[i]) + carry2
                                new_digit = jnp.uint32(ssum & BASE_MASK)
                                carry2 = ssum >> jnp.uint64(32)
                                u_norm = u_norm.at[idx].set(new_digit)
                                return u_norm, carry2
                            u_norm, carry2 = lax.fori_loop(
                                0, m, add_body, (u_norm, carry2))
                            u_norm = u_norm.at[ujm].set(jnp.uint32(
                                jnp.uint64(u_norm[ujm]) + jnp.uint64(1)))
                            return u_norm, q, qhat, j

                        u_norm, q, qhat, _ = lax.cond(
                            underflow,
                            fix_underflow,
                            lambda uq_state: uq_state,
                            operand=(u_norm, q, qhat, j),
                        )

                        q = q.at[j].set(jnp.uint32(qhat & BASE_MASK))
                        return u_norm, q

                    u_norm, q = lax.cond(
                        active, do_step, lambda state: state, operand=(u_norm, q))
                    return u_norm, q

                u_norm, q = lax.fori_loop(0, N, body, (u_norm, q))

                # Build r_full by copying the first m limbs (dynamic) of u_norm, then de-normalize
                r_norm_full = jnp.zeros_like(u)

                def copy_body(i, r_acc):
                    val = jnp.where(i < m, u_norm[i], jnp.uint32(0))
                    return r_acc.at[i].set(val)
                r_norm_full = lax.fori_loop(0, N, copy_body, r_norm_full)

                r = _shr_bits(r_norm_full, s)
                return r, q

            return lax.cond(m == jnp.int32(1), single_limb_path, multi_limb_path)

        return lax.cond(n_eff < m, less_path, ge_path)

    return lax.cond(m == jnp.int32(0), div_by_zero, normal_path)


@jax.jit
def bi_modinv(a: BigInteger, m: BigInteger) -> BigInteger:
    """
    Modular inverse using an Extended Euclidean Algorithm variant.

    Finds `t` such that `(a * t) % m == 1`, assuming `gcd(a, m) == 1`.
    Returns the non-negative representative in `[0, m)`.

    Parameters
    ----------
    a : BigInteger
        Value to invert (same width as `m`).
    m : BigInteger
        Modulus (must be > 0; often odd in Montgomery contexts).

    Returns
    -------
    BigInteger
        `t = a^{-1} mod m`.

    Raises
    ------
    AssertionError
        If inputs do not have the same width.

    Notes
    -----
    Uses `//` and `%` which rely on exact multi-precision division implemented
    in this module. Both `a` and `m` are treated as fixed-width unsigned values.
    """
    n = a.digits.shape[0]
    bi0 = BigInteger.create(0, n)
    bi1 = BigInteger.create(1, n)

    t, new_t = bi0, bi1
    r, new_r = m, a

    def cond(state):
        _, new_t, r, new_r = state
        return new_r != bi0

    def body(state):
        t, new_t, r, new_r = state
        quotient = r // new_r

        # Unsigned arithmetic with wraparound simulated modulo behavior
        t_updated = (t + m - (quotient * new_t) % m) % m
        r_updated = r - quotient * new_r

        return new_t, t_updated, new_r, r_updated

    final_t, _, final_r, _ = lax.while_loop(cond, body, (t, new_t, r, new_r))
    return final_t


@jax.jit
def bi_extended_euclidean(a, b):
    """
    Extended Euclidean Algorithm (fixed-width arithmetic).

    Computes `g, x, y` such that `a*x + b*y = g = gcd(a, b)`. Compatible with
    `jax.jit`. Inputs are expected to be `BigInteger` of the same width (or
    scalars that are promoted to that width).

    Parameters
    ----------
    a : BigInteger or int
        First number.
    b : BigInteger or int
        Second number.

    Returns
    -------
    tuple
        `(g, x, y)` where `g` is the gcd and `x, y` are Bézout coefficients
        (all `BigInteger`).

    Notes
    -----
    All operations occur modulo 2^(32*n), so the interpretation follows
    fixed-width arithmetic semantics.
    """
    n = a.digits.shape[0]
    if not isinstance(b, BigInteger):
        b = BigInteger.create(b, n)

    bi0 = BigInteger.create(0, n)
    bi1 = BigInteger.create(1, n)

    # State: (r, old_r, s, old_s, t, old_t)
    state = (b, a, bi0, bi1, bi1, bi0)

    def cond_fun(state):
        r, old_r, s, old_s, t, old_t = state
        return r != bi0

    def body_fun(state):
        r, old_r, s, old_s, t, old_t = state
        quotient = old_r // r
        return (
            old_r - quotient * r,
            r,
            old_s - quotient * s,
            s,
            old_t - quotient * t,
            t,
        )

    r, old_r, s, old_s, t, old_t = lax.while_loop(cond_fun, body_fun, state)
    # old_r is gcd, old_s and old_t are the Bézout coefficients
    return old_r, old_s, old_t


@jax.jit
def bi_montgomery_encode(x: BigInteger, R: BigInteger, modulus: BigInteger) -> BigInteger:
    """
    Montgomery encode: map x to (x * R) mod modulus, without intermediate wraparound.

    This routine widens all operands to 2n limbs (where n is the current width) to compute
    the product exactly, reduces the 2n-limb product modulo the widened modulus (also 2n
    limbs), and then shrinks the remainder back to n limbs. This avoids the truncation
    that would occur if the n-limb product were computed directly (since BigInteger
    arithmetic is modulo 2^(32*n)).

    Parameters
    ----------
    x : BigInteger
        Value to encode (width n limbs).
    R : BigInteger
        Montgomery radix (width n limbs), typically R = 2^m in the same limb system or
        equivalent precomputed value for the chosen modulus.
    modulus : BigInteger
        Modulus (width n limbs). In radix-2 Montgomery arithmetic, modulus is assumed odd.

    Returns
    -------
    BigInteger
        x in Montgomery form, i.e., (x * R) mod modulus, represented on n limbs.

    Notes
    -----
    - The 2n-limb widening ensures that (x * R) is computed without loss before the
      modulo reduction. The remainder is strictly less than modulus < 2^(32*n), so it
      fits in n limbs and can be safely truncated back.
    - All three inputs must share the same limb width n.
    """
    n = modulus.digits.shape[0]

    # Widen to 2n limbs
    pad = jnp.zeros(n, dtype=DTYPE)
    x2 = BigInteger(jnp.concatenate([x.digits, pad], axis=0))
    R2 = BigInteger(jnp.concatenate([R.digits, pad], axis=0))
    N2 = BigInteger(jnp.concatenate([modulus.digits, pad], axis=0))

    # Exact product at 2n limbs, reduce modulo widened modulus, then shrink to n limbs
    prod2 = x2 * R2
    rem2 = prod2 % N2
    return BigInteger(rem2.digits[:n])


@jax.jit
def bi_montgomery_decode(x_mon: BigInteger, R: BigInteger, modulus: BigInteger) -> BigInteger:
    """
    Montgomery decode: map x_mon to (x_mon * R^{-1}) mod modulus, without wraparound.

    This routine computes invR = R^{-1} mod modulus at width n, widens x_mon and invR
    to 2n limbs, performs the 2n-limb product and reduction modulo the widened modulus,
    and then shrinks the remainder back to n limbs. This avoids truncation during the
    multiply step in fixed-width arithmetic.

    Parameters
    ----------
    x_mon : BigInteger
        Value in Montgomery form (width n limbs).
    R : BigInteger
        Montgomery radix used for encoding (width n limbs).
    modulus : BigInteger
        Modulus (width n limbs). In radix-2 Montgomery arithmetic, modulus is assumed odd.

    Returns
    -------
    BigInteger
        Decoded value, i.e., (x_mon * R^{-1}) mod modulus, represented on n limbs.

    Notes
    -----
    - The 2n-limb widening ensures that (x_mon * invR) is computed exactly before the
      modulo reduction. The remainder is < modulus < 2^(32*n), so it fits within n limbs.
    - All three inputs must share the same limb width n.
    """
    n = modulus.digits.shape[0]

    # Compute R^{-1} mod modulus at n limbs
    invR = bi_modinv(R, modulus)

    # Widen operands and modulus to 2n limbs
    pad = jnp.zeros(n, dtype=DTYPE)
    x2 = BigInteger(jnp.concatenate([x_mon.digits, pad], axis=0))
    inv2 = BigInteger(jnp.concatenate([invR.digits,  pad], axis=0))
    N2 = BigInteger(jnp.concatenate([modulus.digits, pad], axis=0))

    # Exact product at 2n limbs, reduce modulo widened modulus, then shrink to n limbs
    prod2 = x2 * inv2
    rem2 = prod2 % N2
    return BigInteger(rem2.digits[:n])


@jax.jit
def _bi_all_ones(n_limbs: int) -> BigInteger:
    """
    Internal: BigInteger with all digits set to 0xFFFFFFFF (acts like +infinity bound).

    Parameters
    ----------
    n_limbs : int
        Number of 32-bit limbs.

    Returns
    -------
    BigInteger
        All-ones value, width n_limbs.
    """
    return BigInteger(jnp.full((n_limbs,), jnp.uint32(0xFFFFFFFF), dtype=DTYPE))


def bi_contfrac_best_approx(a: BigInteger,
                            b: BigInteger,
                            max_den: BigInteger | None = None,
                            max_iters: int | None = None) -> tuple[BigInteger, BigInteger]:
    """
    Best rational approximation p/q to a/b via continued fractions, with q <= max_den.

    This computes a convergent (or intermediate convergent) of the continued fraction
    expansion of a/b using the standard recurrence, stopping when either:
      - remainder becomes 0 (exact rational found), or
      - the next denominator would exceed `max_den`; in that case the best
        "intermediate" convergent within the bound is returned using
        t = floor((max_den - q0)/q1), p = t*p1 + p0, q = t*q1 + q0.

    The implementation uses a single JAX while_loop; it allocates no dynamic lists
    and is suitable for very large BigIntegers and JIT/vmap usage.

    Parameters
    ----------
    a : BigInteger
        Numerator (non-negative). Width must match `b`.
    b : BigInteger
        Denominator (non-zero, non-negative). Width must match `a`.
    max_den : BigInteger or None, optional
        Upper bound on the denominator q. If None, a built-in all-ones
        value of the same width is used (effectively unbounded).
    max_iters : int or None, optional
        Maximum number of continued-fraction steps (safety cap). If None,
        defaults to 2*bitlen + 4, where bitlen = 32 * n_limbs.

    Returns
    -------
    (BigInteger, BigInteger)
        A pair (p, q) with q > 0, p/q the best approximation to a/b
        under the bound q <= max_den. For exact rationals, returns the exact p/q.

    Raises
    ------
    AssertionError
        If widths of a and b differ, or if b == 0.

    Notes
    -----
    - All operands must share the same width (number of limbs).
    - Arithmetic is exact in the fixed-width ring; for valid inputs arising in Shor's
      post-processing (with q <= b and typical bounds <= b), no wrap-around occurs.
    """
    n_limbs = a.digits.shape[0]
    assert n_limbs == b.digits.shape[0], "BigInteger widths must match"
    bi0 = BigInteger.create(0, n_limbs)
    bi1 = BigInteger.create(1, n_limbs)
    assert b != bi0, "Denominator b must be non-zero"

    # Prepare bound and iteration cap
    if max_den is None:
        max_den = _bi_all_ones(n_limbs)
    else:
        assert max_den.digits.shape[0] == n_limbs, "max_den must match width of a,b"

    if max_iters is None:
        max_iters = int(2 * 32 * n_limbs + 4)

    @jax.jit
    def _loop(a0: BigInteger, b0: BigInteger,
              p0: BigInteger, p1: BigInteger,
              q0: BigInteger, q1: BigInteger,
              res_p: BigInteger, res_q: BigInteger,
              done: bool, i: int) -> tuple:
        """
        One step of CF with bound handling. Internal helper for while_loop.
        """
        # Compute quotient and remainder
        quot = a0 // b0
        rem = a0 % b0

        # Next convergent (pn/qn)
        pn = quot * p1 + p0
        qn = quot * q1 + q0

        # Conditions
        exact_end = (rem == bi0)
        # exceed if next denominator would be > max_den
        exceed = jnp.logical_not(qn <= max_den)

        # Intermediate convergent if exceed: t = floor((max_den - q0)/q1)
        # Guard q1==0 (only possible at very first step if max_den < 1)
        q1_is_zero = (q1 == bi0)
        t_num = (max_den - q0)
        t = t_num // jax.lax.cond(q1_is_zero, lambda: bi1, lambda: q1)
        p_bound = t * p1 + p0
        q_bound = t * q1 + q0

        # Choose final result when done at this step
        sel_p = jax.lax.select(exceed, p_bound, pn)
        sel_q = jax.lax.select(exceed, q_bound, qn)

        # Update result slots if we finish now
        res_p_next = jax.lax.select(jnp.logical_or(exceed, exact_end), sel_p, res_p)
        res_q_next = jax.lax.select(jnp.logical_or(exceed, exact_end), sel_q, res_q)
        done_next = jnp.logical_or(done, jnp.logical_or(exceed, exact_end))

        # Prepare state for next iteration (if not done)
        a_next = jax.lax.select(done_next, a0, b0)
        b_next = jax.lax.select(done_next, b0, rem)
        p0_next = jax.lax.select(done_next, p0, p1)
        p1_next = jax.lax.select(done_next, p1, pn)
        q0_next = jax.lax.select(done_next, q0, q1)
        q1_next = jax.lax.select(done_next, q1, qn)
        i_next = i + jnp.int32(1)
        return (a_next, b_next, p0_next, p1_next, q0_next, q1_next,
                res_p_next, res_q_next, done_next, i_next)

    def cond_fn(state):
        a0, b0, p0, p1, q0, q1, res_p, res_q, done, i = state
        under_cap = (i < jnp.int32(max_iters))
        return jnp.logical_and(jnp.logical_not(done), under_cap)

    # Initial CF state: p[-2]=0, p[-1]=1; q[-2]=1, q[-1]=0
    init_state = (a, b, bi0, bi1, bi1, bi0, bi0, bi1, jnp.bool_(False), jnp.int32(0))
    final_state = lax.while_loop(cond_fn, _loop, init_state)
    _, _, _, _, _, _, out_p, out_q, _, _ = final_state
    return out_p, out_q


def bi_shor_recover_denominator(a: BigInteger,
                                b: BigInteger,
                                N_bound: BigInteger | int,
                                max_iters: int | None = None) -> BigInteger:
    """
    Recover the candidate period denominator r from a/b for Shor's algorithm.

    This computes the best convergent p/q of a/b with q <= N_bound, where N_bound
    is typically the number to factor (or a small multiple), returning q.

    Parameters
    ----------
    a : BigInteger
        Numerator of measurement ratio (0 <= a < b).
    b : BigInteger
        Denominator (typically a power of two, b = 2^t).
    N_bound : BigInteger or int
        Upper bound for the denominator r (e.g., N).
    max_iters : int or None, optional
        Maximum CF steps (safety cap). See `bi_contfrac_best_approx`.

    Returns
    -------
    BigInteger
        The recovered denominator r (candidate period). Use with standard
        Shor post-checks (e.g., verify a close approximation and test that
        x^r ≡ 1 mod N, handle r even, etc.).

    Notes
    -----
    - If N_bound is an int, it is promoted to a BigInteger of the same width as a,b.
    - Returns the denominator of the CF-derived convergent; caller should perform
      the usual validity checks for Shor (closeness, non-trivial factor conditions, etc.).
    """
    n_limbs = a.digits.shape[0]
    if isinstance(N_bound, BigInteger):
        max_den = N_bound
        assert max_den.digits.shape[0] == n_limbs, "N_bound width must match a,b"
    else:
        max_den = BigInteger.create(int(N_bound), n_limbs)
    p, q = bi_contfrac_best_approx(a, b, max_den=max_den, max_iters=max_iters)
    return q


def bi_contfrac_convergents(a: BigInteger,
                            b: BigInteger,
                            max_terms: int | None = None):
    """
    Generator of convergents p/q for a/b (Python generator; not JAX-traced).

    Yields successive convergents (p_k, q_k) using the standard recurrence until
    remainder becomes 0 or `max_terms` is reached. Useful for debugging or
    non-jitted workflows. For high-performance in-jit use, prefer
    `bi_contfrac_best_approx`.

    Parameters
    ----------
    a : BigInteger
        Numerator.
    b : BigInteger
        Denominator (non-zero).
    max_terms : int or None, optional
        Maximum number of convergents to yield.

    Yields
    ------
    (BigInteger, BigInteger)
        Convergent pairs (p_k, q_k), k = 0, 1, 2, ...
    """
    n_limbs = a.digits.shape[0]
    assert n_limbs == b.digits.shape[0], "BigInteger widths must match"
    bi0 = BigInteger.create(0, n_limbs)
    bi1 = BigInteger.create(1, n_limbs)
    assert b != bi0, "Denominator b must be non-zero"

    a0, b0 = a, b
    p0, p1 = bi0, bi1
    q0, q1 = bi1, bi0
    k = 0
    while True:
        q = a0 // b0
        r = a0 % b0
        pn = q * p1 + p0
        qn = q * q1 + q0
        yield pn, qn
        if r == bi0:
            break
        a0, b0 = b0, r
        p0, p1 = p1, pn
        q0, q1 = q1, qn
        k += 1
        if max_terms is not None and k >= max_terms:
            break
