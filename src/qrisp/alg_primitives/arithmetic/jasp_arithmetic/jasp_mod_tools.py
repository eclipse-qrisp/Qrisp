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

"""
Utilities for modular arithmetic that work with both Python ints and BigInteger.

All functions are JAX-friendly under tracing and avoid Python-side int() casts
and non-traceable conditionals where they could occur during tracing.

Notes
-----
- For BigInteger inputs, operations stay in BigInteger space.
- For Python ints, standard integer arithmetic is used.
- best_montgomery_shift implements the paper’s choice using N when available,
  and falls back to ceil(log2(n)) when N is not provided.
"""

from typing import Union
from jax import jit
import jax.numpy as jnp
import jax.lax as lax
from qrisp.jasp import check_for_tracing_mode

from .jasp_bigintiger import (
    BigInteger,
    bi_modinv,
    bi_montgomery_encode,
    bi_montgomery_decode,
)


@jit
def pow2_mod_N(m, N):
    """
    Compute ``2 ** m % N`` for traced scalar inputs.

    Uses square-and-multiply inside ``lax.while_loop`` so the computation
    remains valid under JAX tracing.

    Parameters
    ----------
    m : int or jnp scalar
        Non-negative exponent, typically traced under JAX.
    N : int or jnp scalar
        Modulus.

    Returns
    -------
    jnp scalar
        ``2 ** m % N`` with the modulus dtype.

    """
    m = jnp.asarray(m, dtype=jnp.int64)
    N = jnp.asarray(N)
    base = jnp.asarray(2, dtype=N.dtype)

    # State tuple: (result, base, exponent, modulus)
    init_state = (jnp.asarray(1, dtype=N.dtype), base % N, m, N)

    def cond_fun(state):
        _, _, exp, _ = state
        return exp > 0

    def body_fun(state):
        res, b, exp, mod = state

        # If exp is odd, update result: res = (res * b) % mod
        # jax.lax.select acts like an if-else statement without breaking traceability
        is_odd = exp % 2 == 1
        res = lax.select(is_odd, (res * b) % mod, res)

        # Square the base and halve the exponent
        b = (b * b) % mod
        exp = exp // 2

        return (res, b, exp, mod)

    result, _, _, _ = lax.while_loop(cond_fun, body_fun, init_state)
    return result


def bi_pow2mod(exp, mod_bi):
    """
    Compute ``2 ** exp % mod_bi`` as a BigInteger.

    Uses square-and-multiply in double-width BigInteger space so that
    intermediate products cannot overflow.  Both *exp* and *mod_bi* may
    contain traced JAX values.

    Parameters
    ----------
    exp : int or jnp scalar
        Non-negative exponent (may be a JAX tracer).
    mod_bi : BigInteger
        Modulus (same fixed width as the desired result).

    Returns
    -------
    BigInteger
        ``2 ** exp % mod_bi`` with the same limb count as *mod_bi*.
    """
    k = mod_bi.digits.shape[0]
    # Double width to avoid wraparound in intermediate products
    mod_w = mod_bi.get_larger()
    init_result = BigInteger.create(1, 2 * k)
    init_base = BigInteger.create(2, 2 * k)

    def cond_fn(state):
        _, _, e = state
        return e > 0

    def body_fn(state):
        result, base, e = state
        odd = (e & jnp.int64(1)) == jnp.int64(1)
        new_result = (result * base) % mod_w
        # Select without lax.cond to keep the pytree flat
        result = BigInteger(jnp.where(odd, new_result.digits, result.digits))
        base = (base * base) % mod_w
        e = e >> jnp.int64(1)
        return result, base, e

    result, _, _ = lax.while_loop(
        cond_fn, body_fn, (init_result, init_base, jnp.int64(exp))
    )
    # Truncate back — result < mod_bi < 2^(32k)
    return BigInteger(result.digits[:k])


def pow2mod(exp, modulus: Union[int, BigInteger]):
    """
    Compute ``2 ** exp % modulus`` across static ints, traced ints, and BigInteger.

    Dispatches to ``bi_pow2mod`` for BigInteger moduli, ``pow2_mod_N`` for
    traced scalar moduli, and Python ``pow`` in static scalar mode.

    Parameters
    ----------
    exp : int or jnp scalar
        Non-negative exponent.
    modulus : int or BigInteger
        Modulus.

    Returns
    -------
    int, jnp scalar, or BigInteger
        ``2 ** exp % modulus`` in the representation matching ``modulus``.
    """
    if isinstance(modulus, BigInteger):
        return bi_pow2mod(exp, modulus)
    if check_for_tracing_mode():
        return pow2_mod_N(exp, modulus)
    return pow(2, int(exp), int(modulus))


def montgomery_encoder(
    x: Union[int, BigInteger], R: Union[int, BigInteger], N: Union[int, BigInteger]
):
    """
    Montgomery-encode x as x*R mod N.

    Parameters
    ----------
    x : int or BigInteger
        Value to encode.
    R : int or BigInteger
        Montgomery radix (e.g., R = 2^m).
    N : int or BigInteger
        Modulus (assumed odd for radix 2).

    Returns
    -------
    int or BigInteger
        x in Montgomery form.
    """
    if (
        isinstance(x, BigInteger)
        or isinstance(R, BigInteger)
        or isinstance(N, BigInteger)
    ):
        xb = x if isinstance(x, BigInteger) else BigInteger.create(x, N.digits.shape[0])
        Rb = R if isinstance(R, BigInteger) else BigInteger.create(R, N.digits.shape[0])
        Nb = (
            N if isinstance(N, BigInteger) else BigInteger.create(N, Rb.digits.shape[0])
        )
        return bi_montgomery_encode(xb, Rb, Nb)
    return ((x % N) * (R % N)) % N


def new_montgomery_decoder(
    y: Union[int, BigInteger], m: Union[int, BigInteger], N: Union[int, BigInteger]
):
    """
    Montgomery-decode y using the shift exponent m instead of an explicit radix.

    The helper supports plain Python ints, traced JAX scalars, and BigInteger
    moduli by routing the power-of-two and modular inverse computations
    through the matching helper implementations.

    Parameters
    ----------
    y : int or BigInteger
        Montgomery-encoded value.
    m : int or jnp scalar
        Montgomery shift. Positive shifts decode with ``(2**m)^{-1} mod N``;
        non-positive shifts decode with ``2**abs(m) mod N``.
    N : int or BigInteger
        Modulus.

    Returns
    -------
    int or BigInteger
        Decoded value in standard representation.
    """
    if check_for_tracing_mode():
        m = jnp.int64(m)
        abs_m = jnp.abs(m)
        power = pow2mod(abs_m, N)
        factor = lax.cond(
            m > 0,
            lambda p: modinv(p, N),
            lambda p: p,
            power,
        )
    else:
        abs_m = abs(m)
        power = pow2mod(abs_m, N)
        factor = modinv(power, N) if m > 0 else power

    return montgomery_encoder(y, factor, N)


def montgomery_decoder(
    y: Union[int, BigInteger], R: Union[int, BigInteger], N: Union[int, BigInteger]
):
    """
    Montgomery-decode y as y*R^{-1} mod N.

    Parameters
    ----------
    y : int or BigInteger
        Montgomery-encoded value.
    R : int or BigInteger
        Montgomery radix used during encoding.
    N : int or BigInteger
        Modulus.

    Returns
    -------
    int or BigInteger
        Decoded value in standard representation.
    """
    if (
        isinstance(y, BigInteger)
        or isinstance(R, BigInteger)
        or isinstance(N, BigInteger)
    ):
        yb = y if isinstance(y, BigInteger) else BigInteger.create(y, N.digits.shape[0])
        Rb = R if isinstance(R, BigInteger) else BigInteger.create(R, N.digits.shape[0])
        Nb = (
            N if isinstance(N, BigInteger) else BigInteger.create(N, Rb.digits.shape[0])
        )
        return bi_montgomery_decode(yb, Rb, Nb)
    # Handle fractional R (from negative Montgomery shifts)
    if isinstance(R, float) and 0 < R < 1:
        R = modinv(int(R**-1), N)
    R1 = modinv(R, N)
    return ((y % N) * (R1 % N)) % N


def egcd(a: int, b: int):
    """
    Extended Euclidean Algorithm (ints, JAX-friendly).

    Computes (g, x, y) such that a*x + b*y = g = gcd(a, b).

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    tuple
        (g, x, y) with gcd and Bézout coefficients (JAX scalars under tracing).
    """

    def cond_fun(state):
        r, _, _, _ = state
        return jnp.logical_not(r == 0)

    def body_fun(state):
        r, old_r, s, old_s, t, old_t = state
        q = old_r // r
        return (old_r - q * r, r, old_s - q * s, s, old_t - q * t, t)

    state = (b, a, 0, 1, 1, 0)
    r, old_r, s, old_s, t, old_t = lax.while_loop(cond_fun, body_fun, state)
    return old_r, old_s, old_t


def modinv(a: Union[int, BigInteger], m: Union[int, BigInteger]):
    """
    Modular inverse t = a^{-1} mod m.

    BigInteger inputs are handled via bi_modinv; ints use a JAX-compatible EEA
    under tracing and a simple loop otherwise.

    Parameters
    ----------
    a : int or BigInteger
        Value to invert (coprime to m).
    m : int or BigInteger
        Modulus.

    Returns
    -------
    int or BigInteger
        Modular inverse in [0, m).
    """
    if isinstance(a, BigInteger) or isinstance(m, BigInteger):
        a_bi = (
            a if isinstance(a, BigInteger) else BigInteger.create(a, m.digits.shape[0])
        )
        m_bi = (
            m if isinstance(m, BigInteger) else BigInteger.create(m, a.digits.shape[0])
        )
        return bi_modinv(a_bi, m_bi)

    if check_for_tracing_mode():
        dtype = jnp.asarray(m).dtype
        zero = jnp.asarray(0, dtype=dtype)
        one = jnp.asarray(1, dtype=dtype)
        a = jnp.asarray(a, dtype=dtype)
        m = jnp.asarray(m, dtype=dtype)

        def cf(val):
            t, nt, r, nr = val
            return nr != 0

        def bf(val):
            t, nt, r, nr = val
            q = r // nr
            return (nt, t - q * nt, nr, r - q * nr)

        t, nt, r, nr = lax.while_loop(cf, bf, (zero, one, m, a))
        return jnp.where(t < zero, t + m, t)
    else:
        t, nt, r, nr = 0, 1, m, a
        while nr != 0:
            q = r // nr
            t, nt = nt, t - q * nt
            r, nr = nr, r - q * nr
        return t + m if t < 0 else t


def smallest_power_of_two(n: Union[int, BigInteger]):
    """
    ceil(log2(n)) computed in a JAX-safe way.

    Parameters
    ----------
    n : int or BigInteger
        Positive integer (returns 0 for n <= 1).

    Returns
    -------
    int or jnp.int64
        ceil(log2(n)) with 0 for n <= 1.
    """
    if isinstance(n, BigInteger):
        # bit_size already yields ceil(log2(n)) with 0 for n==0
        return n.bit_size()

    if hasattr(n, "bit_length"):
        return (n - 1).bit_length() if n > 1 else 0

    if check_for_tracing_mode():
        nj = jnp.asarray(n)
        # Avoid log2(0); define result 0 for n<=1
        return jnp.where(
            nj <= 1, jnp.int64(0), jnp.ceil(jnp.log2(nj)).astype(jnp.int64)
        )

    raise TypeError(
        "smallest_power_of_two expects int, BigInteger, or traced JAX scalar, "
        f"got {type(n).__name__}"
    )


def best_montgomery_shift(n: Union[int, BigInteger], N: Union[int, BigInteger] = None):
    """
    Minimal auxiliary exponent m for Montgomery reduction.

    If N is provided (classical integer, possibly JAX-traced), use the tighter bound
    derived from t < n·(N-1):
        m = ceil( log2( ceil( n·(N-1)/N ) ) )
    Otherwise, fall back to the paper's simpler choice:
        m = ceil(log2(n))

    Parameters
    ----------
    n : int or BigInteger
        Number of partial products (e.g., bit-width of the quantum multiplicand).
    N : int or BigInteger, optional
        Odd modulus. If None, falls back to ceil(log2(n)).

    Returns
    -------
    int or jnp.int64
        Exponent m such that R = 2^m.
    """
    # Fallback (no modulus or unsupported BigInteger modulus): ceil(log2 n)
    if (N is None) or isinstance(N, BigInteger):
        return smallest_power_of_two(n)

    # Use integer-safe ceil-division with JAX or Python ints
    if check_for_tracing_mode():
        nj = jnp.asarray(n)
        Nj = jnp.asarray(N)
        num = nj * (Nj - 1)
        den = Nj
        # ceil(num/den) = (num + den - 1) // den
        ceil_div = (num + den - 1) // den
        return smallest_power_of_two(ceil_div)
    else:
        num = n * (N - 1)
        ceil_div = (num + N - 1) // N
        return smallest_power_of_two(ceil_div)
