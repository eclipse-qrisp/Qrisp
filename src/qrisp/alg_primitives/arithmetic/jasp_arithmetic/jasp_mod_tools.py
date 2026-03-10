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

import jax.numpy as jnp
import jax.lax as lax
from qrisp import check_for_tracing_mode

from .jasp_bigintiger import (
    BigInteger,
    bi_modinv,
    bi_montgomery_encode,
    bi_montgomery_decode,
)


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

        def cf(val):
            t, nt, r, nr = val
            return nr != 0

        def bf(val):
            t, nt, r, nr = val
            q = r // nr
            return (nt, t - q * nt, nr, r - q * nr)

        t, nt, r, nr = lax.while_loop(cf, bf, (0, 1, m, a))
        return jnp.where(t < 0, t + m, t)
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

    if check_for_tracing_mode():
        nj = jnp.asarray(n)
        # Avoid log2(0); define result 0 for n<=1
        return jnp.where(
            nj <= 1, jnp.int64(0), jnp.ceil(jnp.log2(nj)).astype(jnp.int64)
        )
    else:
        # Pure Python int path, exact and safe
        if hasattr(n, "bit_length"):
            return (n - 1).bit_length() if n > 1 else 0
        # Fallback (should not occur in practice)
        return 0


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
