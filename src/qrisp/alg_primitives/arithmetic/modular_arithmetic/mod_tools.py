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

"""Provides Montgomery encoding/decoding and modular inverse helper functions."""

import jax.numpy as jnp
from jax import Array, lax

from qrisp import check_for_tracing_mode


def montgomery_decoder(y: int | float, R: int | float, N: int) -> int | float:
    """Montgomery-decode y as y*R^{-1} mod N.

    Not traceable under ``jax.jit``. Use ``jasp_mod_tools.montgomery_decoder``
    for Jasp-traced code.

    Parameters
    ----------
    y : int
        Montgomery-encoded value.
    R : int or float
        Montgomery radix used during encoding. A value with ``0 < R < 1`` is
        interpreted as ``2**m`` for a negative Montgomery shift ``m``.
    N : int
        Modulus.

    Returns
    -------
    int
        Decoded value in standard representation.

    Examples
    --------
    >>> encoded = montgomery_encoder(42, 1024, 97)
    >>> montgomery_decoder(encoded, 1024, 97)
    42

    """
    # TODO: this whole function crashes under jax.jit tracing (Python `if`/`int()`
    # on a value that can be a tracer here, and mixed int/float dtypes reaching
    # modinv's lax.while_loop). This needs a tracing-safe rewrite.
    if 0 < R < 1:
        R = int(modinv(int(R**-1), N))
    return (y * modinv(R, N)) % N  # type: ignore[return-value]


def montgomery_encoder(y: int | float, R: int | float, N: int) -> int:
    """Montgomery-encode y as y*R mod N.

    Not traceable under ``jax.jit``. Use ``jasp_mod_tools.montgomery_encoder``
    for Jasp-traced code.

    Parameters
    ----------
    y : int
        Value to encode.
    R : int or float
        Montgomery radix (e.g., R = 2^m). A value with ``0 < R < 1`` is
        interpreted as ``2**m`` for a negative Montgomery shift ``m``.
    N : int
        Modulus.

    Returns
    -------
    int
        y in Montgomery form.

    Examples
    --------
    >>> montgomery_encoder(42, 1024, 97)
    37

    """
    if 0 < R < 1:
        R = int(modinv(int(R**-1), N))
    return (int(y) % N * int(R) % N) % N


def egcd(
    a: int | float | Array, b: int | float | Array
) -> tuple[int | float | Array, int | float | Array, int | float | Array]:
    """Extended Euclidean Algorithm.

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
        (g, x, y) with gcd and Bézout coefficients.

    Examples
    --------
    >>> g, x, y = egcd(35, 15)
    >>> g
    5
    >>> 35 * x + 15 * y
    5

    """
    if a == 0:
        return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)


def modinv(a: int | float | Array, m: int | float | Array) -> int | float | Array:
    """Modular inverse t = a^{-1} mod m.

    Parameters
    ----------
    a : int
        Value to invert (must be coprime to m).
    m : int
        Modulus.

    Returns
    -------
    int
        Modular inverse in [0, m).

    Raises
    ------
    ValueError
        If a and m are not coprime, so no inverse exists (non-traced mode only).

    Examples
    --------
    >>> t = modinv(3, 11)
    >>> t
    4
    >>> (3 * t) % 11
    1

    """
    if check_for_tracing_mode():

        def cf(val):
            _, _, _, new_r = val
            return new_r != 0

        def bf(val):
            t, new_t, r, new_r = val
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r - quotient * new_r
            return t, new_t, r, new_r

        t, _, _, _ = lax.while_loop(cf, bf, (0, 1, m, a))

        # Ensure result is in [0, MOD)
        return jnp.where(t < 0, t + m, t)

    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m
