"""********************************************************************************
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

import jax
import jax.numpy as jnp
from jax import Array, lax
import numpy as np

from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import smallest_power_of_two
from qrisp.typing import NDArrayLike, ScalarLike


@jax.jit
def bitrev7(r: ScalarLike) -> Array:
    """
    Compute the bit-reversal of r with respect to 7 bits.

    Parameters
    ----------
    r : ScalarLike
        The integer to be bit-reversed.

    Returns
    -------
    jax.Array
        The bit-reversed integer.
    """
    x = jnp.asarray(r, dtype=jnp.uint8)

    def body(i, val):
        rev, x = val
        rev = (rev << 1) | (x & 1)
        x = x >> 1
        return (rev, x)

    rev, _ = lax.fori_loop(0, 7, body, (jnp.uint8(0), x))
    return rev.astype(jnp.int32)


@jax.jit
def bitrevm(r: ScalarLike, m: ScalarLike) -> Array:
    """
    Compute the bit-reversal of r with respect to m bits.

    Parameters
    ----------
    r : ScalarLike
        The integer to be bit-reversed.
    m : ScalarLike
        The number of bits to consider.

    Returns
    -------
    jax.Array
        The bit-reversed integer.
    """
    x = jnp.asarray(r)

    def body(i, val):
        rev, x = val
        rev = (rev << 1) | (x & 1)
        x = x >> 1
        return (rev, x)

    rev, _ = lax.fori_loop(0, m, body, (jnp.uint32(0), x))
    return rev


@jax.jit
def modpow_jax(a: ScalarLike, x: ScalarLike, q: ScalarLike) -> Array:
    """
    Computes (a ** x) % q efficiently in a JAX-traceable manner.

    This function utilizes the square-and-multiply algorithm (binary exponentiation)
    implemented with `jax.lax.while_loop`, making it fully compatible with `@jax.jit`,
    `@jax.vmap`, and other JAX transformations.

    Parameters
    ----------
    a : ScalarLike
        The base value(s). Can be a scalar or an array.
    x : ScalarLike
        The exponent value(s). Must be non-negative integer(s).
    q : ScalarLike
        The modulus value(s).

    Returns
    -------
    jax.Array
        The result of (a ** x) % q. The output shape matches the broadcasted
        shape of the inputs.

    Notes
    -----
    - Inputs are automatically broadcasted against each other.
    - The intermediate multiplications scale up to `(q-1)**2`. Ensure that the
      inferred JAX data type (usually `int32` by default) is large enough to
      hold this value to prevent integer overflow.
    """
    a = jnp.asarray(a) % jnp.asarray(q)
    exp = jnp.asarray(x)
    mod = jnp.asarray(q)

    # loop state: (base, exp, result)
    init = (a, exp, jnp.ones_like(a))

    def cond_fn(state):
        _, e, _ = state
        return jnp.any(e > 0)

    def body_fn(state):
        base, e, res = state

        # If the lowest bit is 1, multiply the result by the base modulo q
        res = jnp.where((e & 1) != 0, (res * base) % mod, res)

        # Square the base modulo q
        base = (base * base) % mod

        # Shift exponent right by 1
        e = e >> 1

        return (base, e, res)

    _, _, result = lax.while_loop(cond_fn, body_fn, init)
    return result


# NIST FIPS 203, Algorithm 9
@jax.jit
def ntt(f: NDArrayLike, q: ScalarLike, root: ScalarLike) -> Array:
    r"""
    Computes the forward Number Theoretic Transform (NTT) for ML-KEM (FIPS 203).

    Note: This is an incomplete NTT that stops at length=2, leaving the
    polynomials as degree 1 polynomials in the NTT domain.

    Parameters
    ----------
    f : NDArrayLike, shape (n,)
        1-D array of polynomial coefficients.
        The size $n$ must be a power of 2, e.g., 256.
    q : ScalarLike
        The modulus.
    root : ScalarLike
        The $n$-th primitive root of unity modulo $q$.

    Returns
    -------
    Array, shape (n,)
        The transformed array of coefficients in the NTT domain.

    """

    # Cast to JAX array (handles np.ndarray, list, or jnp.ndarray seamlessly)
    f = jnp.asarray(f)
    n = f.shape[0]
    m = smallest_power_of_two(n)

    def outer_cond(val):
        len_, i, current_f = val
        return len_ >= 2

    def outer_body(val):
        len_, i, current_f = val

        def inner_cond(inner_val):
            start, len_inner, i_inner, f_inner = inner_val
            return start < n

        def inner_body(inner_val):
            start, len_inner, i_inner, f_inner = inner_val

            # Replaces bitrev7 by a general procedure for arbitrary n
            zeta = modpow_jax(root, bitrevm(i_inner, m - 1), q)

            def innermost_body(j, f_innermost):
                t = (zeta * f_innermost[j + len_inner]) % q

                new_val_high = (f_innermost[j] - t) % q
                new_val_low = (f_innermost[j] + t) % q

                f_innermost = f_innermost.at[j + len_inner].set(new_val_high)
                f_innermost = f_innermost.at[j].set(new_val_low)
                return f_innermost

            # lax.fori_loop handles the innermost loop, replacing jrange
            f_inner = lax.fori_loop(start, start + len_inner, innermost_body, f_inner)

            return start + 2 * len_inner, len_inner, i_inner + 1, f_inner

        # lax.while_loop handles dynamic state, replacing q_while_loop
        _, _, i_new, f_new = lax.while_loop(inner_cond, inner_body, (0, len_, i, current_f))

        return len_ // 2, i_new, f_new

    _, _, f_final = lax.while_loop(outer_cond, outer_body, (n // 2, 1, f))
    return f_final


# NIST FIPS 203, Algorithm 10
@jax.jit
def ntt_inv(f: NDArrayLike, q: ScalarLike, root: ScalarLike) -> Array:
    r"""
    Computes the inverse Number Theoretic Transform (INTT) for ML-KEM (FIPS 203).

    Reconstructs the polynomial from its incomplete NTT domain representation.

    Parameters
    ----------
    f : NDArrayLike, shape (n,)
        1-D array of transformed polynomial coefficients.
        The size $n$ must be a power of 2, e.g., 256.
    q : int
        The modulus.
    root : int
        The $n$-th primitive root of unity modulo $q$.

    Returns
    -------
    Array, shape (n,)
        The inverse-transformed array of coefficients.

    """
    # Cast to JAX array (handles np.ndarray, list, or jnp.ndarray seamlessly)
    f = jnp.asarray(f)

    n = f.shape[0]
    m = smallest_power_of_two(n)
    i = n // 2 - 1

    def outer_cond(val):
        len_, i_val, current_f = val
        return len_ <= n // 2

    def outer_body(val):
        len_, i_val, current_f = val

        def inner_cond(inner_val):
            start, len_inner, i_inner, f_inner = inner_val
            return start < n

        def inner_body(inner_val):
            start, len_inner, i_inner, f_inner = inner_val

            zeta = modpow_jax(root, bitrevm(i_inner, m - 1), q)

            def innermost_body(j, f_innermost):

                # Classical equivalent of reversible steps 9-10
                new_j = (f_innermost[j] + f_innermost[j + len_inner]) % q

                # f[j + len_] *= 2, f[j + len_] -= f[j], f[j + len_] *= zeta
                temp = (f_innermost[j + len_inner] * 2) % q
                temp = (temp - new_j) % q
                new_j_len = (temp * zeta) % q

                f_innermost = f_innermost.at[j].set(new_j)
                f_innermost = f_innermost.at[j + len_inner].set(new_j_len)
                return f_innermost

            f_inner = lax.fori_loop(start, start + len_inner, innermost_body, f_inner)

            return start + 2 * len_inner, len_inner, i_inner - 1, f_inner

        _, _, i_new, f_new = lax.while_loop(inner_cond, inner_body, (0, len_, i_val, current_f))

        return 2 * len_, i_new, f_new

    _, _, f_intermediate = lax.while_loop(outer_cond, outer_body, (2, i, f))

    # Multiply by the modular inverse of n/2
    n_half_inv = modinv(n // 2, q)

    # Vectorized final multiplication for JAX efficiency
    f_final = (f_intermediate * n_half_inv) % q

    return f_final


def compute_ntt(f: np.ndarray, n: int, q: int, root: int) -> np.ndarray:
    """
    Computes the forward Number Theoretic Transform (NTT) for ML-KEM (FIPS 203).

    Note: This is an incomplete NTT that stops at length=2, leaving the
    polynomials as degree 1 polynomials in the NTT domain.

    Parameters
    ----------
    f : np.ndarray
        1-D array of polynomial coefficients.
    n : int
        The size of the transform (must be a power of 2, e.g., 256).
    q : int
        The modulus.
    root : int
        The n-th primitive root of unity modulo q.

    Returns
    -------
    np.ndarray
        The transformed array of coefficients in the NTT domain.

    """
    m = int(np.ceil(np.log2(n)))
    f = f.copy() % q
    i = 1
    length = n // 2

    while length >= 2:
        for start in range(0, n, 2 * length):
            zeta = modpow_jax(root, bitrevm(i, m - 1), q)
            i += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % q
                f[j + length] = (f[j] - t) % q
                f[j] = (f[j] + t) % q
        length //= 2
    return f


def compute_inv_ntt(f: np.ndarray, n: int, q: int, root: int) -> np.ndarray:
    """
    Computes the inverse Number Theoretic Transform (INTT) for ML-KEM (FIPS 203).

    Reconstructs the polynomial from its incomplete NTT domain representation.

    Parameters
    ----------
    f : np.ndarray
        1-D array of transformed polynomial coefficients.
    n : int
        The size of the transform (must be a power of 2, e.g., 256).
    q : int
        The modulus.
    root : int
        The n-th primitive root of unity modulo q.

    Returns
    -------
    np.ndarray
        The inverse-transformed array of coefficients.

    """
    m = int(np.ceil(np.log2(n)))
    f = f.copy() % q
    i = n // 2 - 1
    length = 2

    while length <= n // 2:
        for start in range(0, n, 2 * length):
            zeta = modpow_jax(root, bitrevm(i, m - 1), q)
            i -= 1
            for j in range(start, start + length):
                t = f[j] % q
                f[j] = (t + f[j + length]) % q
                f[j + length] = (zeta * (f[j + length] - t)) % q
        length *= 2

    n_half_inv = modinv(n // 2, q)

    for idx in range(n):
        f[idx] = (f[idx] * n_half_inv) % q

    return f


def base_case_multiply(a_0: int, a_1: int, b_0: int, b_1: int, gamma: int, q: int):

    c = np.zeros(2, dtype=np.int64)

    c[0] = (a_0 * b_0 + ((a_1 * b_1) % q) * gamma) % q
    c[1] = (a_0 * b_1 + a_1 * b_0) % q
    return c


def multiply_ntts(f_hat, g_hat, n: int, q: int, root: int):

    h_hat = np.zeros(n, dtype=np.int64)
    m = int(np.ceil(np.log2(n)))

    for i in range(n // 2):
        gamma = modpow_jax(root, int(2 * bitrevm(i, m - 1)) + 1, q)
        h = base_case_multiply(f_hat[2 * i], f_hat[2 * i + 1], g_hat[2 * i], g_hat[2 * i + 1], gamma, q)
        h_hat[2 * i] = h[0]
        h_hat[2 * i + 1] = h[1]

    return h_hat
