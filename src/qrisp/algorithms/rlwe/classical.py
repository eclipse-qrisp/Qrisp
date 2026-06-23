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

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from qrisp import modinv


@jax.jit
def bitrev7(r):
    x = jnp.asarray(r, dtype=jnp.uint8)

    def body(i, val):
        rev, x = val
        rev = (rev << 1) | (x & 1)
        x = x >> 1
        return (rev, x)

    rev, _ = lax.fori_loop(0, 7, body, (jnp.uint8(0), x))
    return rev.astype(jnp.int32)


@jax.jit
def bitrevm(r, m):
    x = jnp.asarray(r, dtype=jnp.uint32)

    def body(i, val):
        rev, x = val
        rev = (rev << 1) | (x & 1)
        x = x >> 1
        return (rev, x)

    rev, _ = lax.fori_loop(0, m, body, (jnp.uint32(0), x))
    return rev.astype(jnp.int32)


@jax.jit
def modpow_jax(a, x, q, dtype=jnp.int64):
    """
    Compute a**x % q in a JAX-traceable way (supports JIT).
    - a, x, q can be scalars or arrays (broadcastable).
    - x must be nonnegative integer(s).
    - dtype: integer dtype to use (choose wide enough for q).
    """
    a = jnp.asarray(a, dtype=dtype) % jnp.asarray(q, dtype=dtype)
    exp = jnp.asarray(x, dtype=dtype)
    mod = jnp.asarray(q, dtype=dtype)

    # loop state: (base, exp, result)
    init = (a, exp, jnp.ones_like(a, dtype=dtype))

    def cond_fn(state):
        _, e, _ = state
        return jnp.any(e > 0)

    def body_fn(state):
        base, e, res = state
        # if (e & 1): res = (res * base) % mod
        res = jnp.where((e & 1) != 0, (res * base) % mod, res)
        base = (base * base) % mod
        e = e >> 1
        return (base, e, res)

    _, _, result = lax.while_loop(cond_fn, body_fn, init)
    return result


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
            zeta = modpow_jax(root, bitrevm(i, m-1), q)
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
            zeta = modpow_jax(root, bitrevm(i, m-1), q)
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
    
        gamma = modpow_jax(root, int(2 * bitrevm(i, m-1)) + 1, q)        
        h = base_case_multiply(f_hat[2*i], f_hat[2*i+1], g_hat[2*i], g_hat[2*i+1], gamma, q)
        h_hat[2*i] = h[0]
        h_hat[2*i+1] = h[1]
        
    return h_hat
