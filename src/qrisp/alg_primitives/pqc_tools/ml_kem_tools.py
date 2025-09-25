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

from qrisp import QuantumBool, QuantumModulus, QuantumArray, control, conjugate, invert
from qrisp.jasp import jrange, q_while_loop
import jax
import numpy as np
import jax.numpy as jnp
from jax import lax

# Auxiliary functions

def bitrev7(r: int) -> int:
    """Reverse the lowest 7 bits of r (0..127)."""
    if not (0 <= r < 128):
        raise ValueError("r must be in [0, 127]")
    rev = 0
    x = r
    for _ in range(7):
        rev = (rev << 1) | (x & 1)
        x >>= 1
    return rev


@jax.jit
def bitrev7(r):
    """
    JAX-traceable version: reverse the lowest 7 bits of integer(s) r (0..127).
    Accepts scalar or array-like input; returns same-shape integer array.
    Note: runtime range checks are omitted for JIT-compatibility. Use a wrapper
    if you want explicit Python-side validation.
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

"""
# Example usage
if __name__ == "__main__":
    import numpy as np
    a = 5
    x = 117
    q = 1000000007
    print(modpow_jax(a, x, q))          # JAX-traceable
    #print(jax.jit(lambda A,X,Q: modpow_jax(A,X,Q))(a, x, q))  # jit compiled

    # vectorized example
    a_vec = jnp.array([2, 3, 4])
    x_vec = jnp.array([10, 20, 30])
    print(modpow_jax(a_vec, x_vec, q))
"""


# NTT

# Efficient O(n log n) implementation
def q_ntt(x : QuantumArray, q: int, n: int, root : int, inv = False):
    """
    Number theoretic transform. (in-place)

    Parameters
    ----------
    x : QuantumArray[QuantumModulus]
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$.
        The modulus $q$ must be prime, and $n | (q-1)$ must be satisfied.
    root : int
        An $n$-th root of unity modulo $q$.

    """

    if inv:
        return q_ntt_inv(x, q, n, root)

    i = 1

    def cond_fun_inner(val):
        start, len_, i, x = val
        return start < n
    

    def body_fun_inner(val):
        start, len_, i, x = val

        # Repleace bitrev7 by general procedure for arbitrary n
        zeta = modpow_jax(root, bitrev7(i), q)

        for j in jrange(start, start + len_):
            # Reversible implementation of steps 8-10
            x[j + len_] *= zeta
            x[j] += x[j + len_]
            x[j + len_] *= -2
            x[j + len_] += x[j]
        
        return start + 2 * len_, len_, i + 1, x
    

    def cond_fun_outer(val):
        len_, i, x = val
        return len_ >= 2


    def body_fun_outer(val):
        len_, i, x = val

        q_while_loop(cond_fun_inner, body_fun_inner, (0, len_, i, x))

        return len_ // 2, i, x
    

    q_while_loop(cond_fun_outer, body_fun_outer, (n // 2, i, x))


# Efficient O(n log n) implementation
def q_ntt_inv(x : QuantumArray, q: int, n: int, root : int):
    """
    Number theoretic transform. (in-place)

    Parameters
    ----------
    x : QuantumArray[QuantumModulus]
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$.
        The modulus $q$ must be prime, and $n | (q-1)$ must be satisfied.
    root : int
        An $n$-th root of unity modulo $q$.

    """

    i = n // 2 - 1

    def cond_fun_inner(val):
        start, len_, i, x = val
        return start < n
    

    def body_fun_inner(val):
        start, len_, i, x = val

        # Repleace bitrev7 by general procedure for arbitrary n
        zeta = modpow_jax(root, bitrev7(i), q)

        for j in jrange(start, start + len_):
            # Reversible implementation of steps 9-10
            x[j] += x[j + len_]
            x[j + len_] *= 2
            x[j + len_] -= x[j]
            x[j + len_] *= zeta
        
        return start + 2 * len_, len_, i - 1, x
    

    def cond_fun_outer(val):
        len_, i, x = val
        return len_ <= n // 2


    def body_fun_outer(val):
        len_, i, x = val

        q_while_loop(cond_fun_inner, body_fun_inner, (0, len_, i, x))

        return 2 * len_, i, x
    

    q_while_loop(cond_fun_outer, body_fun_outer, (2, i, x))

    n_inv = pow(n, -1, q)
    # Multiply every entry by modular inverse of n mod q
    for i in jrange(n):
        x[i] *= n_inv


# Multiply NTTs

def q_base_case_multipy(a0: QuantumModulus, a1: QuantumModulus, b0: QuantumModulus, b1: QuantumModulus, c0: QuantumModulus, c1: QuantumModulus, gamma: int, inv = False):
    """
    Computes the product of two degree-one polynomials with respect to a quadratic modulus.

    .. math::

        c_0+c_1X = (a_0+a_1X)(b_0+b_1X)/(X^2-\gamma)

    Parameters
    ----------
    a0, a1 : QuantumModulus or int
        The coefficients of $a_0+a_1X$.
    b0, b1 : QuantumModulus or int
        The coefficients of $b_0+b_1X$.
    c0, c1 : QuantumModulus or int
        The coefficients of $c_0+c_1X$.
    gamma : int
        The modulus is $X^2-\gamma$.
    
    """

    gamma_inv = pow(gamma, -1, q)

    # c0 = a0 * b0 + a1 * b1 * gamma

    temp = a0 * b0
    
    # c1 = a0 * b1 + a1 * b0

    # TODO


def q_multiply_ntt(x: QuantumArray, y: QuantumArray, result: QuantumArray, q: int, n: int, root : int, inv = False):
    """
    Computes the product of two NTT representations.

    Parameters
    ----------
    x : QuantumArray[QuantumModulus] or array
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$ in NTT representation.
    y : QuantumArray[QuantumModulus] or array
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$ in NTT representation.
    result : QuantumArray[QuantumModulus]
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$ in NTT representation.
        The result of the multiplication in the NTT domain.
    root : int
        An $n$-th root of unity modulo $q$.

    """

    #n = x.size
    #q = x[0].modulus

    #result = QuantumArray(QuantumModulus(q), shape = (n,))

    for i in jrange(n // 2):
        gamma = modpow_jax(root, 2*bitrev7(i) + 1, q) # Does this work for general n?
        q_base_case_multipy(x[2*i], x[2*i + 1], y[i], y[2*i + 1], gamma, result[2*i], result[2*i + 1], inv = inv)

    return result


# ML-KEM LWE oracle

def create_lwe_oracle(a_hat, t_hat, q, n, root):
    """
    Generates the LWE oracle for a given ML-KEM instance. The oracle marks states $(s,e)$ that are solutuons to the LWE problem

    .. math::

        \hat{a}\circ\hat{s} + \hat{e} = \hat{0}
    
    Parameters
    ----------
    a_hat : numpy.ndarray
        An array of size ``(n,)`` representing a vector in $\mathbb Z_q^n$ in NTT representation.
    t_hat : numpy.ndarray
        An array of size ``(n,)`` representing a vector in $\mathbb Z_q^n$ in NTT representation.

    Returns
    -------
    function 
        A function that receives a pair QuantumArray[QuantumModulus] of arrays of size ``(n,)`` of type QuantumModulus(q) representing vectors in $\mathbb Z_q^n$,
        and marks states $(s,e)$ that are solutions to the LWE problem.

    """

    def lwe_oracle(s_hat, e_hat):
        """
        Marks all solutions $(\hat{s},\hat{e})$ satisfying the LWE equation in NTT domain $\hat{a}\circ\hat{s}+\hat{e}=\hat{t}$.

        Parameters
        ----------
        s_hat : QuantumArray[QuantumModulus]
            An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$.
        e_hat : QuantumArray[QuantumModulus]
            An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$.

        """

        # Calculate NTT
        q_ntt(s_hat, q, n, root)
        q_ntt(e_hat, q, n, root)

        result = QuantumArray(QuantumModulus(q), shape = (n,))
        result[:] = -t_hat

        result += e_hat

        temp = QuantumArray(QuantumModulus(q), shape = (n,))

        q_multiply_ntt(a_hat, s_hat, temp, q, n, root)

        result += temp

        # Comparison 
        comp_func = lambda x, y : y == y

        qb_array = QuantumArray(QuantumBool(), shape = (n,))

        injected_comp_func = (qb_array << comp_func)

        zeros = np.zeros(shape = (n,))

        with conjugate(injected_comp_func)(result, zeros):
            # Tag state where all entries of qb_array are True
            pass

        # Uncompute result
        result -= temp
        result -= e_hat
        result.delete()

        # Uncompute temp
        q_multiply_ntt(a_hat, s_hat, temp, q, n, root, inv = True)
        temp.delete()

        # Calculate inverse NTT
        with invert():
            q_ntt(s_hat, q, n, root)
            q_ntt(e_hat, q, n, root)


    return lwe_oracle

