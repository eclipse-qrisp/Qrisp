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

import jax.numpy as jnp

from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv
from qrisp.core import QuantumArray
from qrisp.environments import conjugate, custom_inversion
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import smallest_power_of_two
from qrisp.jasp import jrange, q_while_loop
from qrisp.qtypes import QuantumModulus
from qrisp.typing import NDArrayLike

from .classical import bitrev7, bitrevm, modpow_jax

############################################################
############### Number-theoretic transform #################
############################################################


# Efficient O(n log n) implementation
def qntt(f: QuantumArray, root: int, inv: bool = False) -> None:
    r"""
    Computes the number theoretic transform (NTT) in-place.

    Parameters
    ----------
    f : QuantumArray, shape (n,)
        An array of QuantumModulus representing a vector in $\mathbb Z_q^n$.
        The modulus $q$ must be prime, and $n | (q-1)$ must be satisfied.
    root : int
        An $n$-th root of unity modulo $q$.

    Examples
    --------

    Compute the NTT of $a=(3,1,4,9)\in\mathbb Z_{13}^4$ with respect to the $4$-th root of unity $\zeta=5$ modulo $13$.

    ::

        @jaspify
        def main():

            n, q, root = 4, 13, 5
            a = jnp.array([3, 1, 4, 9])

            qa = QuantumArray(QuantumModulus(q), shape=(n,))
            qa[:] = a

            qntt(qa, root)

            return measure(qa)

        print(main())
        # [10.  7.  9.  8.]

    """

    if inv:
        return qntt_inv(f, root)

    n = f.shape[0]
    m = smallest_power_of_two(n)
    q = f.qtype.modulus

    def cond_fun_inner(val):
        start, len_, i, f = val
        return start < n

    def body_fun_inner(val):
        start, len_, i, f = val

        # Repleace bitrev7 by general procedure for arbitrary n
        zeta = modpow_jax(root, bitrevm(i, m - 1), q)

        for j in jrange(start, start + len_):
            # Reversible implementation of steps 8-10
            f[j + len_] *= zeta
            f[j] += f[j + len_]
            f[j + len_] *= q - 2  # the same as *= -2
            f[j + len_] += f[j]

        return start + 2 * len_, len_, i + 1, f

    def cond_fun_outer(val):
        len_, i, f = val
        return len_ >= 2

    def body_fun_outer(val):
        len_, i, f = val

        _, _, i, _ = q_while_loop(cond_fun_inner, body_fun_inner, (0, len_, i, f))

        return len_ // 2, i, f

    q_while_loop(cond_fun_outer, body_fun_outer, (n // 2, 1, f))


# Efficient O(n log n) implementation
def qntt_inv(f: QuantumArray, root: int) -> None:
    r"""
    Computes the inverse number theoretic transform (NTT) in-place.

    Parameters
    ----------
    f : QuantumArray[QuantumModulus]
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$.
        The modulus $q$ must be prime, and $n | (q-1)$ must be satisfied.
    root : int
        An $n$-th root of unity modulo $q$.

    """

    n = f.shape[0]
    m = smallest_power_of_two(n)
    q = f.qtype.modulus

    i = n // 2 - 1

    def cond_fun_inner(val):
        start, len_, i, f = val
        return start < n

    def body_fun_inner(val):
        start, len_, i, f = val

        # Repleace bitrev7 by general procedure for arbitrary n
        zeta = modpow_jax(root, bitrevm(i, m - 1), q)

        for j in jrange(start, start + len_):
            # Reversible implementation of steps 9-10
            f[j] += f[j + len_]
            f[j + len_] *= 2
            f[j + len_] -= f[j]
            f[j + len_] *= zeta

        return start + 2 * len_, len_, i - 1, f

    def cond_fun_outer(val):
        len_, i, f = val
        return len_ <= n // 2

    def body_fun_outer(val):
        len_, i, f = val

        _, _, i, _ = q_while_loop(cond_fun_inner, body_fun_inner, (0, len_, i, f))

        return 2 * len_, i, f

    q_while_loop(cond_fun_outer, body_fun_outer, (2, i, f))

    n_half_inv = modinv(n // 2, q)

    for i in jrange(n):
        f[i] *= n_half_inv


def _base_case_multipy(
    a0: QuantumModulus,
    a1: QuantumModulus,
    b0: QuantumModulus,
    b1: QuantumModulus,
    c0: QuantumModulus,
    c1: QuantumModulus,
    gamma: int,
    inv: bool = False,
) -> None:
    r"""
    Computes the product of two degree-one polynomials with respect to a quadratic modulus.

    .. math::

        c_0+c_1X = (a_0+a_1X)(b_0+b_1X) mod (X^2-\gamma)

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
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import modinv

    q = a0.modulus
    aux = QuantumModulus(q)
    injected_mul = aux << (lambda a, b: a * b)

    gamma_inv = modinv(gamma, q)

    with conjugate(injected_mul)(a0, b0):
        if inv:
            c0 -= aux
        else:
            c0 += aux
    with conjugate(injected_mul)(a1, b1):
        aux *= gamma
        if inv:
            c0 -= aux
        else:
            c0 += aux
        aux *= gamma_inv
    with conjugate(injected_mul)(a0, b1):
        if inv:
            c1 -= aux
        else:
            c1 += aux
    with conjugate(injected_mul)(a1, b0):
        if inv:
            c1 -= aux
        else:
            c1 += aux

    aux.delete()


def multiply_qntts(f: QuantumArray, g: NDArrayLike, root: int, inv: bool = False) -> QuantumArray:
    r"""
    Computes the product of two NTT representations.

    Parameters
    ----------
    f : QuantumArray, shape (n,)
        An array of QuantumModulus representing a vector in $\mathbb Z_q^n$ in NTT representation.
    g : NDArrayLike, shape (n,)
        An array representing a vector in $\mathbb Z_q^n$ in NTT representation.

    Returns
    -------
    QuantumArray, shape (n,)
        An array of QuantumModulus representing a vector in $\mathbb Z_q^n$ in NTT representation.
        The result of the multiplication in the NTT domain.

    Examples
    --------

    Multiply the NTTs $\hat{a}=(3,1,4,9), \hat{b}=(1,2,3,4)\in\mathbb Z_{13}^4$ with respect to the $4$-th root of unity $\zeta=5$ modulo $13$.

    ::

        @jaspify
        def main():

            n, q, root = 4, 13, 5
            a_hat = jnp.array([3, 1, 4, 9])
            b_hat = jnp.array([1, 2, 3, 4])

            qa = QuantumArray(QuantumModulus(q), shape=(n,))
            qa[:] = a_hat

            res = multiply_qntts(qa, b_hat, root)

            return measure(res)

        print(main())
        # [0. 7. 1. 4.]

    """

    result = f.duplicate()
    n = f.shape[0]
    m = smallest_power_of_two(n)
    q = f.qtype.modulus

    for i in jrange(n // 2):
        gamma = modpow_jax(root, 2 * bitrevm(i, m - 1) + 1, q)  # Does this work for general n?
        _base_case_multipy(
            f[2 * i], f[2 * i + 1], g[2 * i], g[2 * i + 1], result[2 * i], result[2 * i + 1], gamma, inv=inv
        )

    return result


""""
def qntt_mat_mul(A, B, out, q, n, k, root):
    for i0 in jrange(k):
        for i1 in jrange(k):

            def true_fun():
                aux = QuantumArray(QuantumModulus(q), (n,))
                multiply_qntts(B[i0, i1, :], A[i0, :], aux, n, q, root)
                out[i1, :] += aux
                multiply_qntts(B[i0, i1, :], A[i0, :], aux, n, q, root, True)
                aux.delete()

            q_cond(A[i0, i1] != 0, true_fun, lambda: None)

    return out
"""

# Workaround to keep the docstring

temp = qntt.__doc__
qntt = custom_inversion(qntt)
qntt.__doc__ = temp

temp = multiply_qntts.__doc__
multiply_qntts = custom_inversion(multiply_qntts)
multiply_qntts.__doc__ = temp
