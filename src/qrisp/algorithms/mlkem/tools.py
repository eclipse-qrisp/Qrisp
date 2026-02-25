from typing import Union
from qrisp import QuantumBool, QuantumModulus, QuantumArray, conjugate, modinv, mcz, mcx, amplitude_amplification, jlen, prepare
from qrisp import h as h_gate
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import smallest_power_of_two
from qrisp.jasp import jrange, q_while_loop, q_cond
import jax
import numpy as np
import jax.numpy as jnp
from jax import lax

from .classical import bitrev7, bitrevm, modpow_jax

############################################################
############### Number-theoretic transform #################
############################################################

# Efficient O(n log n) implementation
def q_ntt(x : QuantumArray, q: int, n: int, root : int, inv = False):
    """
    Number theoretic transform. (in-place)

    Parameters
    ----------
    x : QuantumArray[QuantumModulus]
        An array of size `(n,)` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$.
        The modulus $q$ must be prime, and $n | (q-1)$ must be satisfied.
    root : int
        An $n$-th root of unity modulo $q$.

    """

    if inv:
        return q_ntt_inv(x, q, n, root)

    m = jnp.int32(jnp.ceil(jnp.log2(n)))

    def cond_fun_inner(val):
        start, len_, i, x = val
        return start < n
    

    def body_fun_inner(val):
        start, len_, i, x = val

        # Repleace bitrev7 by general procedure for arbitrary n
        zeta = modpow_jax(root, bitrevm(i, m-1), q)

        for j in jrange(start, start + len_):
            # Reversible implementation of steps 8-10
            x[j + len_] *= zeta
            x[j] += x[j + len_]
            x[j + len_] *= q-2 # the same as *= -2 
            x[j + len_] += x[j]
        
        return start + 2 * len_, len_, i + 1, x
    

    def cond_fun_outer(val):
        len_, i, x = val
        return len_ >= 2


    def body_fun_outer(val):
        len_, i, x = val

        _, _, i, _ = q_while_loop(cond_fun_inner, body_fun_inner, (0, len_, i, x))

        return len_ // 2, i, x
    

    q_while_loop(cond_fun_outer, body_fun_outer, (n // 2, 1, x))

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
    m = jnp.int32(jnp.ceil(jnp.log2(n)))

    def cond_fun_inner(val):
        start, len_, i, x = val
        return start < n
    

    def body_fun_inner(val):
        start, len_, i, x = val

        # Repleace bitrev7 by general procedure for arbitrary n
        zeta = modpow_jax(root, bitrevm(i, m-1), q)

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

        _, _, i, _ = q_while_loop(cond_fun_inner, body_fun_inner, (0, len_, i, x))

        return 2 * len_, i, x
    

    q_while_loop(cond_fun_outer, body_fun_outer, (2, i, x))

    n_inv = modinv(n, q)
    for i in jrange(n):
        x[i] *= n_inv


def q_base_case_multipy(a0: QuantumModulus, a1: QuantumModulus, b0: QuantumModulus, b1: QuantumModulus, c0: QuantumModulus, c1: QuantumModulus, gamma: int, inv = False):
    """
    Computes the product of two degree-one polynomials with respect to a quadratic modulus. Alg. 12.

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
    # More efficient
    p00 = a0 * b0
    p11 = a1 * b1
    p11 *= gamma  # gamma is a classical constant in Z_q
    p01 = a0 * b1
    p10 = a1 * b0

    if inv:
        c0 -= p00
        c0 -= p11
        c1 -= p01
        c1 -= p10
    else:
        c0 += p00
        c0 += p11
        c1 += p01
        c1 += p10

    for tmp in (p10, p01, p11, p00):
        tmp.delete()


def q_multiply_ntt(f: QuantumArray, g: QuantumArray, result: QuantumArray, q: int, n: int, root : int, inv = False):
    """
    Computes the product of two NTT representations. Alg. 11.

    Parameters
    ----------
    x : QuantumArray[QuantumModulus] or array
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$ in NTT representation.
    y : QuantumArray[QuantumModulus] or array
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$ in NTT representation.
    result : QuantumArray[QuantumModulus]
        An array of size ``(n,)`` of type QuantumModulus(q) representing a vector in $\mathbb Z_q^n$ in NTT representation.
        The result of the multiplication in the NTT domain.

    """

    m = jnp.int32(jnp.ceil(jnp.log2(n)))

    for i in jrange(n // 2):
        gamma = modpow_jax(root, 2*bitrevm(i, m-1) + 1, q) # Does this work for general n?
        q_base_case_multipy(f[2*i], f[2*i + 1], g[2*i], g[2*i + 1], result[2*i], result[2*i + 1], gamma, inv = inv)

    return result

def q_ntt_mat_mul(A, B, out, q, n, k, root):
    for i0 in jrange(k):
        for i1 in jrange(k):
            def true_fun():
                aux = QuantumArray(QuantumModulus(q), (n,))
                q_multiply_ntt(B[i0,i1,:], A[i0,:], aux, q, n, root)
                out[i1,:] += aux
                q_multiply_ntt(B[i0,i1,:], A[i0,:], aux, q, n, root, True)
                aux.delete()
            q_cond(A[i0, i1] != 0, true_fun, lambda: None)

    return out

############################################################
###################### Grover utility ######################
############################################################

def grover_prepare_uniform_bits(arg, n, k, q, distr):
    arr = np.zeros(2**int(np.ceil(np.log2(q))), dtype=jnp.float64) # q must be static
    for i, v in enumerate(distr):
        arr[i] = arr[-i-1] = v
    arr /= np.sqrt(np.sum(arr))
    for i in jrange(k):
        for j in range(n):
            prepare(arg[i, j], arr)


def create_lwe_oracle(a_hat, t_hat, q, n, k, root, B_e):
    """
    Marks s if r = t - A*s has all coefficients with centered magnitude <= B_e.
    Assumes a_hat, t_hat, s_hat are in NTT domain.
    """
    def oracle(s_hat: QuantumArray):
        #q_ntt(s, q, n, root)
        rhat = QuantumArray(QuantumModulus(q), shape=(k,n))
        with conjugate(q_ntt_mat_mul)(a_hat, s_hat, rhat, q, n, k, root):
        #q_ntt_mat_mul(a_hat, s_hat, rhat, q, n, k, root)

            rhat -= t_hat
            for i in jrange(k):
                q_ntt_inv(rhat[i], q, n, root)

            qb = QuantumArray(QuantumBool(), shape=(k,n))
            small_inj = (qb << (lambda v: ((v <= B_e) or (v >= q - B_e))))
            with conjugate(small_inj)(rhat):
                h_gate(qb.flatten()[-1])
                mcx(qb.flatten()[:-1].qb_array, qb[-1], "balauca") #TODO: use mcz
                h_gate(qb.flatten()[-1])

            for i in jrange(k):
                q_ntt(rhat[i], q, n, root)
            rhat += t_hat
        
        #q_ntt_mat_mul(a_hat, s_hat, rhat, q, n, k, root, inv=True)
        rhat.delete()
    return oracle

def run_lwe_grover_amplification(a_hat, t_hat, q, n, k, root, iterations, B_e, distr):
    """
    Parameters
    ----------
    a_hat
        Array of shape (k,k,n)
    t_hat : QuantumArray[QuantumModulus] or array
        Array of shape (k,n)
    """
    # State prep and oracle
    def state_fn(arg): grover_prepare_uniform_bits(arg, n, k, q, distr)
    args = QuantumArray(QuantumModulus(q), shape=(k,n))
    oracle_fn = create_lwe_oracle(a_hat, t_hat, q, n, k, root, B_e)

    amplitude_amplification(
        [args],
        state_function=state_fn,
        oracle_function=oracle_fn,
        kwargs_oracle={}, 
        iter=iterations,
        reflection_indices=None 
    )

    return args