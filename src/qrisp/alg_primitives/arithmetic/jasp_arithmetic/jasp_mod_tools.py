from qrisp import check_for_tracing_mode
from .jasp_bigintiger import BigInteger, bi_modinv
import jax.numpy as jnp
import numpy as np
import jax.lax as lax

def montgomery_decoder(y, R, N):
    R1 = modinv(R, N)
    return ((y % N) * (R1 % N)) % N


def montgomery_encoder(y, R, N):
    return ((y % N) * (R % N)) % N


def egcd(a, b):
    """
    Extended Euclidean Algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    Works with JAX and supports JIT compilation.
    """
    def cond_fun(state):
        r, _, _, _ = state
        return jnp.logical_not(r == 0)

    def body_fun(state):
        r, old_r, s, old_s, t, old_t = state
        quotient = old_r // r
        return (old_r - quotient * r,
                r,
                old_s - quotient * s,
                s,
                old_t - quotient * t,
                t)

    state = (b, a, 0, 1, 1, 0)
    state = lax.while_loop(cond_fun, body_fun, state)

    r, old_r, s, old_s, t, old_t = state
    return old_r, old_s, old_t


def modinv(a, m):
    if check_for_tracing_mode():
        where = jnp.where
        loop = lax.while_loop
    else:
        where = lambda a,b,c : int(np.where(a,b,c))
        def loop(cond_fun, body_fun, init_val):
            val = init_val
            while cond_fun(val):
              val = body_fun(val)
            return val

    def cf(val):
        t, new_t, r, new_r = val
        return new_r != 0
    
    def bf(val):
        t, new_t, r, new_r = val
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
        return t, new_t, r, new_r
    
    if isinstance(a, BigInteger):
        return bi_modinv(a, m)
    else:
        t, new_t, r, new_r = loop(cf, bf, (0, 1, m, a))
    # Ensure result is in [0, MOD)
        return where(t < 0, t + m, t)


def smallest_power_of_two(n):
    if check_for_tracing_mode():
        if isinstance(n, BigInteger):
            return n.bit_size()
        else:
            return (jnp.ceil(jnp.log2(n))).astype(jnp.int64)
    else:
        return int(np.ceil(np.log2(n)))
    
def best_montgomery_shift(b, N):
    """
    Find the smallest number m, such that $(N-1)*b < N*2**m$.
    """
    #TODO: Improve by taking N/(N-1) into account
    return smallest_power_of_two(b)
