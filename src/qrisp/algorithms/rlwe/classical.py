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

def compute_ntt(f, n, q, root):
    m = int(np.ceil(np.log2(n)))
    f = f.copy() % q
    i = 1
    length = n//2
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

def compute_inv_ntt(f, n, q, root):
    m = int(np.ceil(np.log2(n)))
    f = f.copy() % q
    i = n//2 - 1
    length = 2
    while length <= n//2:
        for start in range(0, n, 2 * length):
            zeta = modpow_jax(root, bitrevm(i, m-1), q)
            i -= 1
            for j in range(start, start + length):
                t = f[j] % q
                f[j] = t + f[j + length]
                f[j + length] = (zeta*(f[j + length] - t)) % q
        length *= 2

    n_inv = modinv(n, q)
    for i in range(n):
        f[i] *= n_inv
    return f