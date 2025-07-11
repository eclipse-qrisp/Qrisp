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


import jax.numpy as jnp
import jax.lax as lax
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

BASE = 2**32
DTYPE = jnp.uint32
BASE_FL = float(BASE)

@register_pytree_node_class
@dataclass(frozen=True)
class BigInteger:
    digits: jnp.ndarray  # Little-endian base-2^32

    def __call__(self):
        r = lax.fori_loop(0, self.digits.shape[0], lambda i, val: jnp.float64(self.digits[i])*BASE_FL**jnp.float64(i) + val, 0.0)
        return r
        #powers = jnp.array([BASE**i for i in range(self.digits.shape[0])], dtype=jnp.float64)
        #return jnp.sum(self.digits.astype(jnp.float64) * powers)

    def tree_flatten(self):
        return (self.digits,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    @staticmethod
    def from_int_python(n : int, size=2):
        digits = []
        for i in range(size):
            digits.append(n % BASE)
            n //= BASE
        return BigInteger(jnp.array(digits, dtype=DTYPE))

    @staticmethod
    def from_int(n: int, size=2):
        def body_fun(i, args):
            digits, num = args
            digits = digits.at[i].set(jnp.uint32(num % BASE))
            num //= BASE
            return digits, num
        digits, _ = lax.fori_loop(0, size, body_fun, (jnp.zeros(size, dtype=DTYPE), n))
        return BigInteger(digits)
    
    @staticmethod
    def from_float(n: float, size=2):
        def body_fun(i, args):
            digits, num = args
            digits = digits.at[i].set(jnp.uint32(num % BASE))
            num //= BASE
            return digits, num
        digits, _ = lax.fori_loop(0, size, body_fun, (jnp.zeros(size, dtype=DTYPE), n))
        return BigInteger(digits)


    def __add__(self, other: "BigInteger") -> "BigInteger":
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, n)
        a, b = self.digits, other.digits
        result = jnp.zeros_like(a)
        carry = jnp.uint64(0)

        def add_step(i, state):
            carry, result = state
            s = jnp.uint64(a[i]) + jnp.uint64(b[i]) + carry
            digit = jnp.uint32(s % BASE)
            new_carry = jnp.uint64(s // BASE)
            result = result.at[i].set(jnp.uint32(digit))
            return new_carry, result

        carry, result = lax.fori_loop(0, a.shape[0], add_step, (carry, result))
        return BigInteger(result)
    
    def __sub__(self, other: "BigInteger") -> "BigInteger":
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, n)
        a, b = self.digits, other.digits
        result = jnp.zeros_like(a)

        def add_step(i, state):
            carry, result = state
            s, new_carry = lax.cond(jnp.uint64(a[i]) >= jnp.uint64(b[i]) + carry,
                                    lambda: (jnp.uint32(a[i] - b[i] - carry), 0),
                                    lambda: (jnp.uint32(jnp.uint64(a[i]) + jnp.uint64(BASE) - jnp.uint64(b[i]) - carry) , 1))
            result = result.at[i].set(jnp.uint32(s))
            return new_carry, result

        carry, result = lax.fori_loop(0, a.shape[0], add_step, (0, result))

        return BigInteger(result)

    def __mul__(self, other: "BigInteger") -> "BigInteger":
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, n)
        a, b = self.digits, other.digits
        result = jnp.zeros(n, dtype=DTYPE)

        def outer_loop(i, result):
            def inner_loop(j, result):
                idx = i + j
                product = jnp.uint64(a[i]) * jnp.uint64(b[j]) + jnp.uint64(result[idx])
                carry = product >> 32
                result = result.at[idx].set(jnp.uint32(product & 0xFFFFFFFF))
                result = result.at[idx + 1].add(jnp.uint32(carry))
                return result

            result = lax.fori_loop(0, n-i, inner_loop, result)
            return result

        result = lax.fori_loop(0, n, outer_loop, result)
        return BigInteger(result) 
    
    def __pow__(self, other):
        def body_fun(i, bi):
            return bi * self
        return lax.fori_loop(0, other - 1, body_fun, BigInteger(self.digits))

    def __repr__(self):
        return f"BigInteger(digits={self.digits.tolist()})"
    
    def __lt__(self, other: "BigInteger"):
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, n)
        m = other.digits.shape[0]
        
        assert(n == m)

        d0 = self.digits
        d1 = other.digits

        def body_fun(val):
            i, res_found = val
            i -= 1

            res_found = lax.cond(d0[i] < d1[i], lambda: 1, lambda: -5)
            res_found = lax.cond(d0[i] > d1[i], lambda: 0, lambda: res_found)

            return i, res_found
        
        def cond_fun(val):
            i, res_found = val
            return jnp.logical_and(res_found == -5, i > 0)
        
        _, res = lax.while_loop(cond_fun, body_fun, (n, -5))
        res = lax.cond(res == -5, lambda: 0, lambda: res)
        return jnp.bool(res)

    def __eq__(self, other: "BigInteger"):
        n = self.digits.shape[0]

        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, n)

        m = other.digits.shape[0]

        assert(n == m)

        d0 = self.digits
        d1 = other.digits

        return jnp.all(d0 == d1)
    
    def __ne__(self, other):
        return jnp.logical_not(self == other)
    
    def __le__(self, other: "BigInteger"):
        n = self.digits.shape[0]
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, n)
        m = other.digits.shape[0]

        assert(n == m)

        d0 = self.digits
        d1 = other.digits

        def body_fun(val):
            i, res_found = val
            i -= 1

            res_found = lax.cond(d0[i] < d1[i], lambda: 1, lambda: -5)
            res_found = lax.cond(d0[i] > d1[i], lambda: 0, lambda: res_found)

            return i, res_found
        
        def cond_fun(val):
            i, res_found = val
            return jnp.logical_and(res_found == -5, i > 0)
        
        _, res = lax.while_loop(cond_fun, body_fun, (n, -5))
        res = lax.cond(res == -5, lambda: 1, lambda: res)
        return jnp.bool(res)
    
    
    def __lshift__(self, shift):

        def body_fun(i, x):
            return lax.cond(jnp.bool(self.get_bit(i)), lambda: x.flip_bit(i + shift), lambda: x)
        
        return lax.fori_loop(0, self.digits.shape[0]*32 - shift, body_fun, BigInteger(jnp.zeros_like(self.digits)))
    

    def __rshift__(self, shift):

        def body_fun(i, x):
            return lax.cond(jnp.bool(self.get_bit(i)), lambda: x.flip_bit(i - shift), lambda: x)
        
        return lax.fori_loop(shift, self.digits.shape[0]*32, body_fun, BigInteger(jnp.zeros_like(self.digits)))
    
    
    def __mod__(self, other: "BigInteger"):
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, self.digits.shape[0])
        r, q = self.remainder_division(other)
        return r
    
    def __floordiv__(self, other: "BigInteger"):
        if not isinstance(other, BigInteger):
            other = BigInteger.from_int(other, self.digits.shape[0])
        r, q = self.remainder_division(other)
        return q
    
    def remainder_division(self, other: "BigInteger"): #TODO: Fails for very very big integers
        """ Return BigInteger r and q such that self = other*q + r with r < other. """
        n = self.digits.shape[0]
        m = other.digits.shape[0]

        assert(n == m)

        def cond_fun(args):
            q_approx, r_approx = args
            return r_approx > other 

        def body_fun(args):
            q_approx, r_approx = args
            nq_approx = BigInteger.from_float(jnp.floor(r_approx()/other()), n)
            r_approx = r_approx - nq_approx*other
            q_approx = q_approx + nq_approx
            return q_approx, r_approx

        q, r = lax.while_loop(cond_fun, body_fun, (BigInteger(jnp.zeros_like(self.digits)), BigInteger(self.digits)))

        return r, q
    
    def get_bit(self, i: int):
        pos = i//32
        pos_in = i%32
        return (self.digits[pos] >> pos_in) & 1
    
    def flip_bit(self, i: int):
        pos = i//32
        pos_in = i%32
        ds = jnp.copy(self.digits)
        ds = ds.at[pos].set(jnp.uint32(ds[pos] ^ (1 << pos_in)))
        return BigInteger(ds)
    
    def bit_size(self):
        pos_i = lax.while_loop(lambda i: jnp.logical_and(i < self.digits.shape[0] - 1, self.digits[i] == 0), lambda i: i+1, 0)
        return 32*pos_i + (jnp.ceil(jnp.log2(self.digits[pos_i]))).astype(jnp.int64)

    
def bi_modinv(a: BigInteger, m: BigInteger) -> BigInteger:
    n = a.digits.shape[0]
    bi0 = BigInteger.from_int(0, n)
    bi1 = BigInteger.from_int(1, n)
    
    t, new_t = bi0, bi1
    r, new_r = m, a

    def cond(state):
        _, new_t, r, new_r = state
        return new_r != 0

    def body(state):
        t, new_t, r, new_r = state
        quotient = r // new_r

        # Unsigned arithmetic with wraparound simulated modulo behavior
        t_updated = (t + m - (quotient * new_t) % m) % m
        r_updated = r - quotient * new_r

        return new_t, t_updated, new_r, r_updated

    final_t, _, final_r, _ = lax.while_loop(cond, body, (t, new_t, r, new_r))

    return final_t


def bi_extended_euclidean(a, b):
    """
    Extended Euclidean Algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    Works with JAX and supports JIT compilation.
    """
    n = a.digits.shape[0]
    bi0 = BigInteger.from_int(0, n)
    bi1 = BigInteger.from_int(1, n)
    def cond_fun(state):
        r, _, _, _ = state
        return jnp.logical_not(r == bi0)

    def body_fun(state):
        r, old_r, s, old_s, t, old_t = state
        quotient = old_r // r
        return (old_r - quotient * r,
                r,
                old_s - quotient * s,
                s,
                old_t - quotient * t,
                t)

    # Initialize variables:
    # (r, old_r, s, old_s, t, old_t)
    state = (b, a, 0, 1, 1, 0)
    # Use lax.while_loop for JAX compatibility
    state = lax.while_loop(cond_fun, body_fun, state)

    r, old_r, s, old_s, t, old_t = state
    # old_r is gcd, old_s and old_t are the Bezout coefficients
    return old_r, old_s, old_t

def bi_montgomery_encode(x, R, modulus):
    return (x * R) % modulus

def bi_montgomery_decode(x_mon, R, modulus):
    return (x_mon * bi_modinv(R, modulus)) % modulus