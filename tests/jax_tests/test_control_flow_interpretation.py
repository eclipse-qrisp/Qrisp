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

from qrisp import *
from jax.lax import fori_loop, switch
from jax import random

def test_control_flow_interpretation():
    
    @qache
    def inner_function(qf, i):
        qf[:] = i

    def test_f(i):
        a = QuantumFloat(5, -1)
        with invert():
            with QuantumEnvironment():
                with invert():
                    inner_function(a, i)
        b = measure(a)
        b += 4
        return b
    
    jasp_program = make_jaspr(test_f)(0.5)
    
    for i in range(5):
        
        res = jasp_program(i + 0.5)
        
        assert res == i + 4.5
    
    @jaspify
    def main():

        params = jnp.array([1.5, -0.5])
        
        rng = random.PRNGKey(4)

        def body_fun(k, val):
            rng, res = val
            rng, rng_input = random.split(rng)
            delta = random.choice(rng_input, jnp.array([1, -1]), shape=(2,))
            res += delta
            return rng, res
        
        rng_, res = fori_loop(0,10,body_fun,(rng, params))

        return res

    main()
    
    @jaspify
    def main():

        params = jnp.array([1.5, -0.5, 0.4, 0.2])
        a = 1

        rng = random.PRNGKey(4)

        def body_fun(k, val):
            rng, res, a = val
            rng, rng_input = random.split(rng)
            delta = random.choice(rng_input, jnp.array([1, -1]), shape=(4,))
            res += delta
            return rng, res, a
        
        rng_, res, a = fori_loop(0,10,body_fun,(rng, params, a))

        return res

    main()
    
    # Test https://github.com/eclipse-qrisp/Qrisp/issues/173
    
    @jaspify
    def main():

        def case0(x):
            return x + 1

        def case1(x):
            return x + 2

        def case2(x):
            return x + 3
        
        def case3(x):
            return x + 4

        def compute(index, x):
            return switch(index, [case0, case1, case2, case3], x)


        qf = QuantumFloat(2)
        qf[:] = 3
        ind = jnp.int8(measure(qf))

        res = compute(ind,jnp.int32(0))

        return ind, res
    
    assert main() == (3,4)

    # Test scan primitive
    
    @jaspify
    def test_scan_basic():
        xs = jnp.array([1, 2, 3])
        def body(c, x):
            return c + x, x * c
        # init=10
        # iter 1: x=1, c=10 -> c=11, y=10
        # iter 2: x=2, c=11 -> c=13, y=22
        # iter 3: x=3, c=13 -> c=16, y=39
        last_c, ys = jax.lax.scan(body, 10, xs)
        return last_c, ys

    c, ys = test_scan_basic()
    assert c == 16
    assert np.all(ys == np.array([10, 22, 39]))

    @jaspify
    def test_scan_reverse():
        xs = jnp.array([1, 2, 3])
        def body(c, x):
            return c + x, c
        
        # reverse=True -> scan over [3, 2, 1]
        # init=0
        # iter 1: x=3, c=0 -> c=3, y=0 (corresponds to input 3)
        # iter 2: x=2, c=3 -> c=5, y=3 (corresponds to input 2)
        # iter 3: x=1, c=5 -> c=6, y=5 (corresponds to input 1)
        # ys stacked in reverse (to match input order) -> [5, 3, 0]
        
        last_c, ys = jax.lax.scan(body, 0, xs, reverse=True)
        return last_c, ys

    c, ys = test_scan_reverse()
    assert c == 6
    assert np.all(ys == np.array([5, 3, 0]))

    @jaspify
    def test_scan_zero_length():
        xs = jnp.array([])
        def body(c, x):
            return c, x * 2
        
        _, ys = jax.lax.scan(body, 0, xs)
        return ys

    ys = test_scan_zero_length()
    assert ys.shape == (0,)

    @jaspify
    def test_scan_multi_input():
         # Test iterating over multiple arrays (like in lax.map(..., (a, b)))
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        
        def body(c, inputs):
            x, y = inputs
            return c, x * y
        
        _, out = jax.lax.scan(body, None, (a, b))
        return out
    
    out = test_scan_multi_input()
    assert np.all(out == np.array([4, 10, 18]))
