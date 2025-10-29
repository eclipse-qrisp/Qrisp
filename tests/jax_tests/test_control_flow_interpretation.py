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

from jax import random
from jax.lax import fori_loop, switch

from qrisp import *


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

        rng_, res = fori_loop(0, 10, body_fun, (rng, params))

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

        rng_, res, a = fori_loop(0, 10, body_fun, (rng, params, a))

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

        res = compute(ind, jnp.int32(0))

        return ind, res

    assert main() == (3, 4)
