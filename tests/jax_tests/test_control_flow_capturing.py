# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests jrange loop capture into Jasp control-flow primitives."""

import jax.numpy as jnp

from qrisp import *


def test_jrange():

    def test_f(k):

        qv = QuantumFloat(k)
        h(qv[0])

        for i in jrange(k - 1):
            cx(qv[0], qv[i + 1])

        return measure(qv)

    jaspr = make_jaspr(test_f)(1)

    for i in range(1, 10):
        assert jaspr(i) in [0, 2**i - 1]

    ################
    @qache
    def int_encoder(qv, encoding_int):
        for i in jrange(qv.size):
            with control(encoding_int & (1 << i)):
                x(qv[i])

    def test_f(a, b):

        qv = QuantumFloat(a)

        int_encoder(qv, b + 1)

        return measure(qv)

    jaspr = make_jaspr(test_f)(1, 1)

    assert jaspr(3, 2) == 3
    assert jaspr(3, 3) == 4
    assert jaspr(3, 5) == 6

    #################

    # Test error messages

    @qache
    def int_encoder(qv, encoding_int):

        flag = True
        for i in jrange(qv.size):
            if flag:
                with control(encoding_int & (1 << i)):
                    x(qv[i])
            else:
                x(qv[0])
            flag = False

    def test_f(a, b):

        qv = QuantumFloat(a)

        int_encoder(qv, b + 1)

        return measure(qv)

    jaspr = make_jaspr(test_f, flatten_envs=False)(1, 1)

    try:
        jaspr(4, 5)
        exception_raised = False
    except:
        exception_raised = True

    assert exception_raised

    ###############

    @qache
    def int_encoder(qv, encoding_int):

        for i in jrange(qv.size):
            with control(encoding_int & (1 << i)):
                x(qv[i])
        return i

    def test_f(a, b):

        qv = QuantumFloat(a)

        int_encoder(qv, b + 1)

        return measure(qv)

    jaspr = make_jaspr(test_f, flatten_envs=False)(1, 1)

    try:
        jaspr(4, 5)
        exception_raised = False
    except:
        exception_raised = True

    assert exception_raised

    ######
    # Test multiple argument jrange

    def test_function(i):

        qv = QuantumFloat(i)
        x(qv[0])

        base_qb = qv[0]
        # Simulate step = 2
        for k in jrange((qv.size - 1) // 2):
            cx(base_qb, qv[2 * k + 2])

        return measure(qv)

    jaspr = make_jaspr(test_function)(100)

    assert jaspr(5) == 21

    def test_function(i):

        qv = QuantumFloat(i)
        x(qv[0])

        base_qb = qv[0]
        for i in jrange(1, qv.size - 1):
            cx(base_qb, qv[i + 1])

        return measure(qv)

    jaspr = make_jaspr(test_function)(100)

    assert jaspr(5) == 29

    ######
    # Test loop inversion

    def test_function():

        qv = QuantumFloat(5)
        x(qv[qv.size - 1])

        with invert():
            for i in jrange(1, qv.size):
                cx(qv[i], qv[i - 1])

        return measure(qv[0])

    jaspr = make_jaspr(test_function)()

    assert jaspr() == True

    def test_function():

        qv = QuantumFloat(5)
        x(qv[qv.size - 1])

        with invert():
            with invert():
                with invert():
                    for i in jrange(1, qv.size):
                        cx(qv[i], qv[i - 1])

        return measure(qv[0])

    jaspr = make_jaspr(test_function)()

    assert jaspr() == True

    # Test constant embedding
    @jaspify
    def main():

        qv = QuantumVariable(5)

        for i in jrange(5):
            a = jnp.array([0, 1, 2])
            x(qv[a[0] + i])

        return measure(qv)

    assert main() == 31
