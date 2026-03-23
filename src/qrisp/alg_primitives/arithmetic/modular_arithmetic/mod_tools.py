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

from qrisp import check_for_tracing_mode
import jax.numpy as jnp
import jax.lax as lax


def montgomery_decoder(y, R, N):
    if 0 < R < 1:
        R = modinv(R**-1, N)
    return (y * modinv(R, N)) % N


def montgomery_encoder(y, R, N):
    if 0 < R < 1:
        R = modinv(R**-1, N)
    return (int(y) % N * int(R) % N) % N


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a, m):
    if check_for_tracing_mode():

        def cf(val):
            t, new_t, r, new_r = val
            return new_r != 0

        def bf(val):
            t, new_t, r, new_r = val
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r - quotient * new_r
            return t, new_t, r, new_r

        t, new_t, r, new_r = lax.while_loop(cf, bf, (0, 1, m, a))

        # Ensure result is in [0, MOD)
        return jnp.where(t < 0, t + m, t)
    else:
        g, x, y = egcd(a, m)
        if g != 1:
            raise Exception("modular inverse does not exist")
        else:
            return x % m
