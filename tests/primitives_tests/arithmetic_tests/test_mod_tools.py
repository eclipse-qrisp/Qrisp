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

"""Tests for the static (non-Jasp) modular-arithmetic helpers in modular_arithmetic/mod_tools.py."""

import math

import jax
import pytest

from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import (
    egcd,
    modinv,
    montgomery_decoder,
    montgomery_encoder,
)


def test_egcd_bezout_identity():
    """`egcd` must return the gcd and Bézout coefficients satisfying a*x + b*y = gcd(a, b)."""
    for a, b in [(35, 15), (240, 46), (17, 5), (100, 1)]:
        g, x, y = egcd(a, b)
        assert g == math.gcd(a, b)
        assert a * x + b * y == g


def test_modinv_static():
    """`modinv` must return the modular inverse, and raise if a and m are not coprime."""
    t = modinv(3, 11)
    assert (3 * t) % 11 == 1
    with pytest.raises(ValueError):
        modinv(2, 4)  # gcd(2, 4) != 1, no inverse exists


def test_modinv_traced_matches_static():
    """`modinv` must give the same result whether called eagerly or under jax.jit tracing."""
    traced = jax.jit(modinv)
    assert int(traced(3, 11)) == modinv(3, 11)


def test_montgomery_encoder_decoder_roundtrip():
    """`montgomery_encoder`/`montgomery_decoder` must round-trip a value under a given radix."""
    radix, modulus, x = 1024, 97, 42
    encoded = montgomery_encoder(x, radix, modulus)
    assert encoded == (x * radix) % modulus
    assert montgomery_decoder(encoded, radix, modulus) == x
