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

"""Tests for the classical helpers and quantum order-finding in Shor's algorithm and RSA."""

from fractions import Fraction

import numpy as np

from qrisp.algorithms.shor.crypto_tools import (
    _bitstring_to_string,
    rsa_decrypt,
    rsa_decrypt_string,
    rsa_encrypt,
    rsa_encrypt_string,
)
from qrisp.algorithms.shor.shors_algorithm import (
    _extract_order,
    _find_optimal_a,
    _get_r_values,
    shors_alg,
)

# ----------------- Classical helpers (no quantum simulation) -----------------


def test_find_optimal_a_returns_coprime_candidates():
    """`_find_optimal_a` must only propose bases coprime to the modulus."""
    n = 15
    proposals = _find_optimal_a(n)
    assert proposals
    assert all(np.gcd(a, n) == 1 for a in proposals)


def test_get_r_values_recovers_denominator():
    """`_get_r_values` must recover the exact denominator of a terminating fraction."""
    # 0.25 = 1/4 exactly, so the only nontrivial convergent denominator is 4
    assert _get_r_values(0.25) == [4]
    assert _get_r_values(0.5) == [2]


def test_extract_order_from_synthetic_measurement():
    """`_extract_order` must recover the order from a synthetic phase-estimation outcome."""
    # 2**4 = 16 = 1 mod 15, so a phase-estimation outcome concentrated on 1/4
    # (and its complement 3/4) should yield order r = 4.
    expected_order = 4
    mes_res = {0.25: 0.5, 0.75: 0.5}
    assert _extract_order(mes_res, 2, 15) == expected_order  # type: ignore[arg-type]


def test_extract_order_rejects_wrong_candidate_before_true_order():
    """`_extract_order` must verify candidates, not accept the first one blindly.

    Regression test: `_find_order` used to pass its repeatedly-squared loop
    variable (not the original base) into `_extract_order`, which can make
    this verification vacuous (e.g. base**(2**n) == 1) and silently accept
    an unverified, possibly-wrong order.
    """
    # a=2, N=15: true order is 4. The synthetic outcome list puts a wrong
    # candidate (order 2) first; 2**2 % 15 != 1, so it must be rejected in
    # favor of the correct order 4 recovered from the second outcome.
    # (Both phases are exact powers of two, like real QPE measurements, so
    # this doesn't hit sympy's float-to-Rational precision artifacts that a
    # non-power-of-two phase like 1/3 would.)
    expected_order = 4
    mes_res = {0.5: 0.5, 0.25: 0.5}
    assert _extract_order(mes_res, 2, 15) == expected_order  # type: ignore[arg-type]


def test_extract_order_combines_three_outcomes():
    """`_extract_order` must find an order that only emerges from combining three outcomes.

    Regression test for a real gap in the pairwise-only combination fallback:
    for order 30 (= lcm(2, 3, 5)), no pairwise combination of 2, 3, and 5
    divides 30, so a fallback that only ever combines two outcomes at a time
    would exhaust its candidates and never find it. `a=99, N=181` is a real
    (non-synthetic) pair with true order exactly 30.
    """
    # Exact Fraction phases (not floats) to avoid sympy's float-to-Rational
    # precision artifacts for non-power-of-two denominators (see the test
    # above for the same reasoning with 1/3).
    expected_order = 30
    mes_res = {Fraction(1, 2): 1 / 3, Fraction(1, 3): 1 / 3, Fraction(1, 5): 1 / 3}
    assert _extract_order(mes_res, 99, 181) == expected_order  # type: ignore[arg-type]


def test_bitstring_to_string_decodes_7bit_chars():
    """`_bitstring_to_string` must decode a 7-bit-per-character bitstring."""
    bitstring = "".join(format(ord(c), "b").zfill(7) for c in "Hi")
    assert _bitstring_to_string(bitstring) == "Hi"


def test_rsa_encrypt_is_modular_exponentiation():
    """`rsa_encrypt` must compute plain modular exponentiation."""
    e, n, message = 7, 33, 8
    assert rsa_encrypt(e, n, message) == pow(message, e, n)


# ----------------- Quantum (Shor's algorithm) coverage -----------------


def test_shors_alg_factors_small_composite():
    """`shors_alg` must find a nontrivial factor of a small composite number."""
    assert shors_alg(15) in (3, 5)


def test_rsa_decrypt_roundtrip_via_shors_alg():
    """`rsa_decrypt` must recover the original message via Shor's-algorithm factorization."""
    n, e, message = 15, 3, 7
    ciphertext = rsa_encrypt(e, n, message)
    assert rsa_decrypt(e, n, ciphertext) == message


def test_rsa_encrypt_decrypt_string_roundtrip():
    """`rsa_encrypt_string`/`rsa_decrypt_string` must round-trip an arbitrary short string."""
    n, e, text = 65, 7, "Hi!"
    ciphertext = rsa_encrypt_string(e, n, text)
    assert rsa_decrypt_string(e, n, ciphertext) == text
