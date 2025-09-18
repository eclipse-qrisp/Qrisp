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

import random
import pytest
import math
import jax.numpy as jnp
import numpy as np
from qrisp import *

CREATE_MEASURE_SEEDS = [0, 1, 2]
CREATE_MEASURE_SIZE = [2, 4, 8]

QC_INPLACE_SEEDS = [0, 1]
QC_INPLACE_SIZE = [1, 2]


# ----------------- Helpers -----------------

def mask_for_size(size: int) -> int:
    return (1 << (32 * size)) - 1


def _digits_array(maybe_bi):
    """
    Return the digits array whether input is a BigInteger or a raw JAX/Numpy array.
    """
    if isinstance(maybe_bi, (jnp.ndarray, np.ndarray)):
        return maybe_bi
    if hasattr(maybe_bi, "digits"):
        return maybe_bi.digits
    raise TypeError(f"Unexpected type for digits extraction: {type(maybe_bi)}")


def to_int(maybe_bi) -> int:
    """
    Convert a BigInteger (or its digits array) to a Python int exactly.
    """
    ds = np.asarray(_digits_array(maybe_bi), dtype=np.uint64).tolist()
    acc = 0
    for i, d in enumerate(ds):
        acc |= int(d) << (32 * i)
    return acc


def limbs_to_int(limbs) -> int:
    """
    Convert a list of uint32 limbs (little-endian) to a Python int.
    """
    acc = 0
    for i, d in enumerate(limbs):
        acc |= (int(d) & 0xFFFFFFFF) << (32 * i)
    return acc


def random_limbs(size):
    return [random.getrandbits(32) for _ in range(size)]


# ----------------- Tests -----------------

@pytest.mark.parametrize("size", CREATE_MEASURE_SIZE)
@pytest.mark.parametrize("seed", CREATE_MEASURE_SEEDS)
def test_modulus_biginteger_create_measure(seed, size):
    random.seed(seed)
    limbs_N = random_limbs(size)
    limbs_V = random_limbs(size)
    N_int = limbs_to_int(limbs_N)
    V_int = limbs_to_int(limbs_V) % N_int

    @boolean_simulation
    def main():
        N = BigInteger.create_static(N_int, size)
        V = BigInteger.create_static(V_int, size)
        qm = QuantumModulus(N)
        qm[:] = V
        return measure(qm)
    assert to_int(main()) == V_int


@pytest.mark.parametrize("size", QC_INPLACE_SIZE)
@pytest.mark.parametrize("seed", QC_INPLACE_SEEDS)
def test_modulus_biginteger_qc_inplace(seed, size):
    random.seed(seed)
    limbs_N = random_limbs(size)
    N_int = limbs_to_int(limbs_N)
    if N_int % 2 == 0:
        N_int += 1

    limbs_A = random_limbs(size)
    A_int = limbs_to_int(limbs_A) % N_int
    while True:
        limbs_B = random_limbs(size)
        B_int = limbs_to_int(limbs_B) % N_int
        if math.gcd(B_int, N_int) == 1:
            break

    @boolean_simulation
    def main():
        N = BigInteger.create_static(N_int, size+2)
        A = BigInteger.create_static(A_int, size+2)
        B = BigInteger.create_static(B_int, size+2)
        qm = QuantumModulus(N)
        qm[:] = A
        qm *= B
        return measure(qm)
    
    assert to_int(main()) == (A_int * B_int) % N_int


def test_modulus_create_measure():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        return measure(a)

    assert test() == 7

    @jaspify
    def test_m():
        a = QuantumModulus(13)
        a.m = 4
        a[:] = 7
        return measure(a)

    assert test_m() == 7


def test_modulus_qc_inplace_multiply():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a[:] = 7
        a *= 11
        return measure(a)

    assert test() == (7*11) % 13


def test_modulus_qc_multiply():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test_l():
        a = QuantumModulus(13)
        a[:] = 7
        b = a * 3
        return measure(a), measure(b)

    assert test_l() == (7, (7*3) % 13)

    @jaspify
    def test_r():
        a = QuantumModulus(13)
        a[:] = 7
        b = 3 * a
        return measure(a), measure(b)

    assert test_r() == (7, (7*3) % 13)


def test_modulus_qq_multiply():
    from qrisp import QuantumModulus, jaspify, measure

    @jaspify
    def test():
        a = QuantumModulus(13)
        a.m = 4
        a[:] = 7
        b = QuantumModulus(13)
        b.m = 4
        b[:] = 12
        c = a * b
        return measure(a), measure(b), measure(c)

    assert test() == (7, 12, (7*12) % 13)
