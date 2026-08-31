"""********************************************************************************
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


def test_montgomery_jasp_qq():
    import numpy as np

    from qrisp import QuantumFloat, best_montgomery_shift, boolean_simulation, gidney_adder, measure, modinv
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import qq_montgomery_multiply

    @boolean_simulation
    def qq(a, b, n, N):
        qa = QuantumFloat(n)
        qa[:] = a
        qb = QuantumFloat(n)
        qb[:] = b
        m = best_montgomery_shift(N)
        res = qq_montgomery_multiply(qa, qb, N, m, gidney_adder)
        return measure(qa), measure(qb), measure(res)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        q = modinv(2**n, N)
        for a in range(1, 50, 3):
            for b in range(1, 50, 5):
                if a % N != 0 and b % N != 0:
                    ar, br, rr = qq(a % N, b % N, n, N)
                    assert ar == a % N
                    assert br == b % N
                    assert rr == (ar * br * q) % N


def test_montgomery_not_jasp_qq():
    from qrisp import QuantumFloat, best_montgomery_shift, gidney_adder, modinv, multi_measurement
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import qq_montgomery_multiply

    X = 29
    y = 21
    N = 31
    n = 5

    def test_qq_g():
        qx = QuantumFloat(n)
        qx[:] = X
        qy = QuantumFloat(n)
        qy[:] = y
        m = best_montgomery_shift(N)
        res = qq_montgomery_multiply(qx, qy, N, m, gidney_adder)
        return multi_measurement([res])

    m = best_montgomery_shift(N)

    assert test_qq_g()[((X * y * modinv(2**m, N)) % N,)] == 1.0


def test_montgomery_jasp_cq():
    import numpy as np

    from qrisp import QuantumFloat, best_montgomery_shift, boolean_simulation, gidney_adder, measure
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply

    @boolean_simulation
    def cq(a, b, n, N):
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(a)
        res = cq_montgomery_multiply(a, qb, N, shift, gidney_adder)
        return measure(qb), measure(res)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                if a % N != 0 and b % N != 0:
                    br, rr = cq(a % N, b % N, n, N)
                    assert br == b % N
                    assert rr == ((a % N) * br) % N


def test_montgomery_jasp_cq_inplace():
    import numpy as np

    from qrisp import QuantumFloat, best_montgomery_shift, boolean_simulation, gidney_adder, measure
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import modinv
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace

    @boolean_simulation
    def icq(a, b, n, N):
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(a)
        cq_montgomery_multiply_inplace(a, qb, N, shift, gidney_adder)
        return measure(qb)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        q = modinv(2**n, N)
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                if a % N != 0 and b % N != 0 and np.gcd(a, N) == 1:
                    br = icq(a % N, b % N, n, N)
                    assert br == ((a % N) * (b % N)) % N


def test_montgomery_jasp_cq_inplace_controlled():
    import numpy as np

    from qrisp import (
        QuantumBool,
        QuantumFloat,
        best_montgomery_shift,
        boolean_simulation,
        control,
        gidney_adder,
        measure,
    )
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import modinv
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace

    @boolean_simulation
    def cicq(a, b, n, N, c):
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(a)
        qc = QuantumBool()
        qc[:] = c
        with control(qc[0]):
            cq_montgomery_multiply_inplace(a, qb, N, shift, gidney_adder)
        return measure(qb)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        q = modinv(2**n, N)
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                for c in [0, 1]:
                    if a % N != 0 and b % N != 0 and np.gcd(a, N) == 1:
                        br = cicq(a % N, b % N, n, N, c)
                        assert br == (((a % N) ** c) * (b % N)) % N


def test_montgomery_jasp_cq_inplace_bi():
    import numpy as np

    from qrisp import (
        BigInteger,
        QuantumBool,
        QuantumFloat,
        best_montgomery_shift,
        boolean_simulation,
        control,
        gidney_adder,
        measure,
    )
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace

    @boolean_simulation
    def bicicq(a, b, n, N, c):
        a = BigInteger.create(a, 3)
        N = BigInteger.create(N, 3)
        qb = QuantumFloat(n)
        qb[:] = b
        shift = best_montgomery_shift(a)
        qc = QuantumBool()
        qc[:] = c
        with control(qc[0]):
            cq_montgomery_multiply_inplace(a, qb, N, shift, gidney_adder)
        return measure(qb)

    for N in range(11, 50, 8):
        n = int(np.ceil(np.log2(N)))
        for a in range(4, 50, 3):
            for b in range(4, 50, 5):
                for c in [0, 1]:
                    if a % N != 0 and b % N != 0 and np.gcd(a, N) == 1:
                        br = bicicq(a % N, b % N, n, N, c)
                        assert br == (((a % N) ** c) * (b % N)) % N


def test_montgomery_find_order():
    import numpy as np

    from qrisp import (
        QFT,
        BigInteger,
        QuantumFloat,
        QuantumModulus,
        control,
        fourier_adder,
        gidney_adder,
        h,
        jasp_fourier_adder,
        jrange,
        terminal_sampling,
        x,
    )

    def find_order(a, N, inpl_adder):
        qg = QuantumModulus(N, inpl_adder)
        qg[:] = 1
        qpe_res = QuantumFloat(2 * qg.size + 1, exponent=-(2 * qg.size + 1))
        h(qpe_res)
        for i in range(len(qpe_res)):
            with control(qpe_res[i]):
                qg *= a
                a = (a * a) % N
        QFT(qpe_res, inv=True)
        return qpe_res.get_measurement()

    dict_norm_fourier = find_order(4, 13, fourier_adder)
    dict_norm_gidney = find_order(4, 13, gidney_adder)

    @terminal_sampling
    def find_order(a, N, inpl_adder):
        qg = QuantumModulus(N, inpl_adder=inpl_adder)
        x(qg[0])
        qpe_res = QuantumFloat(2 * qg.size + 1, exponent=-(2 * qg.size + 1))
        h(qpe_res)
        for i in jrange(qpe_res.size):
            with control(qpe_res[i]):
                qg *= a
            a = (a * a) % N
        QFT(qpe_res, inv=True)
        return qpe_res

    dict_jasp_fourier = find_order(4, 13, jasp_fourier_adder)
    dict_jasp_gidney = find_order(4, 13, gidney_adder)

    dict_bim_fourier = find_order(BigInteger.create(4, 1), BigInteger.create(13, 1), jasp_fourier_adder)
    dict_bim_gidney = find_order(BigInteger.create(4, 1), BigInteger.create(13, 1), gidney_adder)

    def check_dict_equality(a, b):
        for key in a.keys():
            if not np.allclose(a[key], b.get(key, -1), rtol=0.001, atol=0.001):
                return False
        return True

    assert check_dict_equality(dict_norm_fourier, dict_norm_gidney)
    assert check_dict_equality(dict_jasp_fourier, dict_jasp_gidney)
    assert check_dict_equality(dict_bim_fourier, dict_bim_gidney)

    assert check_dict_equality(dict_norm_gidney, dict_jasp_gidney)
    assert check_dict_equality(dict_jasp_gidney, dict_bim_gidney)


def test_egcd_bezout_identity():
    """`egcd` must return the gcd and Bézout coefficients satisfying a*x + b*y = gcd(a, b)."""
    import math

    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import egcd

    for a, b in [(35, 15), (240, 46), (17, 5), (100, 1)]:
        g, x, y = egcd(a, b)
        g, x, y = int(g), int(x), int(y)
        assert g == math.gcd(a, b)
        assert a * x + b * y == g


def test_bi_pow2mod_matches_python_pow():
    """`bi_pow2mod` must compute 2**exp mod m as a BigInteger, matching Python's pow."""
    from qrisp import BigInteger
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import bi_pow2mod

    for exp, mod in [(10, 97), (0, 97), (1, 3), (37, 1009)]:
        mod_bi = BigInteger.create_static(mod, 4)
        result = bi_pow2mod(exp, mod_bi)
        assert result() == pow(2, exp, mod)


def test_pow2_mod_n_traced_matches_python_pow():
    """`pow2_mod_N` must compute 2**exp mod N under jax.jit tracing, matching Python's pow."""
    import jax

    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import pow2_mod_N

    traced = jax.jit(pow2_mod_N)
    for exp, mod in [(10, 97), (0, 97), (37, 1009)]:
        assert int(traced(exp, mod)) == pow(2, exp, mod)


def test_montgomery_encoder_decoder_mixed_bigint_roundtrip():
    """`montgomery_encoder`/`montgomery_decoder` round-trip a BigInteger with plain-int args."""
    from qrisp import BigInteger
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
        montgomery_decoder,
        montgomery_encoder,
    )

    radix, modulus, x = 1024, 97, 42
    x_bi = BigInteger.create(x, 4)
    encoded = montgomery_encoder(x_bi, radix, modulus)
    assert isinstance(encoded, BigInteger)
    assert encoded() == (x * radix) % modulus
    decoded = montgomery_decoder(encoded, radix, modulus)
    assert isinstance(decoded, BigInteger)
    assert decoded() == x


def test_new_montgomery_decoder_positive_and_negative_shift():
    """`new_montgomery_decoder` must decode both positive (inverse) and non-positive shifts."""
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
        new_montgomery_decoder,
    )

    modulus, x, m = 97, 55, 10
    encoded = (pow(2, m, modulus) * x) % modulus
    assert new_montgomery_decoder(encoded, m, modulus) == x
    # Non-positive shift decodes with 2**abs(m) mod N instead of its inverse
    assert new_montgomery_decoder(x, -3, modulus) == (x * pow(2, 3, modulus)) % modulus


def test_qq_montgomery_multiply_modulus():
    """`qq_montgomery_multiply_modulus` must compute the montgomery product of two QuantumModuli."""
    from qrisp import QuantumModulus, gidney_adder, multi_measurement
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import (
        qq_montgomery_multiply_modulus,
    )

    qx = QuantumModulus(97, inpl_adder=gidney_adder)
    qy = QuantumModulus(97, inpl_adder=gidney_adder)
    qx[:] = 12
    qy[:] = 7
    res = qq_montgomery_multiply_modulus(qx, qy)
    assert multi_measurement([qx, qy, res]) == {(12, 7, 84): 1.0}


def test_cq_montgomery_mat_multiply():
    """Regression test for `QuantumArray @ np.ndarray`.

    This previously crashed for any standard numpy integer matrix, because
    entries indexed out as numpy.int64 were rejected by
    smallest_power_of_two's int type check.
    """
    import numpy as np

    from qrisp import QuantumArray, QuantumModulus, gidney_adder, multi_measurement

    modulus = 7
    a_array = QuantumArray(qtype=QuantumModulus(modulus, inpl_adder=gidney_adder), shape=(2, 2))
    a_array[:] = np.array([[1, 2], [3, 4]])
    b_array = np.array([[1, 2], [3, 4]])
    r_array = a_array @ b_array
    (outcome,) = list(multi_measurement([r_array]).keys())
    assert outcome[0].tolist() == [[0, 3], [1, 1]]
