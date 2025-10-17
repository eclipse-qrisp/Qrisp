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


def test_montgomery_jasp_qq():
    import numpy as np
    from qrisp import boolean_simulation, QuantumFloat, modinv, gidney_adder, measure, best_montgomery_shift
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
                    assert rr == (ar*br*q) % N


def test_montgomery_not_jasp_qq():
    from qrisp import QuantumFloat, modinv, gidney_adder, multi_measurement, best_montgomery_shift
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

    assert test_qq_g()[((X*y*modinv(2**m, N)) % N,)] == 1.0


def test_montgomery_jasp_cq():
    import numpy as np
    from qrisp import boolean_simulation, QuantumFloat, gidney_adder, measure, best_montgomery_shift
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
                    assert rr == ((a % N)*br) % N


def test_montgomery_jasp_cq_inplace():
    import numpy as np
    from qrisp import boolean_simulation, QuantumFloat, gidney_adder, measure, best_montgomery_shift
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import modinv

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
                    assert br == ((a % N)*(b % N)) % N


def test_montgomery_jasp_cq_inplace_controlled():
    import numpy as np
    from qrisp import boolean_simulation, QuantumFloat, gidney_adder, measure, QuantumBool, control, best_montgomery_shift
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import modinv

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
                        assert br == (((a % N)**c)*(b % N)) % N


def test_montgomery_jasp_cq_inplace_bi():
    import numpy as np
    from qrisp import boolean_simulation, QuantumFloat, QuantumBool, gidney_adder, measure, control, BigInteger, best_montgomery_shift
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
                        assert br == (((a % N)**c)*(b % N)) % N


def test_montgomery_find_order():
    import numpy as np
    import jax
    from qrisp import terminal_sampling, QuantumModulus, QuantumFloat, jrange, control, QFT, h, x, BigInteger
    from qrisp import fourier_adder, jasp_fourier_adder, gidney_adder, jasp_cq_gidney_adder

    def find_order(a, N, inpl_adder):
        qg = QuantumModulus(N, inpl_adder)
        qg[:] = 1
        qpe_res = QuantumFloat(2*qg.size + 1, exponent=-(2*qg.size + 1))
        h(qpe_res)
        for i in range(len(qpe_res)):
            with control(qpe_res[i]):
                qg *= a
                a = (a*a) % N
        QFT(qpe_res, inv=True)
        return qpe_res.get_measurement()

    dict_norm_fourier = find_order(4, 13, fourier_adder)
    dict_norm_gidney = find_order(4, 13, gidney_adder)

    @terminal_sampling
    def find_order(a, N, inpl_adder):
        qg = QuantumModulus(N, inpl_adder=inpl_adder)
        x(qg[0])
        qpe_res = QuantumFloat(2*qg.size + 1, exponent=-(2*qg.size + 1))
        h(qpe_res)
        for i in jrange(qpe_res.size):
            with control(qpe_res[i]):
                qg *= a
            a = (a*a) % N
        QFT(qpe_res, inv=True)
        return qpe_res

    dict_jasp_fourier = find_order(4, 13, jasp_fourier_adder)
    dict_jasp_gidney = find_order(4, 13, jasp_cq_gidney_adder)

    dict_bim_fourier = find_order(BigInteger.create(4, 1), BigInteger.create(13, 1), jasp_fourier_adder)
    dict_bim_gidney = find_order(BigInteger.create(4, 1), BigInteger.create(13, 1), jasp_cq_gidney_adder)

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
