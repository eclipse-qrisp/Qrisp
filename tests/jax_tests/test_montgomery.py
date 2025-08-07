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
    from qrisp import jaspify, QuantumFloat, modinv, jasp_fourier_adder, gidney_adder, measure
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import qq_montgomery_multiply, compute_aux_radix_exponent

    X = 29
    y = 21
    N = 31
    n = 5

    @jaspify
    def test_qq_ft():
        qx = QuantumFloat(n)
        qx[:] = X
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
        res = qq_montgomery_multiply(qx, qy, N, m, jasp_fourier_adder)
        return measure(res)
    
    @jaspify
    def test_qq_g():
        qx = QuantumFloat(n)
        qx[:] = X
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
        res = qq_montgomery_multiply(qx, qy, N, m, gidney_adder)
        return measure(res)
    
    m = compute_aux_radix_exponent(N, n)
    
    assert test_qq_ft() == (X*y*modinv(2**m, N))%N
    assert test_qq_g() == (X*y*modinv(2**m, N))%N

def test_montgomery_not_jasp_qq():
    from qrisp import QuantumFloat, modinv, gidney_adder, multi_measurement
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import qq_montgomery_multiply, compute_aux_radix_exponent

    X = 29
    y = 21
    N = 31
    n = 5
    
    def test_qq_g():
        qx = QuantumFloat(n)
        qx[:] = X
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
        res = qq_montgomery_multiply(qx, qy, N, m, gidney_adder)
        return multi_measurement([res])
    
    m = compute_aux_radix_exponent(N, n)
    
    assert test_qq_g()[((X*y*modinv(2**m, N))%N,)] == 1.0


def test_montgomery_jasp_cq():
    from qrisp import jaspify, QuantumFloat, modinv, jasp_fourier_adder, gidney_adder, measure
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply, compute_aux_radix_exponent

    X = 29
    y = 21
    N = 31
    n = 10

    @jaspify
    def test_cq_ft():
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, qy.size)
        res = cq_montgomery_multiply(X, qy, N, m, jasp_fourier_adder)
        return measure(res)
    
    @jaspify
    def test_cq_g():
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
        res = cq_montgomery_multiply(X, qy, N, m, gidney_adder)
        return measure(res)
    
    m = compute_aux_radix_exponent(N, n)
    
    assert test_cq_ft() == (X*y*modinv(2**m, N))%N
    assert test_cq_g() == (X*y*modinv(2**m, N))%N


def test_montgomery_jasp_cq_inplace():
    from qrisp import jaspify, QuantumFloat, modinv, jasp_fourier_adder, gidney_adder, measure
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace, compute_aux_radix_exponent

    X = 29
    y = 21
    N = 31
    n = 5

    @jaspify
    def test_cq_ft():
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, qy.size)
        cq_montgomery_multiply_inplace(X, qy, N, m, jasp_fourier_adder)
        return measure(qy)
    
    @jaspify
    def test_cq_g():
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
        cq_montgomery_multiply_inplace(X, qy, N, m, gidney_adder)
        return measure(qy)
    
    m = compute_aux_radix_exponent(N, n)
    
    assert test_cq_ft() == (X*y*modinv(2**m, N))%N
    assert test_cq_g() == (X*y*modinv(2**m, N))%N

def test_montgomery_jasp_cq_inplace_bi():
    from qrisp import jaspify, QuantumFloat, modinv, jasp_fourier_adder, gidney_adder, measure, BigInteger
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace, compute_aux_radix_exponent
    import jax.numpy as jnp

    X_int = 29
    y = 21
    N_int = 31
    n = 5
    
    @jaspify
    def test_cq_g():
        qy = QuantumFloat(n)
        qy[:] = y
        X = BigInteger.from_int(X_int, 1)
        N = BigInteger.from_int(N_int, 1)
        m = (jnp.ceil(jnp.log2(N()))).astype(jnp.int64)
        cq_montgomery_multiply_inplace(X, qy, N, m, gidney_adder)
        return measure(qy)
    
    m = compute_aux_radix_exponent(N_int, n)
    
    assert test_cq_g() == (X_int*y*(2**m%N_int))%N_int

def test_montgomery_find_order():
    import numpy as np
    import jax
    from qrisp import terminal_sampling, QuantumModulus, QuantumFloat, jrange, control, jasp_fourier_adder, QFT, h, x, fourier_adder, BigInteger
    
    def find_order(a, N):
        qg = QuantumModulus(N, inpl_adder=fourier_adder)
        qg[:] = 1
        qpe_res = QuantumFloat(2*qg.size + 1, exponent = -(2*qg.size + 1))
        h(qpe_res)
        for i in range(len(qpe_res)):
            with control(qpe_res[i]):
                qg *= a
                a = (a*a)%N
        QFT(qpe_res, inv = True)
        return qpe_res.get_measurement()

    dict_norm = find_order(4, 13)
    
    
    @terminal_sampling
    def find_order(a, N):
        qg = QuantumModulus(N, inpl_adder=jasp_fourier_adder)
        qg[:] = 1
        qpe_res = QuantumFloat(2*qg.size + 1, exponent = -(2*qg.size + 1))
        h(qpe_res)
        for i in jrange(qpe_res.size):
            with control(qpe_res[i]):
                qg *= a
            a = (a*a)%N
        QFT(qpe_res, inv = True)
        return qpe_res

    dict_jasp = find_order(4, 13)

    @terminal_sampling
    def find_order(a, N):
        qg = QuantumModulus(N, inpl_adder=jasp_fourier_adder)
        x(qg[0])
        qpe_res = QuantumFloat(2*qg.size + 1, exponent = -(2*qg.size + 1))
        h(qpe_res)
        for i in jrange(qpe_res.size):
            with control(qpe_res[i]):
                qg *= a
            a = (a*a)%N
        QFT(qpe_res, inv = True)
        return qpe_res

    dict_bim = find_order(BigInteger.from_int_python(4, 1), BigInteger.from_int(13, 1))


    def check_dict_equality(a, b):
        for key in a.keys():
            if not np.allclose(a[key], b.get(key, -1), rtol=0.001, atol=0.001):
                return False
        return True

    assert check_dict_equality(dict_jasp, dict_norm)
    assert check_dict_equality(dict_jasp, dict_bim)