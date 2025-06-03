

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
    n = 5

    @jaspify
    def test_cq_ft():
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
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

def test_montgomery_not_jasp_cq():
    from qrisp import QuantumFloat, modinv, gidney_adder, multi_measurement
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply, compute_aux_radix_exponent

    X = 29
    y = 21
    N = 31
    n = 5
    
    def test_cq_g():
        qy = QuantumFloat(n)
        qy[:] = y
        m = compute_aux_radix_exponent(N, n)
        res = cq_montgomery_multiply(X, qy, N, m, gidney_adder)
        return multi_measurement([res])
    
    m = compute_aux_radix_exponent(N, n)
    
    assert test_cq_g()[((X*y*modinv(2**m, N))%N,)] == 1.0