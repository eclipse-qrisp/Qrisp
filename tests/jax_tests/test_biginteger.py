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

def test_biginteger_from_float():
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    b_fl = 5.31241224e25

    @jaspify
    def test():
        b = BigInteger.from_float(b_fl, size=16)
        return (b)()

    assert (b_fl == test())

def test_biginteger_from_int():
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    b_int = 532_323_451_122_411

    @jaspify
    def test():
        b = BigInteger.from_float(b_int, size=16)
        return (b)()

    assert ((float(b_int)) == test())

def test_biginteger_mul():
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    a_fl = 1.34343424e9
    b_fl = 5.31241224e25

    @jaspify
    def test():
        a = BigInteger.from_float(a_fl, size=16)
        b = BigInteger.from_float(b_fl, size=16)
        return (b * a)()

    assert(b_fl*a_fl == test())

def test_biginteger_add():
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    a_fl = 1.34343424e24
    b_fl = 5.31241224e25

    @jaspify
    def test():
        a = BigInteger.from_float(a_fl, size=16)
        b = BigInteger.from_float(b_fl, size=16)
        return (b + a)()

    assert(b_fl+a_fl == test())

def test_biginteger_sub():
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    a_fl = 1.34343424e124
    b_fl = 5.31241224e125

    @jaspify
    def test():
        a = BigInteger.from_float(a_fl, size=16)
        b = BigInteger.from_float(b_fl, size=16)
        return (b - a)()

    assert(b_fl-a_fl == test())

def test_biginteger_mod():
    import jax
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    a_fl = 1.34343424e124
    b_fl = 5.31241224e125

    #@jaspify #Does not work with jaspify
    #def test():
    #    a = BigInteger.from_float(a_fl, size=16)
    #    b = BigInteger.from_float(b_fl, size=16)
    #    return (b % a)()

    @jax.jit
    def inner():
       a = BigInteger.from_float(a_fl, size=16)
       b = BigInteger.from_float(b_fl, size=16)
       return b % a
    
    @jaspify
    def test():
        return inner()()

    assert(b_fl%a_fl == test())

def test_biginteger_floordiv():
    import jax
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger

    a_fl = 112_312
    b_fl = 531_123

    #@jaspify #Does not work with jaspify
    #def test():
    #    a = BigInteger.from_float(a_fl, size=16)
    #    b = BigInteger.from_float(b_fl, size=16)
    #    return (b // a)()
    
    @jax.jit
    def inner():
       a = BigInteger.from_float(a_fl, size=16)
       b = BigInteger.from_float(b_fl, size=16)
       return b // a
    
    @jaspify
    def test():
        return inner()()

    assert(b_fl // a_fl == test())

def test_biginteger_modinv():
    import jax
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger, bi_modinv
    from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv

    a_int = 2_321
    b_int = 3_319

    #@jaspify #Does not work with jaspify
    #def test():
    #    a = BigInteger.from_float(a_int, size=16)
    #    b = BigInteger.from_float(b_int, size=16)
    #    q = bi_modinv(a, b)
    #    return (q)()
    
    @jax.jit
    def inner():
       a = BigInteger.from_float(a_int, size=16)
       b = BigInteger.from_float(b_int, size=16)
       return bi_modinv(a, b)
    
    @jaspify
    def test():
        return inner()()

    assert(float(modinv(a_int, b_int)) == test())

def test_biginteger_lshift():
    import jax
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger, bi_modinv
    from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv

    a_int = 2_312_321
    shift = 71
    
    @jax.jit
    def inner():
       a = BigInteger.from_int(a_int, size=16)
       return a << shift
    
    @jaspify
    def test():
        return inner()()

    assert(test() == float(a_int << shift))
    
def test_biginteger_rshift():
    import jax
    from qrisp import jaspify
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic import BigInteger, bi_modinv
    from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import modinv

    a_int = 2_312_321
    
    @jax.jit
    def inner():
       a = BigInteger.from_int(a_int, size=16)
       return (a << 112) >> 42
    
    @jaspify
    def test():
        return inner()()

    assert(test() == float((a_int << 112) >> 42))