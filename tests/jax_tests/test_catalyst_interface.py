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

import time
import itertools

from qrisp import *
from qrisp.jasp import *
from qrisp.vqe.problems.electronic_structure import *


def test_catalyst_interface():
    
    try:
        import catalyst
    except ModuleNotFoundError:
        return
    
    def test_fun(i):
        qv = QuantumFloat(i, -2)
        with invert():
            cx(qv[0], qv[qv.size-1])
            x(qv[0])
        meas_res = measure(qv)
        return meas_res + 3
    
    jaspr = make_jaspr(test_fun)(2)
    
    jaspr.to_qir()
    jaspr.to_mlir()
    jaspr.to_catalyst_jaxpr()
    
    assert jaspr.qjit(4) == 5.25
    
    
    def int_encoder(qv, encoding_int):
        for i in jrange(qv.size):
            with control(encoding_int & (1<<i)):
                x(qv[i])

    @qjit
    def test_f(a):
        time.sleep(10)        
        qv = QuantumFloat(4)
        int_encoder(qv, a)
        return measure(qv)

    t0 = time.time()

    # Test classical control flow    
    assert test_f(4) == 4
    assert test_f(5) == 5
    assert test_f(6) == 6
    assert test_f(7) == 7

    # Test QJIT caching
    assert time.time() - t0 < 20

    # Test U3 translation    
    def main(a, b, c, d):
        qv = QuantumFloat(1)
        with control(d == 0):
            h(qv[0])
        u3(a*np.pi, b*np.pi, c*np.pi, qv[0])
        with control(d == 0):
            h(qv[0])
        return measure(qv[0])
    
    for a,b,c,d in itertools.product(*4*[[0,1]]):
        assert qjit(main)(a,b,c,d) == jaspify(main)(a,b,c,d)
        
    for i in range(8):
        statevector_array = 8*[0]
        statevector_array[i] = 1
        def main():
            qv = QuantumFloat(3)
            prepare(qv, statevector_array)
            return measure(qv)
        assert jaspify(main)() == qjit(main)()
        
    ## Test fuse primitive
    
    @qjit
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a.reg + b.reg)

    assert main() == 63
        
    @qjit
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a.reg + b[0])

    assert main() == 15
    
    @qjit
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a[0] + b.reg)

    assert main() == 15
    
    @qjit
    def main():
        
        a = QuantumFloat(3)
        b = QuantumFloat(3)
        a[:] = 7
        b[:] = 7
        
        return measure(a[0] + b[0])

    assert main() == 3

    # Test for https://github.com/eclipse-qrisp/Qrisp/issues/180    
    from pyscf import gto
    @make_jaspr
    def main():
    
        mol = gto.M(
            atom = '''H 0 0 0; H 0 0 0.74''',
            basis = 'sto-3g')
    
        vqe = electronic_structure_problem(mol)
    
        energy = vqe.run(lambda : QuantumFloat(4), depth=1, max_iter=100, optimizer="SPSA")
    
        return energy
    
    jaspr = main()
    qir_str = jaspr.to_qir()

def test_parity_catalyst():
    """Test parity primitive with catalyst interface."""
    
    try:
        import catalyst
    except ModuleNotFoundError:
        return
    
    # Test basic parity
    @qjit
    def test_basic_parity():
        qv = QuantumVariable(3)
        x(qv[0])
        x(qv[2])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        m3 = measure(qv[2])
        
        result = parity(m1, m2, m3)
        return result
    
    # XOR of (True, False, True) = False
    assert test_basic_parity() == False
    
    # Test parity with expectation
    @qjit
    def test_parity_expectation():
        qv = QuantumVariable(2)
        x(qv[0])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        
        # Parity is True (one 1), expectation is False
        # XOR(True, False) = True (mismatch indicator)
        result = parity(m1, m2, expectation=False)
        return result
    
    assert test_parity_expectation() == True
    
    # Test parity returns correct type that can be used in subsequent operations
    @qjit
    def test_parity_type():
        qv = QuantumVariable(2)
        x(qv[0])
        
        m1 = measure(qv[0])
        m2 = measure(qv[1])
        
        p = parity(m1, m2)
        
        # Parity result should be usable in boolean operations
        return p
    
    # Parity of (True, False) = True
    assert test_parity_type() == True


def test_parity_catalyst_with_scan():
    """Test parity with array inputs (scan primitive) in catalyst."""
    
    try:
        import catalyst
    except ModuleNotFoundError:
        return
    
    import jax.numpy as jnp
    
    @qjit
    def test_array_parity():
        qv0 = QuantumVariable(3)
        qv1 = QuantumVariable(3)
        
        # Set specific states
        x(qv0[0])  # qv0 = [1, 0, 1]
        x(qv0[2])
        x(qv1[1])  # qv1 = [0, 1, 0]
        
        # Measure individual qubits
        m0_0 = measure(qv0[0])
        m0_1 = measure(qv0[1])
        m0_2 = measure(qv0[2])
        
        m1_0 = measure(qv1[0])
        m1_1 = measure(qv1[1])
        m1_2 = measure(qv1[2])
        
        # Create arrays and compute parity (triggers scan)
        meas_array_0 = jnp.array([m0_0, m0_1, m0_2])
        meas_array_1 = jnp.array([m1_0, m1_1, m1_2])
        
        result = parity(meas_array_0, meas_array_1)
        return result
    
    result = test_array_parity()
    # Expected: [1 XOR 0, 0 XOR 1, 1 XOR 0] = [1, 1, 1]
    expected = jnp.array([1, 1, 1])
    assert jnp.array_equal(result, expected), f"Expected {expected}, got {result}"
