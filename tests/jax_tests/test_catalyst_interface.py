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