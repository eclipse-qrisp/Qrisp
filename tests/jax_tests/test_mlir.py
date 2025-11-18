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

from qrisp import *
from qrisp.operators import a, c

def test_mlir_generation():
    
    
    # Test some features to make sure mlir generation works properly
    def inner(i):
        
        a = QuantumVariable(i)
        b = QuantumFloat(i)
        
        h(a)
        
        meas_res = measure(a)
        
        with control(meas_res == 0):
            for i in jrange(b.size):
                rz(1/i, b[i])
                cx(a[i], b[i])

        return a, b                
    
    def main(i):
        return expectation_value(inner, shots = i*10)(i)

    jaspr = make_jaspr(main)(2)
    xdsl_module = jaspr.to_mlir()

    # Test wheter stablehlo control flow is properly removed    
    
    from xdsl.printer import Printer
    
    def main():
        
        qv = QuantumVariable(2)
        h(qv[0])
        
        c = measure(qv[0])
        
        for i in jrange(qv.size):
            with control(c):
                x(qv[1])
        

    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()

    mlir_str = str(xdsl_module)
    
    assert "stablehlo.case" not in mlir_str
    assert "stablehlo.while" not in mlir_str
    assert "stablehlo.return" not in mlir_str
    
    # Test https://github.com/eclipse-qrisp/Qrisp/pull/296#issuecomment-3468979932
    
    H = a(0)
    orbital_amount = H.find_minimal_qubit_amount()
    U = qache(H.trotterization(forward_evolution=False))

    # Finding the gound state energy of the Water molecule with QPE
    def main():
        qv = QuantumFloat(orbital_amount)
        [x(qv[i]) for i in range(1)] # Prepare Hartree-Fock state, H2O molecule has 10 electrons

        qpe_res = QPE(qv,U,precision=1,kwargs={"steps":1})
        phi = measure(qpe_res)
        return phi

    jaspr = make_jaspr(main)()
    mlir = jaspr.to_mlir()
    