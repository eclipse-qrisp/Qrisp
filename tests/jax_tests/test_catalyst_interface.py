"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

from qrisp import *
from qrisp.jasp import *
import time

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
    
    assert jaspr.qjit(4)[0] == 5.25
    
    
    def int_encoder(qv, encoding_int):
        for i in jrange(qv.size):
            with control(encoding_int & (1<<i)):
                x(qv[i])

    @qjit
    def test_f(a):
        time.sleep(1)        
        qv = QuantumFloat(4)
        int_encoder(qv, a)
        return measure(qv)

    t0 = time.time()

    # Test classical control flow    
    assert test_f(4)[0] == 4
    assert test_f(5)[0] == 5
    assert test_f(6)[0] == 6
    assert test_f(7)[0] == 7

    # Test QJIT caching
    assert time.time() - t0 < 2
    
    