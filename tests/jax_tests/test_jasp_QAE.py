"""
********************************************************************************
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

def test_jasp_QAE():
    from qrisp import QuantumFloat, ry, z, QAE
    from qrisp.jasp import terminal_sampling
    import numpy as np

    def state_function(qb):
        ry(np.pi/4,qb)

    def oracle_function(qb):   
        z(qb)

    @terminal_sampling
    def main():
        qb = QuantumFloat(1)
        res = QAE([qb], state_function, oracle_function, precision=3)
        return res

    meas_res = main()
    
    assert np.round(meas_res[0.125],2) == 0.5
    assert np.round(meas_res[0.875],2) == 0.5


def test_QAE_integration():
    from qrisp import QuantumFloat, QuantumBool, control, z, h, ry, QAE
    from qrisp.jasp import terminal_sampling, jrange
    import numpy as np

    # We compute the integral of f(x)=(sin(x))^2 from 0 to 1
    def state_function(inp, tar):
        h(inp) # Distribution
    
        N = 2**inp.size
        for k in jrange(inp.size):
            with control(inp[k]):
                ry(2**(k+1)/N,tar)
    
    def oracle_function(inp, tar):
        z(tar)

    @terminal_sampling
    def main():
        n = 6 # 2^n sampling points for integration
        inp = QuantumFloat(n,-n)
        tar = QuantumFloat(1)
        input_list = [inp, tar]

        prec = 6 # precision
        res = QAE(input_list, state_function, oracle_function, precision=prec)
        return res
    
    meas_res = main()
    theta = np.pi*max(meas_res, key=meas_res.get)
    a = np.sin(theta)**2  

    assert np.abs(a-0.26430) < 1e-4