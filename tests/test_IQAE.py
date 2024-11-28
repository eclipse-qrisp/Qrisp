"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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


def test_IQAE_integration():
    from qrisp import QuantumFloat, QuantumBool, control, z, h, ry, IQAE
    import numpy as np

    # We compute the integral of f(x)=(sin(x))^2 from 0 to 1
    def state_function(inp, tar):
        h(inp) # Distribution
    
        N = 2**inp.size
        for k in range(inp.size):
            with control(inp[k]):
                ry(2**(k+1)/N,tar)
    
    def oracle_function(inp, tar):
        z(tar)

    n = 6 # 2^n sampling points for integration
    inp = QuantumFloat(n,-n)
    tar = QuantumBool()
    input_list = [inp, tar]

    eps = 0.01
    alpha = 0.01

    a = IQAE(input_list, state_function, oracle_function, eps=eps, alpha=alpha)
    assert np.abs(a-0.26716231971793425)<0.01


