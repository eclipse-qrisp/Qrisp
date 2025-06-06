"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp import *
from qrisp.jasp import *

def test_L1_ladder():
    def main(N):
        qf = QuantumFloat(N)
        ladder1_synth_jax(qf)
        

    for N in range(2, 10):
        circ1 = QuantumFloat(N)
        with invert():
            for i in range(N-1):
                cx(circ1[i],circ1[i+1])
        jsp = make_jaspr(main, garbage_collection = "none")(1)
        circ2 = jsp.to_qc(N)
        
        assert circ2.compare_unitary(circ1.qs)
        
def test_remaud_adder():
    def main(N):
        x2 = QuantumFloat(N)
        y2 = QuantumFloat(N)
        x(x2)
        x(y2)
        ladder2_synth_jax(x2, y2, method='balauca')
        return x2,y2
    
    for N in range(2, 15):
        x1 = QuantumFloat(N)
        y1 = QuantumFloat(N)
        x(x1)
        x(y1)
        with invert():
            for i in range(N-1):
                mcx([x1[i],y1[i]],x1[i+1])
        res1 = multi_measurement([x1, y1])
        print(dict(res1))
        
        res2 = terminal_sampling(main)(N)
        print(dict(res2))
        assert res1.keys() == res2.keys(), f"Key mismatch:\nres1: {res1.keys()}\nres2: {res2.keys()}"
        res1_vals = np.array([res1[k] for k in sorted(res1)])
        res2_vals = np.array([res2[k] for k in sorted(res2)])

        assert np.allclose(res1_vals, res2_vals, rtol=1e-7, atol=1e-7), \
            f"Value mismatch:\nres1: {res1_vals}\nres2: {res2_vals}"