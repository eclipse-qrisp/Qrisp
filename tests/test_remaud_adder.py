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

def test_remaud_adder():

    for N in range(2, 15):
        circ1 = QuantumFloat(N)
        with invert():
            for i in range(N-1):
                cx(circ1[i],circ1[i+1])

        jsp = make_jaspr(ladder1_synth_jax, garbage_collection = "none")(1)
        circ2 = jsp.to_qc(N)
        
        assert circ2.compare_unitary(circ1.qs)