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
from jax import make_jaxpr

def test_stim_extraction():

    @extract_stim
    def main():
        qv = QuantumVariable(4)
        x(qv[1])
        a = measure(qv[0])
        b = measure(qv[1])
        return measure(qv)

    meas_indices, stim_circuit = main()
    sampler = stim_circuit.compile_sampler()
    
    assert np.all(sampler.sample(5)[:,meas_indices] == np.array([[False, True, False, False]]*5))
