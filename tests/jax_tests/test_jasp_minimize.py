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

def test_jasp_minimize():
    from qrisp import QuantumFloat, ry
    from qrisp.jasp import expectation_value, minimize, jaspify
    import jax.numpy as jnp
    import numpy as np

    def state_prep(theta):
        qv = QuantumFloat(1)
        ry(theta[0], qv)
        return qv
    
    def objective(theta, state_prep):
        return expectation_value(state_prep, shots=100)(theta)
    

    @jaspify(terminal_sampling=True)
    def main():

        x0 = jnp.array([1.0])

        return minimize(objective,x0,args=(state_prep,))

    results = main()
    print(results.x)
    print(results.fun)
    assert np.round(results.x,1)==0
    assert np.round(results.fun,1)==0