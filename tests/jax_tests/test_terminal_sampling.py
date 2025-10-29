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

from jax import make_jaxpr

from qrisp import *
from qrisp.jasp import *


def test_terminal_sampling():

    @terminal_sampling
    def main():

        qbl = QuantumBool()
        qf = QuantumFloat(4)

        # Bring qbl into superposition
        h(qbl)

        # Perform a measure
        cl_bl = measure(qbl)

        # Perform a conditional operation based on the measurement outcome
        with control(cl_bl):
            qf[:] = 1
            h(qf[2])

        return qf, qbl

    assert main() in [{(1.0, True): 0.5, (5.0, True): 0.5}, {(0.0, False): 1.0}]

    @terminal_sampling
    def main(i, j):
        qf = QuantumFloat(3)
        a = QuantumFloat(3)
        qbl = QuantumBool()
        h(qf[i])
        cx(qf[i], a[j])
        cx(qf[i], qbl[0])
        return qf, a, qbl

    for i in range(3):
        for j in range(3):
            assert main(i, j) == {(0.0, 0.0, False): 0.5, (2**i, 2**j, True): 0.5}

    def ansatz(theta):

        qv = QuantumFloat(2)
        rx(theta[0], qv)
        return qv

    @terminal_sampling
    def main():

        x0 = jnp.array([0.5])

        qf = ansatz(x0)

        return qf

    main()

    @terminal_sampling
    def main():
        a = QuantumFloat(2, signed=True)
        b = QuantumFloat(2)

        h(a)

        return a, b  # This does not work

    res = main()
