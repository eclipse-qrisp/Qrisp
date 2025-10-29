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


def test_quantum_kernel():

    @quantum_kernel
    def inner_f(k):
        qf = QuantumFloat(4)

        with conjugate(h)(qf):
            for i in jrange(k):
                t(qf[0])

        return measure(qf)

    def main(a, b):

        a = inner_f(a)
        b = inner_f(b)

        return a + b

    jaxpr = make_jaxpr(main)(1, 1)
    main = jaspify(main)

    assert main(4, 8) == 1
    assert main(4, 4) == 2
    assert main(8, 4) == 1
    assert main(8, 8) == 0
