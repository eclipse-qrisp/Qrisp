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


def test_jasp_QITE_GC():
    from qrisp import rx, h, ry, s_dg, swap, QuantumFloat
    from qrisp.jasp import terminal_sampling
    from qrisp.algorithms.qite import QITE
    from qrisp import multi_measurement
    import jax.numpy as jnp
    import numpy as np

    def U_s(a):
        h(a)
        s_dg(a[2])

    def U_0(a):
        rx(1.01, a[0])
        h(a[1])
        h(a[2])
        ry(0.142, a[1])
        swap(a[0], a[1])
        swap(a[2], a[1])

    def exp_H(a, time):
        rx(-time, a)

    s = jnp.array([0.2, 1.1, 0.7, 0.43, 0.1])
    k = 5

    @terminal_sampling
    def QITE_jasp():
        qarg = QuantumFloat(3)
        U_s(qarg)
        QITE(qarg, U_0, exp_H, s, k, method="GC")
        return qarg

    def QITE_basic():
        qarg = QuantumFloat(3)
        U_s(qarg)
        QITE(qarg, U_0, exp_H, s, k, method="GC")
        return multi_measurement([qarg])

    a = QITE_jasp()
    b = QITE_basic()

    for i in range(8):
        assert(np.abs(a[float(i)] - b[(i,)]) < 0.001)

def test_jasp_QITE_HOPF():
    from qrisp import rx, h, ry, s_dg, swap, QuantumFloat
    from qrisp.jasp import terminal_sampling
    from qrisp.algorithms.qite import QITE
    from qrisp import multi_measurement
    import jax.numpy as jnp
    import numpy as np

    def U_s(a):
        h(a)
        s_dg(a[2])

    def U_0(a):
        rx(1.01, a[0])
        h(a[1])
        h(a[2])
        ry(0.142, a[1])
        swap(a[0], a[1])
        swap(a[2], a[1])

    def exp_H(a, time):
        rx(-time, a)

    s = jnp.array([0.2, 1.1, 0.7])
    k = 3

    @terminal_sampling
    def QITE_jasp():
        qarg = QuantumFloat(3)
        U_s(qarg)
        QITE(qarg, U_0, exp_H, s, k, method="HOPF")
        return qarg

    def QITE_basic():
        qarg = QuantumFloat(3)
        U_s(qarg)
        QITE(qarg, U_0, exp_H, s, k, method="HOPF")
        return multi_measurement([qarg])

    a = QITE_jasp()
    b = QITE_basic()

    for i in range(8):
        assert(np.abs(a[float(i)] - b[(i,)]) < 0.001)