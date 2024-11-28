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


def test_QMCI():
    from qrisp import QuantumFloat
    from qrisp.algorithms.qmci import QMCI
    import numpy as np

    def f(qf):
        return qf*qf

    qf = QuantumFloat(3,-3)
    a = QMCI([qf], f)

    assert np.abs(a-0.2734375)<0.01