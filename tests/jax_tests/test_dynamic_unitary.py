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

import numpy as np

from qrisp import QuantumFloat, jaspify, measure, unitary


def test_unitary_jasp_trace():
    unitary_array = np.eye(2, dtype=np.complex128)

    @jaspify
    def main():
        qv = QuantumFloat(1)
        unitary(unitary_array, qv[0])
        return measure(qv[0])

    assert not bool(main())
