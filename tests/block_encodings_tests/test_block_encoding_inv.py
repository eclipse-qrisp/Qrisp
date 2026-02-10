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

import numpy as np
from qrisp import *
from qrisp.block_encodings import BlockEncoding


def test_block_encoding_inv():

    A = np.array([[0.73255474, 0.14516978, -0.14510851, -0.0391581],
                [0.14516978, 0.68701415, -0.04929867, -0.00999921],
                [-0.14510851, -0.04929867, 0.76587818, -0.03420339],
                [-0.0391581, -0.00999921, -0.03420339, 0.58862043]])

    b = np.array([0, 1, 0, 1])

    BA = BlockEncoding.from_array(A)

    BA_inv = BA.inv(0.01, np.linalg.cond(A))

    # Prepares operand variable in state |b>
    def prep_b():
        operand = QuantumVariable(2)
        prepare(operand, b)
        return operand

    @terminal_sampling
    def main():
        operand = BA_inv.apply_rus(prep_b)()
        return operand

    res_dict = main()
    for k, v in res_dict.items():
        res_dict[k] = v**0.5
    q = np.array([res_dict.get(key, 0) for key in range(len(b))])
    
    c = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    assert np.linalg.norm(np.abs(c) - q) < 1e-2
