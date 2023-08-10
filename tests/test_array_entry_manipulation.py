"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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


def array_entry_manipulation_test():
    from qrisp import QuantumBool, QuantumArray, QuantumFloat, h, multi_measurement

    q_array = QuantumArray(QuantumBool(), (4, 4))
    index_0 = QuantumFloat(2, signed=False)
    index_1 = QuantumFloat(2, signed=False)

    index_0[:] = 2
    index_1[:] = 1

    with q_array[(index_0, index_1)] as entry:
        entry.flip()

    res_array = q_array.most_likely()

    assert res_array[index_0.most_likely(), index_1.most_likely()]
