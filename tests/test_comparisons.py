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


from qrisp import h, QuantumFloat, multi_measurement
import numpy as np


def test_comparison():
    def test_comparison_helper(qf_0, qf_1, comp):
        qf_0 = qf_0.duplicate()
        h(qf_0)

        if isinstance(qf_1, QuantumFloat):
            qf_1 = qf_1.duplicate()
            h(qf_1)

        if comp == "eq":
            qf_res = qf_0 == qf_1
        elif comp == "neq":
            qf_res = qf_0 != qf_1
        elif comp == "leq":
            qf_res = qf_0 <= qf_1
        elif comp == "geq":
            qf_res = qf_0 >= qf_1
        elif comp == "lt":
            qf_res = qf_0 < qf_1
        elif comp == "gt":
            qf_res = qf_0 > qf_1

        if isinstance(qf_1, QuantumFloat):
            mes_res = multi_measurement([qf_0, qf_1, qf_res])
            for a, b, c in mes_res.keys():
                print(a, b, c)
                if comp == "eq":
                    assert (a == b) == c
                elif comp == "neq":
                    assert (a != b) == c
                elif comp == "leq":
                    assert (a <= b) == c
                elif comp == "geq":
                    assert (a >= b) == c
                elif comp == "lt":
                    assert (a < b) == c
                elif comp == "gt":
                    assert (a > b) == c

            # Test correct phase behavior7
            statevector = qf_res.qs.statevector("array")
            angles = np.angle(
                statevector[
                    np.abs(statevector) > 1 / 2 ** ((qf_0.size + qf_1.size) / 2)
                ]
            )
            assert np.sum(np.abs(angles)) < 0.1
        else:
            mes_res = multi_measurement([qf_0, qf_res])
            for a, c in mes_res.keys():
                b = qf_1
                print(a, b, c)
                if comp == "eq":
                    assert (a == b) == c
                elif comp == "neq":
                    assert (a != b) == c
                elif comp == "leq":
                    assert (a <= b) == c
                elif comp == "geq":
                    assert (a >= b) == c
                elif comp == "lt":
                    assert (a < b) == c
                elif comp == "gt":
                    assert (a > b) == c

    a = QuantumFloat(5, 2, signed=False)
    b = QuantumFloat(5, -1, signed=True)
    # Test all the operators
    for comp in ["eq", "neq", "lt", "gt", "leq", "geq"]:
        test_comparison_helper(a, b, comp)

    # Test specific constellations of QuantumFloat parameters
    a = QuantumFloat(3, -1, signed=True)
    b = QuantumFloat(5, 1, signed=True)
    test_comparison_helper(a, b, "lt")

    a = QuantumFloat(5, 2, signed=False)
    b = QuantumFloat(5, -1, signed=True)
    test_comparison_helper(a, b, "lt")

    a = QuantumFloat(4, -4, signed=False)
    b = QuantumFloat(4, 0, signed=True)
    test_comparison_helper(a, b, "lt")

    a = QuantumFloat(4, 1, signed=True)
    b = QuantumFloat(3, -1, signed=False)
    test_comparison_helper(a, b, "lt")

    # Test semi-classical comparisons
    a = QuantumFloat(4, signed=False)
    b = 4
    test_comparison_helper(a, b, "lt")
    test_comparison_helper(a, b, "gt")

    a = QuantumFloat(4, signed=True)
    b = 4
    test_comparison_helper(a, b, "lt")
    test_comparison_helper(a, b, "gt")

    a = QuantumFloat(4, -2, signed=False)
    b = 4
    test_comparison_helper(a, b, "lt")
    test_comparison_helper(a, b, "gt")

    a = QuantumFloat(8, 3, signed=False)
    b = 16
    test_comparison_helper(a, b, "lt")
    test_comparison_helper(a, b, "gt")
