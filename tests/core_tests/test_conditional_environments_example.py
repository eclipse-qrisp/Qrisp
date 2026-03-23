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

# Created by ann81984 at 29.04.2022
# import pytest
import numpy

from qrisp.interface import QiskitBackend
from qrisp import (
    QuantumFloat,
    QuantumChar,
    QuantumArray,
    merge,
    x,
    h,
    OutcomeArray,
    multi_measurement,
    auto_uncompute,
    invert
)


# check if the respone from QuantumArray.get_measurement() has the expected formet.
# Should return a list of tuples of the type (numpy.ndarray, float)
# ie. [(array([1,1,0]), 232), (array([1,1,3]), 115), ...]
def test_conditional_environments_example():
    qc = QuantumChar(name="q ch")

    qc[:] = "f"

    qf = QuantumFloat(4, -1, signed=True)
    q_array = QuantumArray(qf, 4)

    h(qc[0])

    q_array[0] += -0.5

    with "f" == qc:
        h(q_array[0][0])

        with q_array[0] == -0.5:
            q_array[2].encode(-0.5, permit_dirtyness = True)
            q_array[1] -= -1
            with q_array[1] == 1:
                q_array[3].init_from(q_array[2])

        # Test in-environment printing
        print(qc.qs)

    print(multi_measurement([qc, q_array]))
    assert multi_measurement([qc, q_array]) == {
        ("e", OutcomeArray([-0.5, 0.0, 0.0, 0.0])): 0.5,
        ("f", OutcomeArray([-1.0, 0.0, 0.0, 0.0])): 0.25,
        ("f", OutcomeArray([-0.5, 1.0, -0.5, -0.5])): 0.25,
    }

    assert len(qc.qs.qv_list) == 5

    from qrisp import as_hamiltonian, QuantumBool, control

    @as_hamiltonian
    def dummy_d_function(i, j):
        return i * j

    def dummy_s_function(qf_list):
        m = len(qf_list)
        for i in range(m):
            dummy_d_function(qf_list[i], qf_list[(i + 1) % m])

    qf_list = [QuantumFloat(2) for j in range(3)]

    qf_list[-1][:] = 1

    qbool = QuantumBool()
    h(qbool)
    with control(qbool[0]):
        dummy_s_function(qf_list)
        print(qbool.qs)

    h(qbool)

    assert qbool.most_likely() == False

    # -------

    a = QuantumFloat(2, name="a")
    b = QuantumFloat(2, name="b")
    c = QuantumFloat(1, name="c")
    d = QuantumFloat(2, name="d")

    a[:] = 1
    b[:] = 1
    c[:] = 1

    def test_f(a):
        return a == 1

    from qrisp import ConditionEnvironment, ControlEnvironment

    with test_f(a):
        with b == c:
            x(d)
            print(d.qs)

    # Test if uncomputation worked properly
    assert len(a.qs.qv_list) == 4

    # Test if condition environments get invoked properly
    with b == 2:
        assert isinstance(b.qs.env_stack[-1], ConditionEnvironment)
        temp = d == 1
        # print(d.qs)
        with temp:
            assert isinstance(d.qs.env_stack[-1], ControlEnvironment)
            x(c)
            # print(c.qs)

    assert len(a.qs.qv_list) == 5

    # ----------
    a = QuantumFloat(2, signed=True)
    b = QuantumFloat(2, signed=True)
    c = QuantumFloat(1, signed=True)
    d = QuantumFloat(2, signed=True)

    a[:] = 1
    b[:] = 1
    c[:] = 1

    def test_f(a):
        return a == 1

    with test_f(a):
        with b == c:
            x(d)
            print(d.qs)

    assert len(a.qs.qv_list) == 4
    
    @auto_uncompute
    def test_function(qf_a, qbl):
        
        with qf_a == 0:
            x(qbl)
        
        # print(qbl.qs)
        return qbl
        
    qf_a = QuantumFloat(2)
    qbl = QuantumBool()

    # qf_a[:] = 3

    with invert():
        test_function(qf_a, qbl)
        
    print(qf_a.qs)
