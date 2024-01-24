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

# Created by ann81984 at 07.05.2022
import numpy as np
from qrisp import *

from qrisp import QuantumVariable, mcx, cx, QuantumEnvironment, x


def test_uncomputation_example():
    verify[0] = 1

    qf1 = QuantumFloat(3, name="a")

    qf2 = QuantumFloat(3, name="b")
    qf1[:] = 4

    h(qf1[0])

    qf2[:] = 5

    res = qf1 * qf2

    temp = res.duplicate()

    qbool = QuantumBool()
    h(qbool)

    with qbool == True:
        cx(res, temp)

    res.uncompute()

    assert multi_measurement([temp, qbool]) == {
        (0.0, False): 0.5,
        (20.0, True): 0.25,
        (25.0, True): 0.25,
    }

    # ------------------
    print("Test 1 passed")

    @auto_uncompute
    def sqrt_oracle(qf):
        z(qf * qf == 0.25)

    qf = QuantumFloat(3, -1, signed=False)
    n = qf.size
    iterations = int((2**n) ** 0.5)
    h(qf)

    from qrisp.grover import diffuser

    for i in range(iterations):
        sqrt_oracle(qf)
        diffuser(qf)
        print(qf)

    assert qf.get_measurement() == {
        0.5: 0.9453,
        0.0: 0.0078,
        1.0: 0.0078,
        1.5: 0.0078,
        2.0: 0.0078,
        2.5: 0.0078,
        3.0: 0.0078,
        3.5: 0.0078,
    }

    # ---------
    print("Test 2 passed")

    city_amount = 4

    distance_matrix = (
        np.array(
            [
                [0, 0.25, 0.125, 0.5],
                [0.25, 0, 0.625, 0.375],
                [0.125, 0.625, 0, 0.75],
                [0.5, 0.375, 0.75, 0],
            ]
        )
        / 4
    )

    def calc_perm_travel_distance(permutation, precision):
        # A QuantumFloat with n qubits and exponent -n
        # can represent values between 0 and 1
        res = QuantumFloat(precision, -precision)

        # Fill QuantumDictionary
        qd = QuantumDictionary(return_type=res)
        for i in range(city_amount):
            for j in range(city_amount):
                qd[(i, j)] = distance_matrix[i, j]

        # Evaluate result
        for i in range(4):
            trip_distance = qd[permutation[i], permutation[(i + 1) % city_amount]]
            res += trip_distance

            trip_distance.uncompute()

        return res

    perm = QuantumArray(QuantumFloat(2))
    perm[:] = [2, 1, 0, 3]

    res = calc_perm_travel_distance(perm, 5)

    assert res.get_measurement() == {0.53125: 1.0}
    
    assert len(res.qs.compile().qubits) == 18

    # --------
    print("Test 3 passed")

    def triple_AND(a, b, c):
        local_quantum_bool = QuantumBool(name="local")
        result = QuantumBool(name="result")

        mcx([a, b], local_quantum_bool)
        mcx([local_quantum_bool, c], result)

        local_quantum_bool.uncompute()

        return result

    a = QuantumBool(name="a")
    b = QuantumBool(name="b")
    c = QuantumBool(name="c")

    h(a)
    b[:] = True
    c[:] = True

    result = triple_AND(a, b, c)

    assert result.get_measurement() == {False: 0.5, True: 0.5}

    # --------

    @auto_uncompute
    def triple_AND(a, b, c):
        local_quantum_bool = QuantumBool(name="local")
        result = QuantumBool(name="result")

        mcx([a, b], local_quantum_bool)
        mcx([local_quantum_bool, c], result)

        return result

    a = QuantumBool(name="a")
    b = QuantumBool(name="b")
    c = QuantumBool(name="c")

    h(a)
    b[:] = True
    c[:] = True

    result = triple_AND(a, b, c)

    assert result.get_measurement() == {False: 0.5, True: 0.5}

    print("Test 4 passed")

    # --------

    # Test uncomputation/environment interaction

    error_thrown = False

    d = QuantumVariable(1, name="d")
    cond_1 = QuantumBool(name="cond_1")
    cond_2 = QuantumBool(name="cond_2")
    cond_3 = QuantumBool(name="cond_3")

    with cond_1:
        a = QuantumVariable(1, name="a")
        b = QuantumVariable(1, name="b")
        # x(a)
        x(b)
        with cond_2:
            x(a)
            x(d)
            with cond_3:
                x(b)
            try:
                a.uncompute()
            except:
                error_thrown = True

        b.uncompute()

    assert error_thrown

    print("Test 5 passed")

    # Test recomputation

    qv_0 = QuantumVariable(1)
    qv_1 = QuantumVariable(1)
    qv_2 = QuantumVariable(1)

    cx(qv_0, qv_1)
    cx(qv_1, qv_2)

    qv_1.uncompute(recompute=True)

    qv_2.uncompute()
    print(qv_1.qs)

    qv_0 = QuantumVariable(1)
    qv_1 = QuantumVariable(1)
    qv_2 = QuantumVariable(1)
    qv_3 = QuantumVariable(1)

    cx(qv_0, qv_1)
    cx(qv_1, qv_2)
    cx(qv_1, qv_3)
    x(qv_2)

    qv_1.uncompute(recompute=True)
    qv_2.uncompute()
    qv_3.uncompute()
    assert qv_1.qs.cnot_count() == 10

    verify[0] = 0
