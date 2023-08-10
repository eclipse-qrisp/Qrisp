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


def test_QuantumArray_indexing():
    from qrisp import QuantumBool, QuantumArray, QuantumFloat, h, multi_measurement

    q_array = QuantumArray(QuantumBool(), shape=(4, 4))
    index_0 = QuantumFloat(2)
    index_1 = QuantumFloat(2)

    index_0[:] = 2
    index_1[:] = 1

    h(index_0[0])

    with q_array[index_0, index_1] as entry:
        entry.flip()

    assert q_array.most_likely()[index_0.most_likely(), index_1.most_likely()] == True


def test_QuantumDictionary_load():
    from qrisp import QuantumDictionary, QuantumFloat, h

    qtype = QuantumFloat(4, -2)
    float_qd = QuantumDictionary(return_type=qtype)
    float_qd[0] = 1
    float_qd[1] = 2

    key_qv = QuantumFloat(1, signed=False)
    h(key_qv)
    value_qv = qtype.duplicate()

    float_qd.load(key_qv, value_qv, synth_method="pprm")
    assert value_qv.get_measurement() == {1.0: 0.5, 2.0: 0.5}


def test_QuantumFloat():
    from qrisp import QuantumFloat, invert

    qf_6 = QuantumFloat(3)
    qf_7 = QuantumFloat(3)
    qf_6[:] = 7
    qf_7[:] = 2
    qf_8 = qf_6 / qf_7
    assert qf_8.get_measurement() == {3.5: 1.0}

    qf_9 = qf_6 // qf_7
    assert qf_9.get_measurement() == {3.0: 1.0}

    qf_10 = QuantumFloat(3, -1)
    qf_10[:] = 3.5
    qf_11 = qf_10**-1
    assert qf_11.get_measurement() == {0.25: 1.0}

    qf_15 = QuantumFloat(3)
    qf_15[:] = 4
    qf_16 = QuantumFloat(3)
    qf_16[:] = 3
    qf_15 += qf_16
    qf_15 -= 2
    assert qf_15.get_measurement() == {5.0: 1.0}


def test_ConditionEnvironment():
    from qrisp import (
        QuantumChar,
        QuantumFloat,
        QuantumBool,
        QuantumVariable,
        h,
        x,
        cx,
        mcx,
        p,
        multi_measurement,
        ConditionEnvironment,
    )
    import numpy as np

    q_ch = QuantumChar()
    qf = QuantumFloat(3, signed=True)

    h(q_ch[0])

    with q_ch == "a":
        qf += 2

    with q_ch == "a" as cond_bool:
        qf += 2
        cond_bool.flip()
        qf -= 2
        p(np.pi, cond_bool)

    assert multi_measurement([q_ch, qf]) == {("a", 4): 0.5, ("b", -2): 0.5}

    def quantum_eq(qv_0, qv_1):
        res = QuantumBool(name="res_bool")

        if qv_0.size != qv_1.size:
            raise Exception(
                "Tried to evaluate equality condition for QuantumVariables of differing size"
            )

        temp_qv = QuantumVariable(qv_0.size)

        cx(qv_0, temp_qv)
        cx(qv_1, temp_qv)
        x(temp_qv)

        mcx(temp_qv, res)

        return res

    # Create some sample arguments on which to evaluate the condition
    q_bool_0 = QuantumBool()
    q_bool_1 = QuantumBool()
    q_bool_2 = QuantumBool()

    h(q_bool_0)

    with ConditionEnvironment(cond_eval_function=quantum_eq, args=[q_bool_0, q_bool_1]):
        q_bool_2.flip()

    assert multi_measurement([q_bool_0, q_bool_1, q_bool_2]) == {
        (False, False, True): 0.5,
        (True, False, False): 0.5,
    }


def test_InversionEnvironment():
    from qrisp import QuantumFloat, invert, QuantumBool, h, q_mult, multi_measurement

    qf = QuantumFloat(3)

    h(qf)

    mult_res = q_mult(qf, qf)

    q_bool = QuantumBool()

    with mult_res < 10:
        q_bool.flip()

    with invert():
        q_mult(qf, qf, target=mult_res)

    mult_res.delete(verify=True)

    assert multi_measurement([qf, q_bool]) == {
        (0, True): 0.125,
        (1, True): 0.125,
        (2, True): 0.125,
        (3, True): 0.125,
        (4, False): 0.125,
        (5, False): 0.125,
        (6, False): 0.125,
        (7, False): 0.125,
    }
