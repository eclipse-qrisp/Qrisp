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

def test_reflection():

    from qrisp import QuantumVariable, QuantumFloat, QuantumArray, OutcomeArray, h, x, cx, reflection, multi_measurement

    # Reflection with QuantumVariable as input
    def ghz(qv):
        h(qv[0])

        for i in range(1, qv.size):
            cx(qv[0], qv[i])


    qv = QuantumVariable(5)
    x(qv)
    res = qv.get_measurement()
    assert res == {'11111': 1.0}

    reflection(qv, ghz)
    res = qv.get_measurement()
    assert res == {'00000': 1.0}


    # Reflection with list[QuantumVariable | QuantumArray] as input
    def ghz(qv, qa):
        h(qv[0])

        for i in range(1, qv.size):
            cx(qv[0], qv[i])

        for var in qa:
            for i in range(var.size):
                cx(qv[0], var[i])


    qv = QuantumVariable(5)
    qa = QuantumArray(QuantumFloat(3), shape=(3,))
    x(qv)
    x(qa)
    res = multi_measurement([qv, qa])
    assert res == {('11111', OutcomeArray([7, 7, 7])): 1.0}

    reflection([qv, qa], ghz)
    res = multi_measurement([qv, qa])
    assert res == {('00000', OutcomeArray([0, 0, 0])): 1.0}


def test_jasp_reflection():

    from qrisp import QuantumVariable, QuantumArray, h, x, cx, reflection
    from qrisp.jasp import terminal_sampling, jrange

    # Reflection with list[QuantumVariable | QuantumArray] as input
    def ghz(qv):
        h(qv[0])

        for i in jrange(1, qv.size):
            cx(qv[0], qv[i])


    @terminal_sampling
    def main():
        qv = QuantumVariable(5)
        x(qv)
        reflection(qv, ghz)
        return qv

    res = main()
    assert res == {0: 1.0}
