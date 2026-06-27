"""********************************************************************************
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

from qrisp import OutcomeArray, QuantumArray, QuantumFloat, QuantumVariable, cx, h, multi_measurement, reflection, x
from qrisp.jasp import jrange, terminal_sampling


def ghz(*args):
    """Prepares a GHZ state on the provided quantum variables."""
    flattened_qargs = []
    for arg in args:
        if isinstance(arg, QuantumVariable):
            flattened_qargs.append(arg)
        elif isinstance(arg, QuantumArray):
            flattened_qargs.extend([qv for qv in arg.flatten()])

    qubits = sum([arg.reg for arg in flattened_qargs], [])
    h(qubits[0])
    for i in range(1, len(qubits)):
        cx(qubits[0], qubits[i])


def test_reflection_quantum_variable():
    """Tests that the reflection primitive correctly applies a reflection around a GHZ state."""
    qv = QuantumVariable(5)
    x(qv)
    res = qv.get_measurement()
    assert res == {"11111": 1.0}

    reflection(qv, ghz)
    res = qv.get_measurement()
    assert res == {"00000": 1.0}


def test_reflection_quantum_array():
    """Tests that the reflection primitive correctly applies a reflection around a GHZ state with QuantumArray input."""
    qa = QuantumArray(QuantumFloat(3), shape=(3,))
    x(qa)
    res = qa.get_measurement()
    assert res == {OutcomeArray([7, 7, 7]): 1.0}

    reflection(qa, ghz)
    res = qa.get_measurement()
    assert res == {OutcomeArray([0, 0, 0]): 1.0}


def test_reflection_list_quantum_variable():
    """Tests that the reflection primitive correctly applies a reflection around a GHZ state with list of QuantumVariable input."""
    qv_list = [QuantumVariable(3), QuantumVariable(2)]
    x(qv_list[0])
    x(qv_list[1])
    res = multi_measurement(qv_list)
    assert res == {("111", "11"): 1.0}

    reflection(qv_list, ghz)
    res = multi_measurement(qv_list)
    assert res == {("000", "00"): 1.0}


def test_reflection_tuple_quantum_variable():
    """Tests that the reflection primitive correctly applies a reflection around a GHZ state with tuple of QuantumVariable input."""
    qv_tuple = (QuantumVariable(3), QuantumVariable(2))
    x(qv_tuple[0])
    x(qv_tuple[1])
    res = multi_measurement(qv_tuple)
    assert res == {("111", "11"): 1.0}

    reflection(qv_tuple, ghz)
    res = multi_measurement(qv_tuple)
    assert res == {("000", "00"): 1.0}


def test_reflection_list_quantum_varaible_quantum_array():
    """Tests that the reflection primitive correctly applies a reflection around a GHZ state with list of QuantumVariable and QuantumArray input."""

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
    assert res == {("11111", OutcomeArray([7, 7, 7])): 1.0}

    reflection([qv, qa], ghz)
    res = multi_measurement([qv, qa])
    assert res == {("00000", OutcomeArray([0, 0, 0])): 1.0}


def test_jasp_reflection():
    """Tests that the reflection primitive correctly applies a reflection around a GHZ state in Jasp."""

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
