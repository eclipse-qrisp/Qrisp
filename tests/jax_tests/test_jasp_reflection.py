# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Tests for the reflection primitive with various input types under Jasp."""

from qrisp import QuantumArray, QuantumFloat, QuantumVariable, cx, h, reflection, x
from qrisp.jasp import jrange, terminal_sampling


def ghz(*args):
    """Prepares a GHZ state on the provided quantum variables."""
    flattened_qargs = []
    for arg in args:
        if isinstance(arg, QuantumVariable):
            flattened_qargs.append(arg)
        elif isinstance(arg, QuantumArray):
            flattened_qargs.extend([qv for qv in arg.flatten()])

    first_qubit = flattened_qargs[0][0]
    h(first_qubit)
    for i in jrange(1, flattened_qargs[0].size):
        cx(first_qubit, flattened_qargs[0][i])
    for qv in flattened_qargs[1:]:
        for i in jrange(qv.size):
            cx(first_qubit, qv[i])


def test_jasp_reflection_quantum_variable():
    """Tests a reflection around a GHZ state with QuantumVariable input in Jasp."""

    @terminal_sampling
    def prepare():
        qv = QuantumVariable(5)
        x(qv)
        return qv

    @terminal_sampling
    def prepare_and_reflect():
        qv = QuantumVariable(5)
        x(qv)
        reflection(qv, ghz)
        return qv

    assert prepare() == {31: 1.0}
    assert prepare_and_reflect() == {0: 1.0}


def test_jasp_reflection_quantum_array():
    """Tests a reflection around a GHZ state with QuantumArray input in Jasp."""

    @terminal_sampling
    def prepare():
        qa = QuantumArray(QuantumFloat(3), shape=(3,))
        x(qa)
        return qa[0], qa[1], qa[2]

    @terminal_sampling
    def prepare_and_reflect():
        qa = QuantumArray(QuantumFloat(3), shape=(3,))
        x(qa)
        reflection(qa, ghz)
        return qa[0], qa[1], qa[2]

    assert prepare() == {(7, 7, 7): 1.0}
    assert prepare_and_reflect() == {(0, 0, 0): 1.0}


def test_jasp_reflection_list_quantum_variable():
    """Tests a reflection around a GHZ state with list of QuantumVariable input in Jasp."""

    @terminal_sampling
    def prepare():
        qv_list = [QuantumVariable(3), QuantumVariable(2)]
        x(qv_list[0])
        x(qv_list[1])
        return qv_list[0], qv_list[1]

    @terminal_sampling
    def prepare_and_reflect():
        qv_list = [QuantumVariable(3), QuantumVariable(2)]
        x(qv_list[0])
        x(qv_list[1])
        reflection(qv_list, ghz)
        return qv_list[0], qv_list[1]

    assert prepare() == {(7, 3): 1.0}
    assert prepare_and_reflect() == {(0, 0): 1.0}


def test_jasp_reflection_tuple_quantum_variable():
    """Tests a reflection around a GHZ state with tuple of QuantumVariable input in Jasp."""

    @terminal_sampling
    def prepare():
        qv_tuple = (QuantumVariable(3), QuantumVariable(2))
        x(qv_tuple[0])
        x(qv_tuple[1])
        return qv_tuple[0], qv_tuple[1]

    @terminal_sampling
    def prepare_and_reflect():
        qv_tuple = (QuantumVariable(3), QuantumVariable(2))
        x(qv_tuple[0])
        x(qv_tuple[1])
        reflection(qv_tuple, ghz)
        return qv_tuple[0], qv_tuple[1]

    assert prepare() == {(7, 3): 1.0}
    assert prepare_and_reflect() == {(0, 0): 1.0}


def test_jasp_reflection_list_quantum_variable_quantum_array():
    """Tests a reflection around a GHZ state with list of QuantumVariable and QuantumArray input in Jasp."""

    def ghz(qv, qa):
        h(qv[0])
        for i in jrange(1, qv.size):
            cx(qv[0], qv[i])

        for var in qa:
            for i in jrange(var.size):
                cx(qv[0], var[i])

    @terminal_sampling
    def prepare():
        qv = QuantumVariable(5)
        qa = QuantumArray(QuantumFloat(3), shape=(3,))
        x(qv)
        x(qa)
        return qv, qa[0], qa[1], qa[2]

    @terminal_sampling
    def prepare_and_reflect():
        qv = QuantumVariable(5)
        qa = QuantumArray(QuantumFloat(3), shape=(3,))
        x(qv)
        x(qa)
        reflection([qv, qa], ghz)
        return qv, qa[0], qa[1], qa[2]

    assert prepare() == {(31, 7, 7, 7): 1.0}
    assert prepare_and_reflect() == {(0, 0, 0, 0): 1.0}
