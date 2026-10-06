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

"""Tests for measure in static mode."""

import pytest

from qrisp import QuantumArray, QuantumBool, QuantumFloat, QuantumVariable, measure
from qrisp.circuit import Clbit


def measurements(qs):
    """Return the (qubit, clbit) pairs of the measurements in a QuantumSession."""
    return [(ins.qubits[0], ins.clbits[0]) for ins in qs.data if ins.op.name == "measure"]


def test_static_single_qubit():
    """A single qubit is measured into one new Clbit."""
    qv = QuantumVariable(2)
    clbit = measure(qv[1])
    assert isinstance(clbit, Clbit)
    assert measurements(qv.qs) == [(qv[1], clbit)]


@pytest.mark.parametrize("make_qubits", [list, tuple])
def test_static_sequence_of_qubits(make_qubits):
    """Each qubit of a list or tuple is measured into its own Clbit, in order."""
    qv = QuantumVariable(3)
    clbits = measure(make_qubits([qv[2], qv[0]]))
    assert isinstance(clbits, list)
    assert measurements(qv.qs) == [(qv[2], clbits[0]), (qv[0], clbits[1])]


def test_static_quantum_variable():
    """Every qubit of a QuantumVariable is measured into its own Clbit."""
    qv = QuantumFloat(3)
    clbits = measure(qv)
    assert len(clbits) == len(set(clbits)) == qv.size
    assert measurements(qv.qs) == list(zip(qv, clbits))


def test_static_quantum_array():
    """A QuantumArray gets one Clbit per element."""
    qa = QuantumArray(QuantumBool(), shape=2)
    clbits = measure(qa)
    assert len(clbits) == len(qa)
    measured = measurements(qa.qs)
    assert [clbit for _, clbit in measured] == clbits
    # Each element is a QuantumBool, so each Clbit belongs to a different qubit.
    assert len({qubit for qubit, _ in measured}) == len(qa)


def test_static_repeated_measurement():
    """Measuring the same qubit twice creates two Clbits."""
    qv = QuantumVariable(1)
    first, second = measure(qv[0]), measure(qv[0])
    assert first is not second
    assert measurements(qv.qs) == [(qv[0], first), (qv[0], second)]


def test_static_measurement_keeps_basis_state():
    """Measuring a basis state does not change its simulated outcome."""
    qf = QuantumFloat(3)
    qf[:] = 5
    measure(qf)
    assert qf.get_measurement() == {5: 1.0}


def test_static_errors():
    """Inputs without a QuantumSession raise the errors of find_qs."""
    with pytest.raises(Exception, match="Couldn't find QuantumSession"):
        measure([])
    with pytest.raises(TypeError, match="not iterable"):
        measure(1)
