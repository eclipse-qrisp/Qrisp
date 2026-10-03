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

"""Tests that ConditionEnvironment uncomputes the intermediate results of its condition."""

from qrisp import QuantumBool, QuantumFloat, conjugate, h, multi_measurement, quantum_condition, z


@quantum_condition
def small_or_seven(qf):
    # Creates two intermediate QuantumBools, which must be uncomputed as well
    return (qf < 3) | (qf == 7)


def test_intermediates_are_uncomputed():
    for value in range(8):
        qf = QuantumFloat(3)
        qf[:] = value
        t = QuantumBool()

        with small_or_seven(qf):
            t.flip()

        assert multi_measurement([qf, t]) == {(value, value < 3 or value == 7): 1.0}
        assert [qv.name for qv in qf.qs.qv_list] == [qf.name, t.name]


def test_phase_oracle():
    qf = QuantumFloat(3)
    h(qf)

    with small_or_seven(qf) as flag:
        z(flag)

    h(qf)

    assert qf.get_measurement() == {4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25}
    assert [qv.name for qv in qf.qs.qv_list] == [qf.name]


def test_comparison_inside_with_line_is_not_entered():
    qf = QuantumFloat(3)
    flag = QuantumBool()

    with conjugate(flag << (lambda qf: qf < 3))(qf):
        z(flag)

    assert [qv.name for qv in qf.qs.qv_list] == [qf.name, flag.name]
