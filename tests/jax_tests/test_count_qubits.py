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

import pytest

from qrisp import (
    QuantumFloat,
    control,
    count_qubits,
    h,
    measure,
)
from qrisp.jasp import jrange
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    always_one,
    always_zero,
)


class TestCountQubits:

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
    def test_count_qubits(self, num_qubits):

        @count_qubits(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)
            h(qf[0])

        assert main(num_qubits) == num_qubits

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4])
    def test_count_qubits_multiple_create_qubits(self, num_qubits):

        @count_qubits(meas_behavior="0")
        def main(num_qubits):
            qf1 = QuantumFloat(num_qubits)
            qf2 = QuantumFloat(num_qubits + 1)
            qf3 = QuantumFloat(num_qubits + 2)
            qf4 = QuantumFloat(num_qubits + 3)
            h(qf1[0])
            h(qf2[0])
            h(qf3[0])
            h(qf4[0])

        expected_count = sum(num_qubits + i for i in range(4))
        assert main(num_qubits) == expected_count

    @pytest.mark.parametrize(
        "meas_behavior,num_qubits,num_qubits2,num_qubits3, expected_count",
        [(always_zero, 2, 3, 4, 5), (always_one, 2, 3, 4, 6)],
    )
    def test_count_qubits_control_flow(
        self, meas_behavior, num_qubits, num_qubits2, num_qubits3, expected_count
    ):
        """Test qubit counting with control flow based on measurement outcomes."""

        @count_qubits(meas_behavior=meas_behavior)
        def circuit(num_qubits, num_qubits2, num_qubits3):
            qv = QuantumFloat(num_qubits)
            m = measure(qv[0])
            with control(m == 0):
                qv2 = QuantumFloat(num_qubits2)
                h(qv2[0])
            with control(m == 1):
                qv3 = QuantumFloat(num_qubits3)
                h(qv3[0])

        assert circuit(num_qubits, num_qubits2, num_qubits3) == expected_count

    def test_count_qubits_loop(self):
        """Test qubit counting in a loop structure."""

        @count_qubits(meas_behavior="0")
        def circuit(num_qubits, num_iterations):
            for i in jrange(num_iterations):
                qv = QuantumFloat(num_qubits)
                h(qv[i])

        assert circuit(5, 5) == 25

    def test_count_qubits_delete_qubits(self):
        """Test qubit counting with qubit deletion."""

        @count_qubits(meas_behavior="0")
        def circuit(num_qubits):
            qv = QuantumFloat(num_qubits)
            h(qv[0])
            qv.delete()

        assert circuit(4) == 0

    def test_count_qubits_delete_qubits2(self):
        """Test qubit counting with qubit deletion."""

        @count_qubits(meas_behavior="0")
        def circuit(num_qubits):
            qv = QuantumFloat(num_qubits)
            h(qv[0])
            qv.delete()
            qv = QuantumFloat(num_qubits)
            h(qv[0])

        assert circuit(4) == 4
