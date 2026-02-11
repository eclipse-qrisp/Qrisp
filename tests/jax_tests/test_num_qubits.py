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
    h,
    measure,
    num_qubits,
)
from qrisp.jasp import jrange
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    always_one,
    always_zero,
)


class TestNumQubitsSimple:
    """Test cases for the num_qubits metric, which counts the number of qubits allocated at the end of a circuit execution."""

    @pytest.mark.parametrize("num_qubits_input", [1, 2, 3, 4])
    def test_num_qubits_simple(self, num_qubits_input):
        """Test that the number of qubits is correctly counted for a simple circuit."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qf = QuantumFloat(num_qubits_input)
            h(qf[0])

        assert main(num_qubits_input) == num_qubits_input

    @pytest.mark.parametrize("num_qubits_input", [1, 2, 3, 4])
    def test_num_qubits_multiple_create_qubits(self, num_qubits_input):
        """Test that multiple allocations are correctly counted."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qf1 = QuantumFloat(num_qubits_input)
            qf2 = QuantumFloat(num_qubits_input + 1)
            qf3 = QuantumFloat(num_qubits_input + 2)
            qf4 = QuantumFloat(num_qubits_input + 3)
            h(qf1[0])
            h(qf2[0])
            h(qf3[0])
            h(qf4[0])

        expected_count = sum(num_qubits_input + i for i in range(4))
        assert main(num_qubits_input) == expected_count

    def test_num_qubits_delete_qubits(self):
        """Test qubit counting with qubit deletion."""

        @num_qubits(meas_behavior="0")
        def circuit_del(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            qv.delete()

        assert circuit_del(4) == 0

    def test_num_qubits_delete_qubits2(self):
        """Test qubit counting with qubit deletion followed by reallocation."""

        @num_qubits(meas_behavior="0")
        def circuit_del2(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            qv.delete()
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])

        assert circuit_del2(4) == 4

    def test_num_qubits_all_deleted_in_end(self):
        """Everything deleted before termination => final count is 0."""

        @num_qubits(meas_behavior="0")
        def circuit_all_del(n):
            a = QuantumFloat(n)
            b = QuantumFloat(n + 1)
            h(a[0])
            h(b[0])
            a.delete()
            b.delete()

        assert circuit_all_del(4) == 0

    def test_num_qubits_unused_allocation_semantics(self):
        """Test the expected behavior for qubits that are allocated but never used (e.g., no gates, no measurements)."""

        @num_qubits(meas_behavior="0")
        def circuit_unused(n):
            _qv = QuantumFloat(n)

        assert circuit_unused(4) == 0

    def test_num_qubits_alias_delete_counts_once(self):
        """
        Test that if we create an alias to a quantum variable and delete the alias,
        it only counts as one deletion.
        """

        @num_qubits(meas_behavior="0")
        def circuit_alias_del(n):
            qv = QuantumFloat(n)
            alias = qv
            h(qv[0])
            alias.delete()

        assert circuit_alias_del(4) == 0


class TestNumQubitsControlFlow:
    """
    Test cases for the num_qubits metric in the presence of control flow,
    possibly with qubit allocation and deletion, and with control flow
    based on measurement outcomes.
    """

    @pytest.mark.parametrize(
        "meas_behavior,num_qubits_input,num_qubits_input2,num_qubits_input3, expected_count",
        [(always_zero, 2, 3, 4, 5), (always_one, 2, 3, 4, 6)],
    )
    def test_num_qubits_control_flow(
        self,
        meas_behavior,
        num_qubits_input,
        num_qubits_input2,
        num_qubits_input3,
        expected_count,
    ):
        """Test qubit counting with control flow based on measurement outcomes."""

        @num_qubits(meas_behavior=meas_behavior)
        def circuit_cf(num_qubits_input, num_qubits_input2, num_qubits_input3):
            qv = QuantumFloat(num_qubits_input)
            m = measure(qv[0])
            with control(m == 0):
                qv2 = QuantumFloat(num_qubits_input2)
                h(qv2[0])
            with control(m == 1):
                qv3 = QuantumFloat(num_qubits_input3)
                h(qv3[0])

        assert (
            circuit_cf(num_qubits_input, num_qubits_input2, num_qubits_input3)
            == expected_count
        )

    def test_num_qubits_loop(self):
        """Test qubit counting in a loop structure."""

        @num_qubits(meas_behavior="0")
        def circuit_loop(num_qubits_input, num_iterations):
            for i in jrange(num_iterations):
                qv = QuantumFloat(num_qubits_input)
                h(qv[i])

        assert circuit_loop(5, 5) == 25

    def test_delete_qubits_in_loop(self):
        """Test qubit counting with qubit deletion inside a loop."""

        num_iterations = 10

        @num_qubits(meas_behavior="1")
        def circuit_loop_del(num_qubits_input):

            list_of_qvs = []

            for i in range(num_iterations):
                qv = QuantumFloat(num_qubits_input)
                h(qv[i])
                list_of_qvs.append(qv)

            qv = QuantumFloat(1)
            m = measure(qv[0])

            with control(m == 0):
                # does not matter the size as
                # it should never be called
                qv2 = QuantumFloat(1000000)
                h(qv2[0])
                list_of_qvs.append(qv2)

            with control(m == 1):
                qv3 = QuantumFloat(10)
                h(qv3[0])
                list_of_qvs.append(qv3)

            for i in range(num_iterations):
                list_of_qvs[i].delete()

            qv.delete()

        assert circuit_loop_del(5) == 10

    def test_num_qubits_branch_dependent_delete(self):
        """Test qubit counting when deletion is performed in a measurement-dependent branch."""

        @num_qubits(meas_behavior=always_zero)
        def circuit_branch_del(n):
            qv = QuantumFloat(n)
            qv_big = QuantumFloat(2 * n)
            m = measure(qv[0])

            with control(m == 0):
                qv_big.delete()

            h(qv[0])

        assert circuit_branch_del(4) == 4

        @num_qubits(meas_behavior=always_one)
        def circuit_branch_del2(n):
            qv = QuantumFloat(n)
            qv_big = QuantumFloat(2 * n)
            m = measure(qv[0])

            with control(m == 0):
                qv_big.delete()

            h(qv[0])

        assert circuit_branch_del2(4) == 12


class TestNumQubitsExceptions:
    """Test cases for exceptions raised by the num_qubits metric."""

    def test_num_qubits_simulation_not_implemented(self):
        """Test that num_qubits via simulation raises NotImplementedError."""

        @num_qubits(meas_behavior="sim")
        def main():
            pass

        with pytest.raises(
            NotImplementedError,
            match="Num qubits metric via simulation is not implemented yet",
        ):
            main()

    def test_error_on_invalid_measurement_behavior(self):
        """Test that an invalid measurement behavior raises ValueError."""

        @num_qubits(meas_behavior="invalid_behavior")
        def main():
            pass

        with pytest.raises(
            ValueError,
            match="Don't know how to compute required resources via method invalid_behavior",
        ):
            main()

    def test_error_on_non_boolean_measurement(self):
        """Test that a non-boolean measurement result raises ValueError."""

        def invalid_meas_behavior(_):
            return 42

        @num_qubits(meas_behavior=invalid_meas_behavior)
        def main():
            qf = QuantumFloat(1)
            return measure(qf[0])

        with pytest.raises(
            ValueError, match="Measurement behavior must return a boolean, got 42"
        ):
            main()
