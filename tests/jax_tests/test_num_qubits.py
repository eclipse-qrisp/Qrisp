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

        expected_dic = {
            "total_allocated": num_qubits_input,
            "total_deallocated": 0,
            "peak_allocations": num_qubits_input,
            "finally_allocated": num_qubits_input,
        }
        assert main(num_qubits_input) == expected_dic

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

        expected_entry = (
            num_qubits_input
            + (num_qubits_input + 1)
            + (num_qubits_input + 2)
            + (num_qubits_input + 3)
        )
        expected_dic = {
            "total_allocated": expected_entry,
            "total_deallocated": 0,
            "peak_allocations": expected_entry,
            "finally_allocated": expected_entry,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_delete_qubits(self):
        """Test qubit counting with qubit deletion."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            qv.delete()

        num_qubits_input = 4
        expected_dic = {
            "total_allocated": num_qubits_input,
            "total_deallocated": num_qubits_input,
            "peak_allocations": num_qubits_input,
            "finally_allocated": 0,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_delete_qubits2(self):
        """Test qubit counting with qubit deletion followed by reallocation."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            qv.delete()
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])

        num_qubits_input = 4
        expected_dic = {
            "total_allocated": num_qubits_input * 2,
            "total_deallocated": num_qubits_input,
            "peak_allocations": num_qubits_input,
            "finally_allocated": num_qubits_input,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_all_deleted_in_end(self):
        """Everything deleted before termination => final count is 0."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            a = QuantumFloat(num_qubits_input)
            b = QuantumFloat(num_qubits_input + 1)
            h(a[0])
            h(b[0])
            a.delete()
            b.delete()

        num_qubits_input = 4
        expected_dic = {
            "total_allocated": num_qubits_input + (num_qubits_input + 1),
            "total_deallocated": num_qubits_input + (num_qubits_input + 1),
            "peak_allocations": num_qubits_input + (num_qubits_input + 1),
            "finally_allocated": 0,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_unused_allocation_semantics(self):
        """Test the expected behavior for qubits that are allocated but never used (e.g., no gates, no measurements)."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            _qv = QuantumFloat(num_qubits_input)

        num_qubits_input = 4
        expected_dic = {
            "total_allocated": 0,
            "total_deallocated": 0,
            "peak_allocations": 0,
            "finally_allocated": 0,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_alias_delete_counts_once(self):
        """
        Test that if we create an alias to a quantum variable and delete the alias,
        it only counts as one deletion.
        """

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            alias = qv
            h(qv[0])
            alias.delete()

        num_qubits_input = 4
        expected_dic = {
            "total_allocated": num_qubits_input,
            "total_deallocated": num_qubits_input,
            "peak_allocations": num_qubits_input,
            "finally_allocated": 0,
        }
        assert main(num_qubits_input) == expected_dic


class TestNumQubitsControlFlow:
    """
    Test cases for the num_qubits metric in the presence of control flow,
    possibly with qubit allocation and deletion, and with control flow
    based on measurement outcomes.
    """

    @pytest.mark.parametrize(
        "meas_behavior,num_qubits_input,num_qubits_input2,num_qubits_input3",
        [(always_zero, 2, 3, 4), (always_one, 2, 3, 4)],
    )
    def test_num_qubits_control_flow(
        self,
        meas_behavior,
        num_qubits_input,
        num_qubits_input2,
        num_qubits_input3,
    ):
        """Test qubit counting with control flow based on measurement outcomes."""

        @num_qubits(meas_behavior=meas_behavior)
        def main(num_qubits_input, num_qubits_input2, num_qubits_input3):
            qv = QuantumFloat(num_qubits_input)
            m = measure(qv[0])
            with control(m == 0):
                qv2 = QuantumFloat(num_qubits_input2)
                h(qv2[0])
            with control(m == 1):
                qv3 = QuantumFloat(num_qubits_input3)
                h(qv3[0])

        expected_alloc = (
            num_qubits_input2 if meas_behavior == always_zero else num_qubits_input3
        )
        peak_alloc = num_qubits_input + expected_alloc
        expected_dic = {
            "total_allocated": peak_alloc,
            "total_deallocated": 0,
            "peak_allocations": peak_alloc,
            "finally_allocated": peak_alloc,
        }
        res = main(num_qubits_input, num_qubits_input2, num_qubits_input3)
        assert res == expected_dic

    def test_num_qubits_loop(self):
        """Test qubit counting in a loop structure."""

        @num_qubits(meas_behavior="0")
        def circuit_loop(num_qubits_input, num_iterations):
            for i in jrange(num_iterations):
                qv = QuantumFloat(num_qubits_input)
                h(qv[i])

        num_qubits_input = 5
        num_iterations = 5
        expected_alloc = num_qubits_input * num_iterations
        expected_dic = {
            "total_allocated": expected_alloc,
            "total_deallocated": 0,
            "peak_allocations": expected_alloc,
            "finally_allocated": expected_alloc,
        }
        res = circuit_loop(num_qubits_input, num_iterations)
        assert res == expected_dic

    def test_delete_qubits_in_loop(self):
        """Test qubit counting with qubit deletion inside a loop."""

        num_iterations = 4

        @num_qubits(meas_behavior="1")
        def main(num_qubits_input):

            list_of_qvs1 = []
            list_of_qvs2 = []

            for i in range(num_iterations):
                qv_1 = QuantumFloat(num_qubits_input)
                h(qv_1[i])
                list_of_qvs1.append(qv_1)

            qv_2 = QuantumFloat(1)
            h(qv_2[0])
            m = measure(qv_2[0])

            qv_2.delete()

            with control(m == 0):
                # does not matter the size as
                # it should never be called
                qv3 = QuantumFloat(1000000)
                h(qv3[0])
                list_of_qvs2.append(qv3)

            with control(m == 1):
                qv4 = QuantumFloat(10)
                h(qv4[0])
                list_of_qvs2.append(qv4)

            # If we try to delete list_of_qvs2 here,
            # we get an Exception `ControlEnvironment with carry value`

            for i in range(num_iterations):
                list_of_qvs1[i].delete()

        # expected_allocations = {
        #     # the 4 allocations from the loop
        #     "alloc1": 5,
        #     "alloc2": 5,
        #     "alloc3": 5,
        #     "alloc4": 5,
        #     # the allocation before the control flow
        #     "alloc5": 1,
        #     # the deletion of qv_2
        #     "alloc6": -1,
        #     # the allocation of qv4 in the control flow (since m == 1)
        #     "alloc7": 10,
        #     # the 4 deletions from the loop
        #     "alloc8": -5,
        #     "alloc9": -5,
        #     "alloc10": -5,
        #     "alloc11": -5,
        # }

        num_qubits_input = 5
        expected_dic = {
            "total_allocated": num_qubits_input * num_iterations + 1 + 10,
            "total_deallocated": num_qubits_input * num_iterations + 1,
            "peak_allocations": num_qubits_input * num_iterations + 10,
            "finally_allocated": 10,
        }
        res = main(num_qubits_input)
        assert res == expected_dic

    def test_num_qubits_branch_dependent_delete(self):
        """Test qubit counting when deletion is performed in a measurement-dependent branch."""

        @num_qubits(meas_behavior=always_zero)
        def circuit_branch_del(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            qv_big = QuantumFloat(2 * num_qubits_input)
            m = measure(qv[0])

            with control(m == 0):
                qv_big.delete()

            h(qv[0])

        num_qubits_input = 4
        expected_dic = {
            "total_allocated": num_qubits_input + 2 * num_qubits_input,
            "total_deallocated": 2 * num_qubits_input,
            "peak_allocations": num_qubits_input + 2 * num_qubits_input,
            "finally_allocated": num_qubits_input,
        }
        res = circuit_branch_del(num_qubits_input)
        assert res == expected_dic

        @num_qubits(meas_behavior=always_one)
        def circuit_branch_del2(n):
            qv = QuantumFloat(n)
            qv_big = QuantumFloat(2 * n)
            m = measure(qv[0])

            with control(m == 0):
                qv_big.delete()

            h(qv[0])

        expected_dic = {
            "total_allocated": num_qubits_input + 2 * num_qubits_input,
            "total_deallocated": 0,
            "peak_allocations": num_qubits_input + 2 * num_qubits_input,
            "finally_allocated": num_qubits_input + 2 * num_qubits_input,
        }
        res = circuit_branch_del2(num_qubits_input)
        assert res == expected_dic


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

    def test_num_qubits_overflow1(self):
        """Test that exceeding the maximum number of allocations raises an error."""

        @num_qubits(meas_behavior="0", max_allocations=2)
        def main():
            qv1 = QuantumFloat(1)
            h(qv1[0])
            qv2 = QuantumFloat(1)
            h(qv2[0])
            qv3 = QuantumFloat(1)  # This allocation should trigger the overflow
            h(qv3[0])

        with pytest.raises(
            ValueError, match="The ``num_qubits`` metric computation overflowed"
        ):
            main()

    def test_num_qubits_overflow2(self):
        """Test that exceeding the maximum number of allocations raises an error."""

        @num_qubits(meas_behavior="0", max_allocations=2)
        def main():
            qv1 = QuantumFloat(1)
            h(qv1[0])
            qv2 = QuantumFloat(1)
            h(qv2[0])
            qv2.delete()  # This should prevent the overflow since it frees up one allocation

        with pytest.raises(
            ValueError, match="The ``num_qubits`` metric computation overflowed"
        ):
            main()
