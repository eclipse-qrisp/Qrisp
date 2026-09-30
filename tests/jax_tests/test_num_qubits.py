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

"""Tests for the num_qubits resource-counting metric, including control flow and callback_threshold."""

import inspect
import warnings

import pytest

from qrisp import (
    QuantumFloat,
    QuantumVariable,
    control,
    cx,
    gidney_adder,
    h,
    measure,
    num_qubits,
    parity,
    reset,
    x,
)
from qrisp.jasp import expectation_value, jlen, jrange, make_jaspr, profile_jaspr, qache
from qrisp.jasp.interpreter_tools.interpreters.num_qubits_metric import (
    extract_num_qubits,
    get_num_qubits_profiler,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    always_one,
    always_zero,
)
from qrisp.misc.exceptions import QrispDeprecationWarning


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

        expected_entry = num_qubits_input + (num_qubits_input + 1) + (num_qubits_input + 2) + (num_qubits_input + 3)
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
        """Test that if we create an alias to a quantum variable and delete the alias,
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

    def test_num_qubits_reset_does_not_change_counts(self):
        """Test that resetting qubits neither allocates nor deallocates them."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            reset(qv)
            qv.delete()

        num_qubits_input = 3
        expected_dic = {
            "total_allocated": num_qubits_input,
            "total_deallocated": num_qubits_input,
            "peak_allocations": num_qubits_input,
            "finally_allocated": 0,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_with_return_values(self):
        """Test that the counts are extracted correctly when the function returns values."""

        @num_qubits(meas_behavior="1")
        def return_measurement(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            return measure(qv)

        @num_qubits(meas_behavior="1")
        def return_multiple(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            qb = QuantumFloat(1)
            h(qv[0])
            h(qb[0])
            return measure(qv), qb

        num_qubits_input = 3
        assert return_measurement(num_qubits_input) == {
            "total_allocated": num_qubits_input,
            "total_deallocated": 0,
            "peak_allocations": num_qubits_input,
            "finally_allocated": num_qubits_input,
        }
        assert return_multiple(num_qubits_input) == {
            "total_allocated": num_qubits_input + 1,
            "total_deallocated": 0,
            "peak_allocations": num_qubits_input + 1,
            "finally_allocated": num_qubits_input + 1,
        }


class TestNumQubitsRegisterOperations:
    """Test cases for slicing and fusing qubit arrays, whose sizes can determine later allocations."""

    @pytest.mark.parametrize(
        "slicer,expected_size",
        [
            (lambda qa: qa[1:3], 2),
            (lambda qa: qa[:-1], 3),
            (lambda qa: qa[1:-1], 2),
            (lambda qa: qa[-2:], 2),
            (lambda qa: qa[-10:], 4),
            (lambda qa: qa[:-10], 0),
            (lambda qa: qa[:9], 4),
            (lambda qa: qa[6:], 0),
            (lambda qa: qa[3:1], 0),
        ],
    )
    def test_num_qubits_slice_size(self, slicer, expected_size):
        """Test that slice sizes follow Python semantics for negative, out-of-range and empty bounds.

        The slice size is used to allocate a second register, so an incorrect
        size shows up directly in the allocation counts.
        """

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            anc = QuantumVariable(jlen(slicer(qv[:])))
            h(anc[0])

        num_qubits_input = 4
        total = num_qubits_input + expected_size
        expected_dic = {
            "total_allocated": total,
            "total_deallocated": 0,
            "peak_allocations": total,
            "finally_allocated": total,
        }
        assert main(num_qubits_input) == expected_dic

    @pytest.mark.parametrize(
        "fuser,expected_size",
        [
            (lambda qa, qb: qa[:] + [qb[0]], 5),
            (lambda qa, qb: [qb[0]] + qa[:], 5),
            (lambda qa, qb: qa[:] + qb[:], 6),
        ],
    )
    def test_num_qubits_fuse_size(self, fuser, expected_size):
        """Test that fusing counts a single qubit as one qubit and a qubit array by its size."""

        @num_qubits(meas_behavior="0")
        def main(num_qubits_input):
            qa = QuantumFloat(num_qubits_input)
            qb = QuantumFloat(2)
            h(qa[0])
            h(qb[0])
            anc = QuantumVariable(jlen(fuser(qa, qb)))
            h(anc[0])

        num_qubits_input = 4
        total = num_qubits_input + 2 + expected_size
        expected_dic = {
            "total_allocated": total,
            "total_deallocated": 0,
            "peak_allocations": total,
            "finally_allocated": total,
        }
        assert main(num_qubits_input) == expected_dic

    def test_num_qubits_adder_on_negative_slice(self):
        """Test that an adder acting on a slice with a negative bound allocates its ancillas.

        Regression test: the Montgomery reduction used in Shor's algorithm adds
        into ``qf[m:-1]``. With the negative stop taken literally, the slice had
        a negative size and the ancillas of the Gidney adder were never counted.
        """

        @num_qubits(meas_behavior="0")
        def negative_stop(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            gidney_adder(3, qv[:-1])

        @num_qubits(meas_behavior="0")
        def positive_stop(num_qubits_input):
            qv = QuantumFloat(num_qubits_input)
            h(qv[0])
            gidney_adder(3, qv[: num_qubits_input - 1])

        num_qubits_input = 4
        # The adder acts on 3 qubits and uses 2 ancillas, which it deallocates again
        expected_dic = {
            "total_allocated": num_qubits_input + 2,
            "total_deallocated": 2,
            "peak_allocations": num_qubits_input + 2,
            "finally_allocated": num_qubits_input,
        }
        assert negative_stop(num_qubits_input) == expected_dic
        assert positive_stop(num_qubits_input) == expected_dic


class TestNumQubitsControlFlow:
    """Test cases for the num_qubits metric in the presence of control flow,
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

        expected_alloc = num_qubits_input2 if meas_behavior == always_zero else num_qubits_input3
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

    def test_num_qubits_parity_controls_allocation(self):
        """Test that the parity of measurement results is evaluated when it controls an allocation."""

        @num_qubits(meas_behavior="1")
        def main():
            qv = QuantumVariable(2)
            m1 = measure(qv[0])
            m2 = measure(qv[1])

            # Both measurements return 1: the parity of (m1, m2) is 0, the parity of m1 alone is 1
            with control(parity(m1, m2)):
                qv_even = QuantumFloat(3)
                h(qv_even[0])

            with control(parity(m1)):
                qv_odd = QuantumFloat(5)
                h(qv_odd[0])

        expected_dic = {
            "total_allocated": 2 + 5,
            "total_deallocated": 0,
            "peak_allocations": 2 + 5,
            "finally_allocated": 2 + 5,
        }
        assert main() == expected_dic


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

        with pytest.raises(ValueError, match="Measurement behavior must return a boolean, got 42"):
            main()

    def test_num_qubits_kernelization_raises(self):
        """Test that num_qubits on a kernelized function raises NotImplementedError."""

        def state_prep():
            qf = QuantumFloat(3)
            h(qf)
            return qf

        @num_qubits(meas_behavior="0")
        def main():
            return expectation_value(state_prep, 10)()

        with pytest.raises(
            NotImplementedError,
            match="Quantum kernel creation not yet supported in profiling interpreter",
        ):
            main()

    def test_num_qubits_many_allocations(self):
        """Test that the number of allocation/deallocation events is not bounded."""

        @num_qubits(meas_behavior="0")
        def main(num_iterations):
            for _ in jrange(num_iterations):
                qv = QuantumFloat(2)
                h(qv[0])
                qv.delete()

        num_iterations = 5000
        expected_dic = {
            "total_allocated": 2 * num_iterations,
            "total_deallocated": 2 * num_iterations,
            "peak_allocations": 2,
            "finally_allocated": 0,
        }
        assert main(num_iterations) == expected_dic


def _three_single_qubits():
    """Allocate three single qubits, more than the ``max_allocations=2`` used below."""
    qv1 = QuantumFloat(1)
    h(qv1[0])
    qv2 = QuantumFloat(1)
    h(qv2[0])
    qv3 = QuantumFloat(1)
    h(qv3[0])


_THREE_SINGLE_QUBITS_DIC = {
    "total_allocated": 3,
    "total_deallocated": 0,
    "peak_allocations": 3,
    "finally_allocated": 3,
}


class TestNumQubitsMaxAllocationsDeprecation:
    """Test cases for the deprecated ``max_allocations`` argument, which is accepted but ignored."""

    def test_decorator_keyword_warns(self):
        """Test that passing ``max_allocations`` by keyword warns and does not limit the computation."""
        with pytest.warns(QrispDeprecationWarning, match="max_allocations"):
            decorator = num_qubits(meas_behavior="0", max_allocations=2)

        assert decorator(_three_single_qubits)() == _THREE_SINGLE_QUBITS_DIC

    def test_decorator_positional_warns(self):
        """Test that passing ``max_allocations`` positionally, as the second argument, still warns."""
        with pytest.warns(QrispDeprecationWarning, match="max_allocations"):
            decorator = num_qubits("0", 2)

        assert decorator(_three_single_qubits)() == _THREE_SINGLE_QUBITS_DIC

    def test_decorator_without_max_allocations_does_not_warn(self):
        """Test that the decorator does not warn when ``max_allocations`` is omitted."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", QrispDeprecationWarning)
            assert num_qubits(meas_behavior="0")(_three_single_qubits)() == _THREE_SINGLE_QUBITS_DIC

    def test_jaspr_method_warns(self):
        """Test that ``Jaspr.num_qubits`` warns about ``max_allocations`` and only then."""
        jaspr = make_jaspr(_three_single_qubits)()

        with pytest.warns(QrispDeprecationWarning, match="max_allocations"):
            assert jaspr.num_qubits(meas_behavior="0", max_allocations=2) == _THREE_SINGLE_QUBITS_DIC

        with warnings.catch_warnings():
            warnings.simplefilter("error", QrispDeprecationWarning)
            assert jaspr.num_qubits(meas_behavior="0") == _THREE_SINGLE_QUBITS_DIC

    def test_internal_profiler_accepts_max_allocations(self):
        """Test that the internal entry points still accept ``max_allocations`` and ignore it.

        ``get_num_qubits_profiler`` keeps ``max_allocations`` as its third parameter,
        so a positional value is not mistaken for ``callback_threshold``.
        """
        jaspr = make_jaspr(_three_single_qubits)()

        assert profile_jaspr(jaspr, "num_qubits", "0", max_allocations=2)() == _THREE_SINGLE_QUBITS_DIC

        parameters = list(inspect.signature(get_num_qubits_profiler).parameters)
        assert parameters == ["jaspr", "meas_behavior", "max_allocations", "callback_threshold"]

        profiler, aux = get_num_qubits_profiler(jaspr, always_zero, 2)
        assert extract_num_qubits(profiler(), jaspr, aux) == _THREE_SINGLE_QUBITS_DIC


def test_callback_threshold_num_qubits():
    """Test that num_qubits with callback_threshold produces correct results.

    The callback_threshold parameter controls whether ``jax.pure_callback``
    wrapping is used for reused sub-jaxprs to prevent XLA compilation blowup.
    Different threshold values should all produce the same profiling results.
    """

    # A qache'd subroutine that allocates/deallocates qubits.
    @qache
    def qubit_allocating_sub(qv):
        """A subroutine with its own qubit allocation."""
        inner = QuantumFloat(2)
        h(inner[0])
        cx(qv[0], inner[0])
        h(inner[1])
        inner.delete()

    def make_circuit():
        qv = QuantumFloat(3)
        h(qv[0])

        # Call multiple times for reuse
        qubit_allocating_sub(qv)
        qubit_allocating_sub(qv)
        qubit_allocating_sub(qv)

        x(qv)
        qv.delete()

    # Baseline: no callbacks
    baseline = num_qubits(meas_behavior="0")(make_circuit)()
    # 3 qubits, plus 2 temporary qubits allocated and deleted by each of the three calls
    assert baseline == {
        "total_allocated": 3 + 3 * 2,
        "total_deallocated": 3 * 2 + 3,
        "peak_allocations": 3 + 2,
        "finally_allocated": 0,
    }

    # callback_threshold=0: wrap every reused sub-jaxpr
    result_0 = num_qubits(meas_behavior="0", callback_threshold=0)(make_circuit)()
    assert result_0 == baseline, f"callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"

    # callback_threshold=500: middle ground
    result_500 = num_qubits(meas_behavior="0", callback_threshold=500)(make_circuit)()
    assert result_500 == baseline, f"callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"

    # Very large threshold
    result_large = num_qubits(meas_behavior="0", callback_threshold=10**9)(make_circuit)()
    assert result_large == baseline, f"callback_threshold=10**9 diverged:\n  baseline={baseline}\n  got={result_large}"

    # Also test with meas_behavior="1"
    baseline_1 = num_qubits(meas_behavior="1")(make_circuit)()
    result_1_0 = num_qubits(meas_behavior="1", callback_threshold=0)(make_circuit)()
    assert result_1_0 == baseline_1, (
        f"meas_behavior='1', callback_threshold=0 diverged:\n  baseline={baseline_1}\n  got={result_1_0}"
    )

    # Verify the result dictionary has expected keys
    for key in ("total_allocated", "total_deallocated", "peak_allocations", "finally_allocated"):
        assert key in baseline, f"Expected '{key}' in num_qubits result, got {baseline}"


def test_callback_threshold_num_qubits_nested():
    """Test callback_threshold with nested @qache'd subroutines for num_qubits.

    An outer qache'd function calls an inner qache'd function that allocates
    its own qubits.  Both are reused.  Quota tracking must remain correct
    regardless of callback wrapping.
    """

    @qache
    def inner_alloc_sub(qv):
        """Inner subroutine: allocates a temporary qubit, uses it, deletes it."""
        tmp = QuantumFloat(2)
        h(tmp[0])
        cx(qv[0], tmp[0])
        h(tmp[1])
        tmp.delete()

    @qache
    def outer_alloc_sub(qv):
        """Outer subroutine: calls inner_alloc_sub and allocates its own temp."""
        tmp = QuantumFloat(1)
        h(tmp[0])
        inner_alloc_sub(qv)
        inner_alloc_sub(qv)
        tmp.delete()

    def make_circuit():
        qv = QuantumFloat(3)
        h(qv[0])

        outer_alloc_sub(qv)
        outer_alloc_sub(qv)
        outer_alloc_sub(qv)

        x(qv)
        qv.delete()

    baseline = num_qubits(meas_behavior="0")(make_circuit)()
    # Each outer call allocates 1 qubit and calls the inner subroutine twice (2 qubits each)
    assert baseline == {
        "total_allocated": 3 + 3 * (1 + 2 * 2),
        "total_deallocated": 3 * (1 + 2 * 2) + 3,
        "peak_allocations": 3 + 1 + 2,
        "finally_allocated": 0,
    }

    result_0 = num_qubits(meas_behavior="0", callback_threshold=0)(make_circuit)()
    assert result_0 == baseline, (
        f"Nested qache num_qubits callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"
    )

    result_500 = num_qubits(meas_behavior="0", callback_threshold=500)(make_circuit)()
    assert result_500 == baseline, (
        f"Nested qache num_qubits callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"
    )

    # threshold=1 edge case
    result_1 = num_qubits(meas_behavior="0", callback_threshold=1)(make_circuit)()
    assert result_1 == baseline, (
        f"Nested qache num_qubits callback_threshold=1 diverged:\n  baseline={baseline}\n  got={result_1}"
    )

    for key in ("total_allocated", "total_deallocated", "peak_allocations", "finally_allocated"):
        assert key in baseline, f"Expected '{key}' in num_qubits result, got {baseline}"


def test_callback_threshold_num_qubits_jrange():
    """Test callback_threshold with a jrange loop + qache'd subroutine for num_qubits.

    The jrange loop calls a subroutine that allocates/deallocates qubits
    many times, creating many call sites for callback wrapping optimization.
    """

    @qache
    def alloc_loop_sub(qv, n):
        """Subroutine with its own allocation, called inside jrange loop."""
        tmp = QuantumFloat(2)
        h(tmp[0])
        cx(qv[n % 2], tmp[0])
        h(tmp[1])
        tmp.delete()

    def make_circuit():
        qv = QuantumFloat(3)
        h(qv[0])

        for i in jrange(10):
            alloc_loop_sub(qv, i)

        x(qv)
        qv.delete()

    baseline = num_qubits(meas_behavior="0")(make_circuit)()
    # 3 qubits, plus 2 temporary qubits allocated and deleted in each of the 10 iterations
    assert baseline == {
        "total_allocated": 3 + 10 * 2,
        "total_deallocated": 10 * 2 + 3,
        "peak_allocations": 3 + 2,
        "finally_allocated": 0,
    }

    result_0 = num_qubits(meas_behavior="0", callback_threshold=0)(make_circuit)()
    assert result_0 == baseline, (
        f"jrange num_qubits callback_threshold=0 diverged:\n  baseline={baseline}\n  got={result_0}"
    )

    result_500 = num_qubits(meas_behavior="0", callback_threshold=500)(make_circuit)()
    assert result_500 == baseline, (
        f"jrange num_qubits callback_threshold=500 diverged:\n  baseline={baseline}\n  got={result_500}"
    )

    for key in ("total_allocated", "total_deallocated", "peak_allocations", "finally_allocated"):
        assert key in baseline, f"Expected '{key}' in num_qubits result, got {baseline}"
