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

import numpy as np
import pytest

from qrisp import QuantumArray, QuantumBool, QuantumVariable, cx, h, measure, x, z
from qrisp.environments.layered_enviroment import GateStack, LayeredEnvironment


def _filter_instructions(qs_data):
    """Strip qb_alloc / qb_dealloc bookkeeping instructions."""
    return [
        instr
        for instr in qs_data
        if not str(instr).startswith(("qb_alloc", "qb_dealloc"))
    ]


def _gate_names(qs_data):
    """Return the string representation of each filtered instruction."""
    return [str(i) for i in _filter_instructions(qs_data)]


class TestLayeredEnvironmentBasicUsage:
    """Test basic usage of LayeredEnvironment and GateStack."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_basic_stabilizer_measurement(self, n):
        """Test that instructions are correctly layered and emitted for a simple stabilizer measurement."""

        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1))
        data_qubits = qubits[::2]
        ancilla_qubits = qubits[1::2]

        for i in range(n - 1):
            cx(data_qubits[i], ancilla_qubits[i])
            cx(data_qubits[i + 1], ancilla_qubits[i])

        expected_result = qubits.qs.statevector_array()

        qubits.qs.clear_data()

        with LayeredEnvironment():
            for i in range(n - 1):
                with GateStack():
                    cx(data_qubits[i], ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])

        result = qubits.qs.statevector_array()

        assert np.allclose(result, expected_result)
        assert qubits.qs.depth() == 2

    def test_parallel_single_qubit_gates(self):
        """
        Test that single-qubit gates on different qubits are correctly layered in parallel.
        """
        n = 6
        qv = QuantumVariable(n)
        with LayeredEnvironment():
            for i in range(n):
                with GateStack():
                    x(qv[i])

        names = _gate_names(qv.qs.data)

        assert len(names) == n
        for i in range(n):
            assert names[i] == f"x(Qubit(qv.{i}))"

    def test_one_deep_one_shallow_stack(self):
        """
        Test that if one stack has fewer layers than another, the layering still works correctly
        and the shorter stack's gates are emitted in the correct order relative to the longer stack.
        """
        qv = QuantumVariable(6)
        with LayeredEnvironment():
            with GateStack():
                x(qv[0])
                x(qv[1])
                x(qv[2])
            with GateStack():
                x(qv[3])
            with GateStack():
                x(qv[4])
                x(qv[5])

        names = _gate_names(qv.qs.data)

        assert len(names) == 6
        assert names[0] == "x(Qubit(qv.0))"
        assert names[1] == "x(Qubit(qv.3))"
        assert names[2] == "x(Qubit(qv.4))"
        assert names[3] == "x(Qubit(qv.1))"
        assert names[4] == "x(Qubit(qv.5))"
        assert names[5] == "x(Qubit(qv.2))"

    def test_nested_gate_stacks_not_supported(self):
        """
        Test that attempting to nest a GateStack inside another GateStack raises a ValueError.
        """
        qv = QuantumVariable(2)

        with pytest.raises(
            ValueError, match="Nested QuantumEnvironments are not supported"
        ):
            with LayeredEnvironment():
                x(qv[0])
                with GateStack():
                    x(qv[0])
                    with GateStack():
                        x(qv[1])

    def test_nested_environments_not_supported(self):
        """
        Test that attempting to nest a LayeredEnvironment inside another LayeredEnvironment raises a ValueError.
        """
        qv = QuantumVariable(2)

        with pytest.raises(
            ValueError, match="Nested QuantumEnvironments are not supported"
        ):
            with LayeredEnvironment():
                x(qv[0])
                with LayeredEnvironment():
                    x(qv[1])

    def test_nested_gate_stack_and_environment_not_supported(self):
        """
        Test that attempting to nest a GateStack inside a LayeredEnvironment raises a ValueError.
        """
        qv = QuantumVariable(2)

        with pytest.raises(
            ValueError, match="Nested QuantumEnvironments are not supported"
        ):
            with LayeredEnvironment():
                x(qv[0])
                with GateStack():
                    x(qv[1])
                    with LayeredEnvironment():
                        x(qv[0])

    def test_no_gate_stacks_first(self):
        """
        Test that a GateStack must be called within a LayeredEnvironment"""
        qv = QuantumVariable(2)

        with pytest.raises(
            ValueError,
            match="GateStack must have a parent LayeredEnvironment to compile",
        ):
            with GateStack():
                x(qv[0])


class TestEmptyCases:
    """Edge cases involving empty or trivial environments."""

    def test_empty_layered_environment(self):
        """
        Test that an empty LayeredEnvironment (with no GateStacks and no gates) produces no instructions.
        """
        qv = QuantumVariable(2)
        with LayeredEnvironment():
            pass

        assert _filter_instructions(qv.qs.data) == []

    def test_single_empty_gate_stack(self):
        """
        Test that a single empty GateStack inside a LayeredEnvironment produces no instructions.
        """
        qv = QuantumVariable(2)
        with LayeredEnvironment():
            with GateStack():
                pass

        assert _filter_instructions(qv.qs.data) == []

    def test_multiple_empty_gate_stacks(self):
        """
        Test that multiple empty GateStacks inside a LayeredEnvironment produce no instructions.
        """
        qv = QuantumVariable(2)
        with LayeredEnvironment():
            for _ in range(4):
                with GateStack():
                    pass

        assert _filter_instructions(qv.qs.data) == []

    def test_empty_gate_stack_among_non_empty(self):
        """
        Test that an empty GateStack among non-empty GateStacks does not produce any instructions
        and does not affect the layering of the non-empty stacks.
        """

        n = 5
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1))
        data_qubits = qubits[::2]
        ancilla_qubits = qubits[1::2]

        for i in range(n - 1):
            cx(data_qubits[i], ancilla_qubits[i])
            cx(data_qubits[i + 1], ancilla_qubits[i])

        expected_result = qubits.qs.statevector_array()

        qubits.qs.clear_data()

        with LayeredEnvironment():

            for i in range((n - 1) // 2):
                with GateStack():
                    cx(data_qubits[i], ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])

            with GateStack():
                pass

            for i in range((n - 1) // 2, n - 1):
                with GateStack():
                    cx(data_qubits[i], ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])

        result = qubits.qs.statevector_array()

        assert np.allclose(result, expected_result)
        assert qubits.qs.depth() == 2


class TestInstructionsOutsideGateStack:
    """
    Gates emitted directly inside LayeredEnvironment (not wrapped in a
    GateStack) should be emitted in their correct relative position.
    """

    def test_bare_gate_before_stacks(self):
        """
        A gate emitted before any GateStack should appear before the
        GateStack's instructions.
        """
        qv = QuantumVariable(4)
        with LayeredEnvironment():
            x(qv[0])
            with GateStack():
                x(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 2
        assert names[0] == "x(Qubit(qv.0))"
        assert names[1] == "x(Qubit(qv.1))"

    def test_bare_gate_after_stacks(self):
        """
        Test that a gate emitted after a GateStack appears after the GateStack's instructions.
        """
        qv = QuantumVariable(4)
        with LayeredEnvironment():
            with GateStack():
                x(qv[0])
            x(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 2
        assert names[0] == "x(Qubit(qv.0))"
        assert names[1] == "x(Qubit(qv.1))"

    def test_bare_gates_between_stacks(self):
        """
        Test that gates emitted between two GateStacks appear in the correct position between the stacks' instructions.
        """
        qv = QuantumVariable(6)
        with LayeredEnvironment():
            with GateStack():
                x(qv[0])
            x(qv[1])
            with GateStack():
                x(qv[2])

        names = _gate_names(qv.qs.data)

        assert len(names) == 3
        assert names[0] == "x(Qubit(qv.0))"
        assert names[1] == "x(Qubit(qv.1))"
        assert names[2] == "x(Qubit(qv.2))"

    def test_bare_gates_between_stacks_2(self):
        """
        Test that multiple gates emitted between two GateStacks appear in the correct position between the stacks' instructions.
        """
        n = 4
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * n - 1,))
        data_qubits = qubits[::2]
        ancilla_qubits = qubits[1::2]

        cx(data_qubits[0], ancilla_qubits[0])
        cx(data_qubits[1], ancilla_qubits[0])

        cx(data_qubits[1], ancilla_qubits[1])
        cx(data_qubits[2], ancilla_qubits[1])

        z(data_qubits[2])

        cx(data_qubits[2], ancilla_qubits[2])
        cx(data_qubits[3], ancilla_qubits[2])

        expected_result = qubits.qs.statevector_array()

        qubits.qs.clear_data()

        with LayeredEnvironment():

            with GateStack():
                cx(data_qubits[0], ancilla_qubits[0])
                cx(data_qubits[1], ancilla_qubits[0])

            z(data_qubits[2])

            with GateStack():
                cx(data_qubits[1], ancilla_qubits[1])
                cx(data_qubits[2], ancilla_qubits[1])

            with GateStack():
                cx(data_qubits[2], ancilla_qubits[2])
                cx(data_qubits[3], ancilla_qubits[2])

        result = qubits.qs.statevector_array()

        assert np.allclose(result, expected_result)

        names = _gate_names(qubits.qs.data)

        assert len(names) == 7
        assert names[0] == "cx(Qubit(qubits.0), Qubit(qubits_1.0))"
        assert names[1] == "cx(Qubit(qubits_2.0), Qubit(qubits_1.0))"
        assert names[2] == "z(Qubit(qubits_4.0))"
        assert names[3] == "cx(Qubit(qubits_2.0), Qubit(qubits_3.0))"
        assert names[4] == "cx(Qubit(qubits_4.0), Qubit(qubits_5.0))"
        assert names[5] == "cx(Qubit(qubits_4.0), Qubit(qubits_3.0))"
        assert names[6] == "cx(Qubit(qubits_6.0), Qubit(qubits_5.0))"

        assert qubits.qs.depth() == 4

    def test_no_gate_stacks_at_all(self):
        """
        Test that if no GateStacks are used, all gates are emitted in the original order.
        """
        qv = QuantumVariable(3)
        with LayeredEnvironment():
            x(qv[0])
            h(qv[1])
            x(qv[2])

        names = _gate_names(qv.qs.data)

        assert len(names) == 3
        assert names[0] == "x(Qubit(qv.0))"
        assert names[1] == "h(Qubit(qv.1))"
        assert names[2] == "x(Qubit(qv.2))"


class TestMeasurements:
    """
    Test that measurement instructions are emitted in the correct order and layering relative to gates.
    """

    def test_measurement_only_single_stack(self):
        """
        Test that a single GateStack containing only measurement instructions is emitted correctly.
        """
        qv = QuantumVariable(2)
        with LayeredEnvironment():
            with GateStack():
                measure(qv[0])
                measure(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 2
        assert names[0] == "measure(Qubit(qv.0), Clbit(clbit_0))"
        assert names[1] == "measure(Qubit(qv.1), Clbit(cb_1))"

    def test_measurement_only_two_stacks_interleaved(self):
        """
        Test that two GateStacks each containing only measurement instructions are emitted in the correct interleaved order.
        """
        qv = QuantumVariable(4)
        with LayeredEnvironment():
            with GateStack():
                measure(qv[0])
            with GateStack():
                measure(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 2
        assert names[0] == "measure(Qubit(qv.0), Clbit(clbit_0))"
        assert names[1] == "measure(Qubit(qv.1), Clbit(clbit_1))"

    def test_measure_then_gate_in_same_stack(self):
        """
        Test that if a measurement is followed by a gate on the same qubit within the same GateStack, the measurement is emitted before the gate.
        """
        qv = QuantumVariable(2)
        with LayeredEnvironment():
            with GateStack():
                measure(qv[0])
                x(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 2
        assert names[0] == "measure(Qubit(qv.0), Clbit(clbit_0))"
        assert names[1] == "x(Qubit(qv.1))"

    def test_interleaved_measure_and_gate_across_stacks(self):
        """
        Test that if measurements and gates are interleaved across different GateStacks, the layering is still correct.
        """
        qv = QuantumVariable(4)

        with LayeredEnvironment():
            with GateStack():
                measure(qv[0])
                x(qv[0])
            with GateStack():
                measure(qv[1])
                x(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 4
        assert names[0] == "measure(Qubit(qv.0), Clbit(clbit_0))"
        assert names[1] == "measure(Qubit(qv.1), Clbit(clbit_1))"
        assert names[2] == "x(Qubit(qv.0))"
        assert names[3] == "x(Qubit(qv.1))"

    def test_single_stack_gate_measure_gate(self):
        """
        Test that if a single GateStack contains a gate, then a measurement, then another gate, the instructions are emitted in the correct order.
        """
        qv = QuantumVariable(2)
        with LayeredEnvironment():
            with GateStack():
                h(qv[0])
                measure(qv[0])
                x(qv[1])

        names = _gate_names(qv.qs.data)

        assert len(names) == 3
        assert names[0] == "h(Qubit(qv.0))"
        assert names[1] == "measure(Qubit(qv.0), Clbit(cb_0))"
        assert names[2] == "x(Qubit(qv.1))"
