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

from qrisp import QuantumFloat, control, cx, depth, h, mcx, measure
from qrisp.jasp import q_cond, jrange


class TestDepthQuantumPrimitiveSingleQubit:
    """Test that the depth is correctly computed for single-qubit quantum primitives."""

    def test_one_qubit_h_static(self):
        """Test depth of a single Hadamard gate on one qubit (static case)."""

        @depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            h(qf[0])
            return measure(qf[0])

        assert main() == 1

    def test_one_qubit_h(self):
        """Test depth of a single Hadamard gate on one qubit (dynamic case)."""

        @depth(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)
            h(qf[0])
            return measure(qf[0])

        assert main(1) == 1

    @pytest.mark.parametrize(
        "num_qubits,qubit_indices,expected_depth",
        [
            (2, [0, 0], 2),
            (2, [0, 1], 1),
            (2, [1, 0], 1),
            (2, [1, 1], 2),
        ],
    )
    def test_two_qubit_h(self, num_qubits, qubit_indices, expected_depth):
        """Test depth of Hadamard gates on various 2-qubit configurations."""

        @depth(meas_behavior="0")
        def main(num_qubits, qubit_indices):
            qf = QuantumFloat(num_qubits)
            h(qf[qubit_indices[0]])
            h(qf[qubit_indices[1]])
            return measure(qf[qubit_indices[-1]])

        assert main(num_qubits, qubit_indices) == expected_depth

    @pytest.mark.parametrize(
        "qubit_indices,expected_depth",
        [
            ([0, 1, 2], 1),  # parallel
            ([0, 0, 0], 3),  # sequential
        ],
    )
    def test_three_qubit_h(self, qubit_indices, expected_depth):
        """Test depth of 3 Hadamard gates in parallel vs sequential configuration."""

        @depth(meas_behavior="0")
        def main(num_qubits, qubit_indices):
            qf = QuantumFloat(num_qubits)
            h(qf[qubit_indices[0]])
            h(qf[qubit_indices[1]])
            h(qf[qubit_indices[2]])
            return measure(qf[0])

        assert main(3, qubit_indices) == expected_depth

    def test_three_qubit_h_parallel_and_sequential(self):
        """Test depth of 3 Hadamard gates applied sequentially on each of 3 qubits."""

        @depth(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)
            h(qf[0])
            h(qf[0])
            h(qf[0])

            h(qf[1])
            h(qf[1])
            h(qf[1])

            h(qf[2])
            h(qf[2])
            h(qf[2])
            return measure(qf[0])

        assert main(3) == 3


class TestDepthQuantumPrimitiveMultiQubit:
    """Test that the depth is correctly computed for multi-qubit quantum primitives."""

    def test_two_qubit_cx(self):
        """Test depth of a single CNOT gate on two qubits."""

        @depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(2)
            cx(qf[0], qf[1])
            return measure(qf[0])

        assert main() == 1

    def test_three_qubit_mcx(self):
        """Test depth of a single Toffoli gate on three qubits."""

        @depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(3)
            mcx([qf[0], qf[1]], qf[2])
            return measure(qf[0])

        assert main() == 1

    def test_h_cx_combination_1(self):
        """Test depth of a combination of Hadamard and CNOT gates."""

        @depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(2)
            h(qf[0])
            cx(qf[0], qf[1])
            h(qf[1])
            return measure(qf[0])

        assert main() == 3

    def test_h_cx_combination_2(self):
        """Test depth of a combination of Hadamard, Toffoli, and CNOT gates."""

        @depth(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)
            h(qf[0])
            h(qf[2])
            mcx([qf[0], qf[1]], qf[2])
            h(qf[1])
            h(qf[1])
            cx(qf[2], qf[3])
            return measure(qf[0])

        assert main(4) == 4


class TestDepthControlStructures:
    """Test that the depth is correctly computed for control structures."""

    @pytest.mark.parametrize("selector", [0, -1])
    def test_conditional_1(self, selector):
        """Test depth computation in a conditional structure with q_cond."""

        @depth(meas_behavior="0")
        def main(num_qubits, selector):
            qf = QuantumFloat(num_qubits)

            def true_fn():
                h(qf[0])
                h(qf[2])
                cx(qf[0], qf[1])
                h(qf[1])

            def false_fn():
                h(qf[0])
                h(qf[2])
                mcx([qf[0], qf[1]], qf[2])
                h(qf[1])
                h(qf[1])
                cx(qf[2], qf[3])

            q_cond(selector >= 0, true_fn, false_fn)

            return measure(qf[0])

        expected_depth = 3 if selector >= 0 else 4
        assert main(4, selector) == expected_depth

    @pytest.mark.parametrize("selector", [0, -1])
    def test_conditional_2(self, selector):
        """Test depth computation in a conditional structure with control context."""

        @depth(meas_behavior="0")
        def main(num_qubits, selector):
            qf = QuantumFloat(num_qubits)

            with control(selector >= 0):
                h(qf[0])
                h(qf[2])
                cx(qf[0], qf[1])
                h(qf[1])
            with control(selector < 0):
                h(qf[0])
                h(qf[2])
                mcx([qf[0], qf[1]], qf[2])
                h(qf[1])
                h(qf[1])
                cx(qf[2], qf[3])

            return measure(qf[0])

        expected_depth = 3 if selector >= 0 else 4
        assert main(4, selector) == expected_depth

    def test_loop_h_sequential(self):
        """Test depth computation in a loop with sequential Hadamard gates."""

        @depth(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)

            for _ in jrange(qf.size):
                h(qf[0])

            return measure(qf[0])

        assert main(4) == 4

    def test_loop_h_parallel(self):
        """Test depth computation in a loop with parallel Hadamard gates."""

        @depth(meas_behavior="0")
        def main(num_qubits):
            qf = QuantumFloat(num_qubits)

            for i in jrange(qf.size):
                h(qf[i])

            return measure(qf[0])

        assert main(5) == 1


def test_depth_simulation_not_implemented():
    """Test that depth via simulation raises NotImplementedError."""

    @depth(meas_behavior="sim")
    def main():
        pass

    with pytest.raises(
        NotImplementedError, match="Depth metric via simulation is not implemented yet"
    ):
        main()
