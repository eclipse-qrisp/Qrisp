"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from typing import Any, List

import pytest

from qrisp import QuantumArray, QuantumBool, cx
from qrisp.environments.layered_enviroment import GateStack, LayeredEnvironment


def _filter_instructions(qs_data: List[Any]) -> List[Any]:
    """Helper to filter out `qb_alloc` instructions from a list of instructions."""
    return [instr for instr in qs_data if not str(instr).startswith("qb_alloc")]


def measure_Z_stabilizer(source: List[Any], target: Any):
    """Measure a Z stabilizer on the given source qubits, using the target ancilla."""
    with GateStack():
        for q in source:
            cx(q, target)


class TestLayeredEnvironmentBasicUsage:
    """Test basic usage of LayeredEnvironment and GateStack."""

    def test_basic_stabilizer_measurement(self):
        """Test that instructions are correctly layered and emitted for a simple stabilizer measurement."""
        N = 6
        qubits = QuantumArray(qtype=QuantumBool(), shape=(2 * N - 1,))
        data_qubits = qubits[::2]
        ancilla_qubits = qubits[1::2]

        with LayeredEnvironment():
            for i in range(N - 1):
                with GateStack():
                    cx(data_qubits[i], ancilla_qubits[i])
                    cx(data_qubits[i + 1], ancilla_qubits[i])

        filtered_data = _filter_instructions(qubits.qs.data)

        expected_instructions = [
            "cx(Qubit(qubits.0), Qubit(qubits_1.0))",
            "cx(Qubit(qubits_2.0), Qubit(qubits_3.0))",
            "cx(Qubit(qubits_4.0), Qubit(qubits_5.0))",
            "cx(Qubit(qubits_6.0), Qubit(qubits_7.0))",
            "cx(Qubit(qubits_8.0), Qubit(qubits_9.0))",
            "cx(Qubit(qubits_2.0), Qubit(qubits_1.0))",
            "cx(Qubit(qubits_4.0), Qubit(qubits_3.0))",
            "cx(Qubit(qubits_6.0), Qubit(qubits_5.0))",
            "cx(Qubit(qubits_8.0), Qubit(qubits_7.0))",
            "cx(Qubit(qubits_10.0), Qubit(qubits_9.0))",
        ]

        for actual, expected in zip(filtered_data, expected_instructions, strict=True):
            assert str(actual) == expected

        assert qubits.qs.depth() == 2
