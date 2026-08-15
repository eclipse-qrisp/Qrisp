# This work is licensed under the European Union Public Licence (EUPL), Version 1.2.
#
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the Licence is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the Licence for the specific language
# governing permissions and limitations under the Licence.

"""Manual qubit layout pass.

Creates a pass function that re-indexes qubits according to a caller-supplied
physical qubit mapping.  Logical qubit *i* is placed on physical qubit
``qubit_mapping[i]``.  The output circuit may have more qubits than the
input if the maximum physical index exceeds the logical qubit count.
"""

from __future__ import annotations

from collections.abc import Callable

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.qubit import Qubit


def manual_layout(
    qubit_mapping: list[int],
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a pass that applies a manual qubit layout to the circuit.

    Parameters
    ----------
    qubit_mapping : list[int]
        A list of physical qubit indices. The *i*-th logical qubit in the
        circuit is mapped to physical qubit ``qubit_mapping[i]``.

        For example, ``[2, 0, 1]`` means:

        * Logical qubit 0 → Physical qubit 2
        * Logical qubit 1 → Physical qubit 0
        * Logical qubit 2 → Physical qubit 1

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A pass function that transforms the circuit.

    Raises
    ------
    ValueError
        If the mapping length does not match the circuit qubit count, if any
        index is negative, or if there are duplicate indices.

    Examples
    --------
    >>> from qrisp import QuantumCircuit, PassManager
    >>> from qrisp import manual_layout
    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.cx(1, 2)
    >>> print(qc)
          ┌───┐
    qb_0: ┤ H ├
          └───┘
    qb_1: ──■──
          ┌─┴─┐
    qb_2: ┤ X ├
          └───┘

    >>> pm = PassManager()
    >>> pm += manual_layout([2, 0, 1])  # Logical 0→2, 1→0, 2→1
    >>> new_layout_qc = pm.run(qc)
    >>> print(new_layout_qc)
    <BLANKLINE>
    qb_1: ──■──
          ┌─┴─┐
    qb_2: ┤ X ├
          ├───┤
    qb_0: ┤ H ├
          └───┘

    """

    @CircuitPass
    def _manual_layout(qc: QuantumCircuit) -> QuantumCircuit:
        num_circuit_qubits = qc.num_qubits()

        if len(qubit_mapping) != num_circuit_qubits:
            raise ValueError(
                f"qubit_mapping specifies {len(qubit_mapping)} qubits, but the circuit has {num_circuit_qubits} qubits."
            )

        for idx in qubit_mapping:
            if idx < 0:
                raise ValueError(f"Qubit index {idx} is invalid. Indices must be non-negative.")

        if len(qubit_mapping) != len(set(qubit_mapping)):
            raise ValueError(
                f"Duplicate qubit indices in qubit_mapping: {qubit_mapping}. "
                f"Each circuit qubit must be mapped to a unique physical qubit."
            )

        num_physical_qubits = max(qubit_mapping) + 1 if qubit_mapping else 0

        new_qc = qc.copy()
        amended_qubits: list[Qubit] = []

        while new_qc.num_qubits() < num_physical_qubits:
            amended_qubits.append(Qubit("amended_qb_" + str(len(amended_qubits))))
            new_qc.add_qubit(amended_qubits[-1])

        reverse_mapping = {phys: log for log, phys in enumerate(qubit_mapping)}
        new_qubit_list: list[Qubit] = []
        for i in range(num_physical_qubits):
            if i in qubit_mapping:
                new_qubit_list.append(qc.qubits[reverse_mapping[i]])
            else:
                new_qubit_list.append(amended_qubits.pop(0))

        new_qc.qubits = new_qubit_list
        return new_qc

    _manual_layout.__name__ = "manual_layout"
    return _manual_layout
