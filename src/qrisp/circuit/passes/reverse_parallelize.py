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

"""
Reverse-order parallelization pass.

This pass reverses the instruction order, runs Qrisp's parallelization,
and restores the original order.  It is useful to group commuting gates
with later SWAP operations so cancellation passes can remove redundant
two-qubit gates.
"""

from __future__ import annotations

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.passes.pass_manager import CircuitPass


@CircuitPass
def reverse_parallelize(qc: QuantumCircuit) -> QuantumCircuit:
    """Run parallelization on the reversed instruction order.

    Reverses the circuit, applies ``parallelize_qc``, then reverses again.
    This surfaces commutation opportunities that are invisible in forward order,
    particularly between SWAP gates and later two-qubit gates.

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.

    Returns
    -------
    QuantumCircuit
        Circuit after reverse-order parallelization.
    """
    # Defer import: qrisp.permeability loads *after* qrisp.circuit, so
    # importing at module level would create a circular import.
    from qrisp.permeability import parallelize_qc

    reversed_qc = qc.copy()
    reversed_qc.data.reverse()
    reversed_qc = parallelize_qc(reversed_qc)
    reversed_qc.data.reverse()
    return reversed_qc
