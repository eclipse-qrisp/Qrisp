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
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


@CircuitPass
def reverse_parallelize(qc: QuantumCircuit) -> QuantumCircuit:
    """
    This pass leverages permeability commutations to move two qubit gates
    to a later point in the circuit. This is especially helpful when
    trying to cancel out SWAP gates with other two qubit interactions.

    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit.

    Returns
    -------
    QuantumCircuit
        Circuit after reverse-order parallelization.

    Examples
    --------

    We demonstrate how to move a CX gate towards a SWAP.

    >>> from qrisp import QuantumCircuit, PassManager
    >>> from qrisp import reverse_parallelize
    >>> qc = QuantumCircuit(2)
    >>> qc.cx(0, 1)
    >>> qc.z(0)
    >>> qc.swap(0, 1)
    >>> print(qc)
                 ┌───┐   
    qb_130: ──■──┤ Z ├─X─
            ┌─┴─┐├───┤ │ 
    qb_131: ┤ X ├┤ X ├─X─
            └───┘└───┘   

    >>> pm = PassManager()
    >>> pm += reverse_parallelize
    >>> optimized_qc = pm.run(qc)
    >>> print(optimized_qc)
           ┌───┐        
    qb_66: ┤ Z ├──■───X─
           ├───┤┌─┴─┐ │ 
    qb_67: ┤ X ├┤ X ├─X─
           └───┘└───┘   

    The CX gate can now be fused through the ``cancel_inverses`` pass.
                 
    """
    # Defer import: qrisp.permeability loads *after* qrisp.circuit, so
    # importing at module level would create a circular import.
    from qrisp.permeability import parallelize_qc

    # Make the one qubit gates slower than the two qubit ones
    # to make the parallelize pass execute the two qubit gates first.
    def depth_indicator(op):
        if op.num_qubits == 1:
            return 10
        return 1

    reversed_qc = qc.copy()
    reversed_qc.data.reverse()
    reversed_qc = parallelize_qc(reversed_qc, depth_indicator)
    reversed_qc.data.reverse()
    return reversed_qc
