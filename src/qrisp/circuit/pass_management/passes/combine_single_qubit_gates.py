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

from __future__ import annotations

import numpy as np

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.operation import ControlledOperation, U3Gate
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


def _apply_combined_gates(qc_new: QuantumCircuit, gate_list: list, qb) -> None:
    """Flush a sequence of single-qubit gates on *qb* into *qc_new*.

    The unitaries of all gates in *gate_list* are multiplied together
    (last gate first, since it acts on the state first).  If the product
    equals the identity (within numerical tolerance), the gates cancel
    entirely and nothing is appended.  Otherwise a single operation is
    appended: the original gate when the sequence length is 1, or a
    generic unitary for longer sequences.

    Parameters
    ----------
    qc_new : QuantumCircuit
        The output circuit being built.
    gate_list : list
        Stack of single-qubit gate operations (last appended = last to act).
    qb : Qubit
        The qubit the gates act on.
    """
    if not gate_list:
        return

    n = len(gate_list)

    # Multiply unitaries in reverse order: the gate that was applied
    # *last* is the rightmost factor in the product.
    m = np.eye(2, dtype=complex)
    while gate_list:
        gate = gate_list.pop()
        m = m @ gate.get_unitary()

    # If the combined unitary is the identity (up to numerical precision),
    # the gates cancel completely.
    if np.linalg.norm(m - np.eye(2)) < 1e-10:
        return

    # For a single gate there is nothing to combine — just re-emit it.
    if n == 1:
        qc_new.append(gate, [qb])
        return

    # Multiple gates: emit a single unitary encapsulating the product.
    qc_new.unitary(m, [qb])


@CircuitPass
def combine_single_qubit_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """Combine adjacent single-qubit gates into unitary operations.

    This pass scans the circuit instruction-by-instruction and accumulates
    consecutive single-qubit gates on each qubit.  When an interruption is
    encountered (multi-qubit gate, allocation, measurement, …) the
    pending gates are flushed: their unitaries are multiplied together and
    emitted as a single operation, cancelling any sequences that reduce to
    the identity.

    **Recursive processing**

    Composite gates (gates with a ``definition``) are processed
    recursively so that single-qubit gate sequences inside compound
    operations are also combined.

    Controlled operations whose ``base_operation`` carries a definition
    are handled analogously: the base definition is combined in place
    and the controlled wrapper is preserved.

    This pass is designed as a local optimisation that reduces gate count
    and depth without changing the overall circuit semantics.  It is
    especially useful after transpilation passes that may introduce
    redundant single-qubit rotations.

    Parameters
    ----------
    qc : QuantumCircuit
        The input circuit.

    Returns
    -------
    QuantumCircuit
        A new circuit with adjacent single-qubit gates combined.

    Examples
    --------

    >>> from qrisp import PassManager, combine_single_qubit_gates
    >>> from qrisp import QuantumCircuit
    >>> qc = QuantumCircuit(1)
    >>> qc.x(0)
    >>> qc.z(0)
    >>> qc.h(0)   # X, Z, Y on the same qubit
    >>> pm = PassManager()
    >>> pm += combine_single_qubit_gates
    >>> optimized_qc = pm.run(qc)
    >>> # X, Z, Y combined into a single U3 gate
    >>> print(optimized_qc)
           ┌──────────────┐
    qb_64: ┤ U3(π/2,3π,0) ├
           └──────────────┘
    """
    # Per-qubit stacks of single-qubit gates waiting to be flushed.
    qb_dic = {qb: [] for qb in qc.qubits}

    qc_new = qc.clearcopy()

    for instr in qc.data:
        op = instr.op

        # Determine whether this instruction is a "barrier" that forces
        # us to flush the accumulated single-qubit gates first.
        is_single_qubit_gate = isinstance(op, U3Gate)

        if not is_single_qubit_gate or op.abstract_params:
            # Flush pending gates for every qubit touched by this instruction.
            for qb in instr.qubits:
                _apply_combined_gates(qc_new, qb_dic[qb], qb)

            # ---- Recursive processing of composite gates ----
            if isinstance(op, ControlledOperation):
                # If the controlled operation's base gate itself has a
                # definition, recursively combine single-qubit gates inside
                # that definition, then re-wrap in a ControlledOperation.
                if op.base_operation.definition:
                    instr = instr.copy()
                    optimized_base = combine_single_qubit_gates(
                        op.base_operation.definition
                    ).to_gate(name=op.base_operation.name)
                    instr.op = ControlledOperation(
                        optimized_base,
                        num_ctrl_qubits=len(op.ctrl_state),
                        ctrl_state=op.ctrl_state,
                    )
            elif op.definition:
                # Generic composite gate: recursively optimise its definition.
                instr = instr.copy()
                instr.op = instr.op.copy()
                instr.op.definition = combine_single_qubit_gates(op.definition)

            qc_new.append(instr)
        else:
            # Single-qubit gate — defer; will be combined with neighbours.
            qb_dic[instr.qubits[0]].append(op)

    # Flush any remaining gates at the end of the circuit.
    for qb in qc.qubits:
        _apply_combined_gates(qc_new, qb_dic[qb], qb)

    return qc_new
