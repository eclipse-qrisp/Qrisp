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

from qrisp.circuit.quantum_circuit import QuantumCircuit
from qrisp.circuit.operation import (
    ControlledOperation,
    Operation,
    PTControlledOperation,
)
from qrisp.circuit.qubit import Qubit
from qrisp.circuit.pass_management.circuit_pass import CircuitPass


# Single-qubit diagonal gates that map \|0⟩ → e^{iφ}\|0⟩
_DIAGONAL_1Q = frozenset({"p", "rz", "z", "s", "t", "s_dg", "t_dg", "id", "gphase"})


def _is_cancelled_by_zero(
    op: Operation, qubits: list[Qubit], fresh: set[Qubit]
) -> bool:
    r"""Return True if *op* on *qubits* is a no-op given a set of \|0⟩ qubits."""

    # Symmetric controlled-phase gates: diag(1,1,1,e^{iφ}).
    # Identity whenever *either* qubit is \|0⟩.
    if op.name in ("cp", "cz"):
        return any(qb in fresh for qb in qubits)

    # PTControlledOperation wrapping a phase gate on 2 qubits (≡ CP)
    if (
        isinstance(op, PTControlledOperation)
        and op.base_operation.name == "p"
        and op.num_qubits == 2
    ):
        return any(qb in fresh for qb in qubits)

    # General controlled operations: cancel when a control-on-\|1⟩ is still \|0⟩.
    if isinstance(op, (ControlledOperation, PTControlledOperation)):
        n_ctrl = len(op.controls)
        ctrl_state = getattr(op, "ctrl_state", "1" * n_ctrl)
        if isinstance(ctrl_state, int):
            ctrl_state = bin(ctrl_state)[2:].zfill(n_ctrl)
        for i in range(n_ctrl):
            if qubits[i] in fresh and ctrl_state[i] == "1":
                return True

    return False


@CircuitPass
def cancel_zero_controls(qc: QuantumCircuit) -> QuantumCircuit:
    r"""Cancel controlled gates whose control qubit is guaranteed to be \|0⟩.

    Every qubit starts in \|0⟩ (and re-enters \|0⟩ after ``qb_alloc``).  A
    controlled gate conditioned on \|1⟩ acting on such a qubit is a no-op.
    For symmetric controlled-phase gates (CP, CZ) the gate is a no-op if
    *either* qubit is \|0⟩.

    This is particularly effective on QFT-style circuits, where many
    controlled-phase gates act on qubits that have not yet been touched.

    Parameters
    ----------
    qc : QuantumCircuit
        The input circuit.

    Returns
    -------
    QuantumCircuit
        A new circuit with redundant controlled gates removed.

    Examples
    --------
    Cancel a CX whose control qubit is still in \|0⟩::

        >>> from qrisp import QuantumCircuit, PassManager
        >>> from qrisp import cancel_zero_controls
        >>> qc = QuantumCircuit(2)
        >>> qc.cx(0, 1)  # Qubit 0 starts in \|0⟩ — the CX is a no-op
        >>> qc.h(1)       # Now qubit 1 is marked as used
        >>> print(qc)
        qb_58: ──■───────
               ┌─┴─┐┌───┐
        qb_59: ┤ X ├┤ H ├
               └───┘└───┘
        >>> pm = PassManager()
        >>> pm += cancel_zero_controls
        >>> optimized_qc = pm.run(qc)
        >>> print(optimized_qc)
        <BLANKLINE>
        qb_58: ─────
               ┌───┐
        qb_59: ┤ H ├
               └───┘

    Symmetric controlled-phase gates (CZ, CP) cancel when *either*
    qubit is \|0⟩::

        >>> qc = QuantumCircuit(2)
        >>> qc.cz(0, 1)   # both qubits start in \|0⟩ — CZ is a no-op
        >>> pm = PassManager()
        >>> pm += cancel_zero_controls
        >>> optimized = pm.run(qc)
        >>> print(optimized)
        <BLANKLINE>
        qb_60:
        <BLANKLINE>
        qb_61:
        <BLANKLINE>
    """
    fresh = set(qc.qubits)  # all qubits start in \|0⟩
    qc_new = qc.clearcopy()

    for instr in qc.data:
        op = instr.op
        qubits = instr.qubits

        if op.name == "qb_alloc":
            fresh.add(qubits[0])
            qc_new.append(instr)
            continue

        if op.name == "qb_dealloc":
            fresh.discard(qubits[0])
            qc_new.append(instr)
            continue

        if op.name == "barrier":
            qc_new.append(instr)
            continue

        # Check whether the gate is a no-op given the fresh set.
        if _is_cancelled_by_zero(op, qubits, fresh):
            continue

        # Gate is kept — append and update freshness.
        qc_new.append(instr)

        # Single-qubit diagonal gates preserve \|0⟩.
        if len(qubits) == 1 and op.name in _DIAGONAL_1Q:
            continue

        # Everything else: conservatively drop freshness.
        for qb in qubits:
            fresh.discard(qb)

    return qc_new
