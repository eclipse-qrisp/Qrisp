"""********************************************************************************
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

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit


@CircuitPass
def promote_barriers(qc: QuantumCircuit) -> QuantumCircuit:
    """Widen every barrier in the circuit to span the full circuit.

    Only a full-width barrier becomes a ``TICK`` in the Stim output, so this pass
    is what gives a Jasp-traced circuit its ``TICK``\\ s.  While tracing, the
    qubits belonging to the program are not yet concrete, so a barrier can only be
    written over a static qubit list and comes out partial.  Barriering a
    **single qubit** is enough: the pass widens whatever it finds, so the traced
    barrier only has to mark the spot::

        trace local barriers → extract to QuantumCircuit → promote_barriers → TICK

    .. warning::

        Promotion adds scheduling constraints the circuit did not have: qubits
        the original barrier did not name must now synchronise with it, which
        widens the schedule.  With automatic noise insertion this also inflates
        the idle-noise budget, since every qubit picks up a noise instruction for
        each time step the promoted barrier creates.

    Parameters
    ----------
    qc : QuantumCircuit
        The input circuit, potentially containing partial barriers.

    Returns
    -------
    QuantumCircuit
        A new circuit in which every barrier spans ``qc.qubits``.  Instructions
        are neither reordered nor otherwise modified.

    Examples
    --------
    A partial barrier is a local fence and produces no ``TICK``; promoting it
    makes the boundary global:

    >>> from qrisp import QuantumCircuit, promote_barriers
    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.barrier([qc.qubits[0]])
    >>> qc.h(1)
    >>> qc.to_stim().num_ticks
    0
    >>> promote_barriers(qc).to_stim().num_ticks
    1

    Under Jasp, ``barrier(a)`` on a whole :class:`~qrisp.QuantumVariable` does not
    trace — sizing the operation needs ``len(a)`` statically — so index a qubit
    out and mark the boundary on that one:

    >>> from qrisp import QuantumFloat, barrier, cx, h
    >>> from qrisp.jasp import make_jaspr
    >>> def main():
    ...     a = QuantumFloat(2)
    ...     b = QuantumFloat(2)
    ...     h(a[0])
    ...     cx(a[0], b[0])
    ...     barrier([a[0]])
    ...     h(a[0])
    ...     cx(a[0], b[0])
    ...     return a
    >>> jaspr = make_jaspr(main)()
    >>> *_, qc = jaspr.to_qc()

    The traced barrier spans one of the four extracted qubits, so it carries no
    ``TICK`` until it is promoted:

    >>> [len(instr.qubits) for instr in qc.data if instr.op.name == "barrier"]
    [1]
    >>> qc.to_stim().num_ticks
    0
    >>> print(promote_barriers(qc).to_stim())
    H 0
    CX 0 2
    TICK
    H 0
    CX 0 2

    """
    new_qc = qc.clearcopy()

    for instr in qc.data:
        if instr.op.name == "barrier":
            # Rebuild rather than mutate: barrier() sizes the Operation to the
            # full register, and the original instruction may be shared with
            # another circuit.
            new_qc.barrier()
            continue
        new_qc.append(instr)

    return new_qc
