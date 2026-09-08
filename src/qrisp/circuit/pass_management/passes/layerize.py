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

from collections.abc import Callable

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.pass_management.scheduling import (
    asap_schedule,
    is_error_channel,
    is_transparent,
)
from qrisp.circuit.quantum_circuit import QuantumCircuit, is_full_width_barrier


def layerize(insert_barriers: bool = False) -> CircuitPass:
    """Create a pass that reorders instructions into "as-soon-as-possible" (ASAP) layers.

    Each instruction goes in the earliest layer where all of its qubits and
    classical bits are free, so instructions on disjoint qubits end up in the
    same layer.  Instructions sharing a qubit keep their relative order, which
    leaves the circuit semantics unchanged.

    This is mainly useful before exporting to Stim.  Stim draws gates in the
    order they appear in the circuit, so running this pass first gives a compact
    timeline diagram.

    ``qb_alloc`` / ``qb_dealloc``, ``gphase``, ``parity``, barriers and Stim
    noise channels take no time step of their own and ride along in the layer of
    the gate they belong to.  Running this pass *after* a noise-insertion pass
    therefore keeps every noise channel next to its gate.  A qubit does take at
    most one noise channel per layer: a second error on it belongs to the next
    time step.

    Barriers only act on the qubits they name, so an instruction on unrelated
    qubits may move past a partial barrier.

    Parameters
    ----------
    insert_barriers : bool, optional
        If ``True``, emit a full-width barrier at every layer boundary.  Each
        one becomes a ``TICK`` in the Stim output, giving one ``TICK`` per time
        step.  Boundaries that already carry a full-width barrier are left alone,
        so the pass stays idempotent and never produces a doubled ``TICK``.  A
        partial barrier only fences its own qubits and does not count as a
        boundary.  The default is ``False``.

    Returns
    -------
    CircuitPass
        The configured pass, mapping a :class:`~qrisp.QuantumCircuit` to a new
        circuit with instructions reordered into ASAP layers.

    Examples
    --------
    Consider two independent qubit chains.  Writing the first chain
    completely before the second is natural for a programmer but forces
    Stim to draw the second chain *after* the first, wasting timeline
    space::

        >>> from qrisp import QuantumCircuit, layerize
        >>> qc = QuantumCircuit(6)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.cx(1, 2)
        >>> qc.h(5)
        >>> qc.cx(5, 4)
        >>> qc.cx(4, 3)
        >>> _ = qc.to_stim()  # convert to Stim for visualisation

    .. image:: /_static/layerize_before.svg
       :alt: Stim timeline diagram before layerize

    The second chain (qubits 3–5) sits far to the right even though it shares no
    qubits with the first chain.  ``layerize`` reorders the instructions so that
    independent gates occupy the same time layer::

        >>> qc = layerize()(qc)    # or via PassManager
        >>> _ = qc.to_stim()

    .. image:: /_static/layerize_after.svg
       :alt: Stim timeline diagram after layerize

    Both chains now execute in parallel — the Stim timeline diagram
    becomes roughly half as wide.  The relative order of the three gates
    on each chain (``H → CX → CX``) is preserved, guaranteeing that the
    circuit semantics are unchanged.

    The pass can also be used inside a :class:`~qrisp.PassManager`::

        >>> from qrisp import PassManager
        >>> pm = PassManager()
        >>> pm += layerize()
        >>> qc = pm.run(qc)

    Instructions annotating a gate — error channels, for instance — are grouped
    so that Stim can still draw the layer in parallel:

    >>> from qrisp.misc.stim_tools import StimNoiseGate
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
    >>> qc.h(1)
    >>> qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[1]])
    >>> print(layerize()(qc).to_stim())
    H 0 1
    X_ERROR(0.5) 0 1

    Without the grouping, Stim would receive ``H 0``, ``X_ERROR 0``, ``H 1``,
    ``X_ERROR 1`` and draw four columns instead of two.

    With ``insert_barriers=True`` the resulting schedule is written back into
    the circuit as barriers, one per time step:

    >>> qc = QuantumCircuit(3)
    >>> qc.h(0)
    >>> qc.h(2)
    >>> qc.cx(0, 1)
    >>> layerized = layerize(insert_barriers=True)(qc)
    >>> [instr.op.name for instr in layerized.data]
    ['h', 'h', 'barrier', 'cx', 'barrier']

    Each of those barriers becomes a ``TICK``, so the Stim timeline carries
    exactly one time-step boundary per layer:

    >>> layerized.to_stim().num_ticks
    2

    Applying the pass again changes nothing, because every boundary is
    already marked:

    >>> twice = layerize(insert_barriers=True)(layerized)
    >>> [instr.op.name for instr in twice.data] == [instr.op.name for instr in layerized.data]
    True

    """

    @CircuitPass
    def _layerize(qc: QuantumCircuit) -> QuantumCircuit:
        # The scheduler already returns the order to emit in: one group of
        # independent instructions at a time, which is what lets Stim draw a
        # layer as a single column.
        schedule = asap_schedule(qc)
        ordered = [(schedule.layers[i], qc.data[i]) for i in schedule.order]

        new_qc = qc.clearcopy()

        if not insert_barriers:
            for _, instr in ordered:
                new_qc.append(instr)
            return new_qc

        # Whether a full-width barrier has been emitted since the last time step.
        # Tracking a whole run rather than just the previous instruction is what
        # makes the duplicate suppression independent of where a barrier happens
        # to sit within its layer.
        boundary_marked = False
        previous_step: int | None = None

        def mark_boundary() -> None:
            nonlocal boundary_marked
            if boundary_marked:
                return
            new_qc.barrier()
            boundary_marked = True

        for layer, instr in ordered:
            # A barrier is a boundary, not something inside a time step, so it
            # never opens one.  Everything else belongs to its layer and has to
            # land on that layer's side of the boundary - readout noise included.
            is_barrier = instr.op.name == "barrier"

            # Opening a new time step: close the previous one first.
            if not is_barrier and previous_step is not None and layer != previous_step:
                mark_boundary()

            new_qc.append(instr)

            # A layer holding nothing but barriers or bookkeeping is not a time
            # step, so it must not become the reference for the next boundary -
            # that would add a TICK for an empty moment.  A layer holding only
            # error channels *is* one: an idle error is something happening.
            if not is_transparent(instr) or is_error_channel(instr):
                previous_step = layer
                boundary_marked = False
            elif is_barrier and is_full_width_barrier(instr, new_qc):
                boundary_marked = True

        # Close the final time step, so there are as many TICKs as time steps.
        if previous_step is not None:
            mark_boundary()

        return new_qc

    return _layerize
