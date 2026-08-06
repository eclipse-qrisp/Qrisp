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
    asap_layers,
    intra_layer_substeps,
    is_full_width_barrier,
    is_transparent,
)
from qrisp.circuit.quantum_circuit import QuantumCircuit


def layerize(insert_barriers: bool = False) -> CircuitPass:
    """Create a pass that reorders instructions into "as-soon-as-possible" (ASAP) layers.

    Each instruction is assigned the earliest layer where all of its qubits
    (and classical bits) are free.  Instructions that act on disjoint sets of
    qubits can therefore appear in the same layer, which compacts the
    timeline.  The relative order of instructions that share a qubit is
    preserved, so the circuit semantics are unchanged.

    This pass is especially useful before exporting a circuit to Stim: Stim
    draws gates strictly in the order they appear in the circuit data, so
    calling this pass first yields a visually compacted timeline diagram.

    Instructions that do not correspond to a physical time step ride along in
    the layer of the gate they belong to instead of opening a layer of their
    own: ``qb_alloc`` / ``qb_dealloc``, ``gphase``, ``parity``, barriers and
    Stim noise channels.  Consequently, running this pass *after* a
    noise-insertion pass keeps every noise channel adjacent to the gate it
    annotates.  A qubit does carry at most one noise channel per layer, though —
    a second error on the same qubit is a second error, and belongs to the next
    time step.

    Barriers act on the qubits they name and nothing else, so an instruction
    on unrelated qubits may legally move past a partial barrier.

    Parameters
    ----------
    insert_barriers : bool, optional
        If ``True``, emit a full-width barrier at every layer boundary, which
        turns the ASAP schedule into an explicit time axis: each barrier
        becomes a ``TICK`` in the Stim output, giving one ``TICK`` per time
        step.  A boundary that already carries a full-width barrier is left
        alone, so the pass is idempotent and never produces an empty moment
        (a doubled ``TICK``).  A *partial* barrier is a local fence rather than
        a time boundary and does not count as a mark.  The default is
        ``False``.

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
        layers = asap_layers(qc)
        substeps = intra_layer_substeps(qc, layers)

        # Sort instructions by their layer, and within a layer by their drawing
        # sub-step so that everything Stim can draw side by side is emitted
        # consecutively.  The sort is stable, which is what keeps the per-qubit
        # instruction order intact.
        ordered = sorted(zip(layers, substeps, qc.data, strict=True), key=lambda item: item[:2])

        new_qc = qc.clearcopy()

        if not insert_barriers:
            for *_, instr in ordered:
                new_qc.append(instr)
            return new_qc

        # Whether a full-width barrier has been emitted since the last time
        # step.  Tracking the whole run of transparent instructions rather than
        # just the immediately preceding one makes the duplicate suppression
        # independent of where exactly a barrier sorts within its layer, and so
        # of the scheduler's barrier-layer convention.
        boundary_marked = False
        previous_step: int | None = None

        def mark_boundary() -> None:
            nonlocal boundary_marked
            if boundary_marked:
                return
            new_qc.barrier()
            boundary_marked = True

        for layer, _, instr in ordered:
            # A barrier *is* a boundary rather than something sitting inside a
            # time step, so it never opens one.  Everything else belongs to its
            # layer and must land on the layer's side of the boundary - including
            # instructions transparent to the layer clock, such as the readout
            # noise written in front of a measurement.
            is_barrier = instr.op.name == "barrier"

            # Opening a new time step: close the previous one first.
            if not is_barrier and previous_step is not None and layer != previous_step:
                mark_boundary()

            new_qc.append(instr)

            # Only instructions that occupy a physical time step make their layer
            # one.  A layer holding nothing but barriers or bookkeeping is not a
            # time step, so it must not become the reference for the next
            # boundary - that would add a TICK for an empty moment.
            if not is_transparent(instr):
                previous_step = layer
                boundary_marked = False
            elif is_barrier and is_full_width_barrier(instr, new_qc):
                boundary_marked = True

        # Close the final time step, so the number of TICKs equals the number
        # of time steps rather than being one short.
        if previous_step is not None:
            mark_boundary()

        return new_qc

    return _layerize
