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

Shared as-soon-as-possible (ASAP) scheduling utilities for circuit passes.

Several passes need the same notion of a circuit *layer* (time step):

* :func:`~qrisp.layerize` reorders instructions into layers.
* ``insert_stim_noise`` gives every qubit one noise instruction per layer.

Both used to carry their own scheduler, and the two disagreed about what a
layer is — most visibly about barriers and about which instructions advance the
clock.  This module is the single definition both consume.

Two conventions are worth spelling out, because passes downstream rely on them.

**Barriers are scoped to the qubits they name.**  A barrier is a compiler
constraint ("do not reorder across me"), and it constrains exactly its own
qubits: instructions on *other* qubits may freely move past it.  Note that
``QuantumCircuit.barrier()`` without arguments names every qubit, so the common
"fence everything" spelling is unaffected.  A global time boundary is therefore
spelled as a full-width barrier — see :func:`is_full_width_barrier` and
:func:`~qrisp.promote_barriers`.

**Barriers open the layer that follows them.**  A barrier occupies no time step
of its own; its reported layer is the layer of the instructions it fences off,
i.e. it sits at the *near* side of the boundary it creates.  Passes that need to
ask "is this layer boundary already marked by a barrier?" can therefore look at
the instruction opening the new layer.
"""

from __future__ import annotations

import functools
from collections import defaultdict

from qrisp.circuit.instruction import Instruction
from qrisp.circuit.quantum_circuit import QuantumCircuit

__all__ = [
    "TRANSPARENT_OP_NAMES",
    "asap_layers",
    "intra_layer_substeps",
    "is_full_width_barrier",
    "is_transparent",
]

# Instructions that do not represent a physical time step.  They do not advance
# the layer clock and (for noise passes) receive no noise instruction:
#
# * ``qb_alloc`` / ``qb_dealloc`` are bookkeeping markers.
# * ``gphase`` is a one-qubit Operation in Qrisp, but it only records a global
#   phase and is therefore not a physical gate.  (It really does carry a qubit,
#   so it cannot be recognised by its qubit count - hence this list.)
# * ``parity`` carries no qubits at all - it becomes a Stim ``DETECTOR`` or
#   ``OBSERVABLE_INCLUDE`` annotation, i.e. classical post-processing.  It does
#   however carry clbits, so without this list it would look like an ordinary
#   resource-consuming instruction and occupy a time step of its own.
# * ``barrier`` is a scheduling constraint rather than an operation.
#
# ``StimNoiseGate`` operations and instructions without any resources are
# transparent as well, but are recognised structurally in :func:`is_transparent`
# rather than listed here.
TRANSPARENT_OP_NAMES = frozenset({"qb_alloc", "qb_dealloc", "gphase", "parity", "barrier"})


@functools.lru_cache(maxsize=1)
def _stim_noise_gate_type() -> type | None:
    """Return the ``StimNoiseGate`` class, or ``None`` if Stim is unavailable.

    ``stim`` is an optional dependency, so scheduling must not require it.  A
    circuit that contains no ``StimNoiseGate`` cannot have been built without
    Stim installed, hence treating the class as absent is safe.
    """
    try:
        from qrisp.misc.stim_tools.error_class import StimNoiseGate
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return StimNoiseGate


def is_transparent(instr: Instruction) -> bool:
    """Check whether *instr* is transparent to the layer clock.

    A transparent instruction rides along at the current layer of its resources
    instead of opening a new one, and therefore does not count as a time step.
    This keeps a noise instruction in the same layer as the gate it annotates,
    and keeps bookkeeping markers from inflating the layer count.

    Parameters
    ----------
    instr : Instruction
        The instruction to classify.

    Returns
    -------
    bool
        ``True`` if the instruction does not occupy a physical time step.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from qrisp.circuit.pass_management.scheduling import is_transparent
    >>> qc = QuantumCircuit(1)
    >>> qc.h(0)
    >>> qc.barrier()
    >>> [is_transparent(instr) for instr in qc.data]
    [False, True]

    """
    if instr.op.name in TRANSPARENT_OP_NAMES:
        return True
    if not instr.qubits and not instr.clbits:
        # Nothing to schedule against, so nothing to advance.
        return True
    noise_gate_type = _stim_noise_gate_type()
    return noise_gate_type is not None and isinstance(instr.op, noise_gate_type)


def is_full_width_barrier(instr: Instruction, qc: QuantumCircuit) -> bool:
    """Check whether *instr* is a barrier spanning every qubit of *qc*.

    Full width is what separates the two things a barrier can mean.  A
    full-width barrier is a **global time boundary** — it becomes a ``TICK`` in
    the Stim output.  A partial barrier is a **local fence** on the qubits it
    names and produces no timeline annotation.

    Parameters
    ----------
    instr : Instruction
        The instruction to classify.
    qc : QuantumCircuit
        The circuit whose register defines "full width".

    Returns
    -------
    bool
        ``True`` if *instr* is a barrier that names all of ``qc.qubits``.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from qrisp.circuit.pass_management.scheduling import is_full_width_barrier
    >>> qc = QuantumCircuit(3)
    >>> qc.barrier()            # no arguments -> spans the whole register
    >>> qc.barrier([qc.qubits[0]])
    >>> [is_full_width_barrier(instr, qc) for instr in qc.data]
    [True, False]

    """
    if instr.op.name != "barrier":
        return False
    if not qc.qubits:
        # Degenerate register: there is nothing for a barrier to fail to span.
        return True
    return len(set(instr.qubits)) == len(qc.qubits)


def asap_layers(qc: QuantumCircuit) -> list[int]:
    """Assign an as-soon-as-possible layer (time step) to every instruction.

    Instructions that act on disjoint resources share a layer; instructions
    that touch a common qubit/clbit are forced into successive layers.
    Transparent instructions (see :func:`is_transparent`) ride along at the
    current layer of their resources without advancing the clock.

    A barrier raises a lower bound ("floor") on the layer of every subsequent
    instruction touching one of **its own** qubits, so it forces those qubits'
    later instructions strictly past its earlier ones.  Qubits the barrier does
    not name are left alone.  The barrier's own reported layer is that floor,
    i.e. the layer it opens.

    Error channels get one refinement on top of plain transparency: they are
    placed in the layer they were *written into* rather than in the layer of the
    last gate on their own qubits.  Without it, an idle channel — one on a qubit
    that has no gate in the layer it annotates — would drift back to that
    qubit's previous gate and be visualised in an earlier time step than the
    error it represents.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to schedule.

    Returns
    -------
    list[int]
        The layer index of every instruction in ``qc.data``, in the order the
        instructions appear there.

    Notes
    -----
    The returned indices are non-decreasing along the instructions of any
    *single* resource, but not along ``qc.data`` as a whole: two independent
    qubit chains written one after the other both start at layer 0.  Consumers
    may rely on the former and must not assume the latter.  In particular, a
    coherent time axis (and hence a ``TICK`` stream) only exists once the
    instructions have been sorted into layer order — which is what
    :func:`~qrisp.layerize` does.

    Indices may be negative: a transparent instruction on an otherwise
    untouched qubit reports ``-1``, which is what places allocation markers
    ahead of the first real layer.

    Examples
    --------
    Two independent chains both start at layer 0:

    >>> from qrisp import QuantumCircuit
    >>> from qrisp.circuit.pass_management.scheduling import asap_layers
    >>> qc = QuantumCircuit(4)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.h(2)
    >>> qc.cx(2, 3)
    >>> asap_layers(qc)
    [0, 1, 0, 1]

    A partial barrier constrains only the qubits it names, so ``h(1)`` stays in
    layer 0 while ``h(0)`` is pushed past the barrier:

    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.barrier([qc.qubits[0]])
    >>> qc.h(0)
    >>> qc.h(1)
    >>> asap_layers(qc)
    [0, 1, 1, 0]

    An error channel stays in the layer it was written into, even on a qubit
    that has no gate there:

    >>> from qrisp.misc.stim_tools import StimNoiseGate
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[0]])
    >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
    >>> asap_layers(qc)
    [0, 0, 0]

    """
    noise_gate_type = _stim_noise_gate_type()

    # Latest layer occupied by each qubit / clbit.  -1 means "untouched", so the
    # first real instruction on a resource lands in layer 0.
    last_time: dict[object, int] = defaultdict(lambda: -1)

    # Per-resource lower bound on the layer of subsequent instructions.  Only
    # barriers raise it; it is what makes them act as fences.  -1 means "no
    # fence yet", which a barrier can never produce (its boundary is always at
    # least 0), so an unfenced resource is never clamped.
    floor: dict[object, int] = defaultdict(lambda: -1)

    layers: list[int] = []

    for instr in qc.data:
        name = instr.op.name

        # Clbits count as resources: a measurement and a later instruction
        # reading the same clbit must not share a time step.
        resources = [*instr.qubits, *instr.clbits]

        if not resources:
            # Nothing to schedule against and nothing that could depend on it:
            # join the layer currently in progress without advancing anything.
            layers.append(max(0, max(last_time.values(), default=-1)))
            continue

        if name == "barrier":
            # A fence *between* layers: it occupies no time step of its own
            # (and so generates no noise), but everything it names must move
            # strictly past everything it named before.  The bound is taken
            # over the barrier's own resources only - a partial barrier is a
            # local fence, not a global synchronisation point.
            boundary = max(
                max(last_time[r] for r in resources) + 1,
                max(floor[r] for r in resources),
            )
            for r in resources:
                floor[r] = boundary
            layers.append(boundary)
            continue

        transparent = name in TRANSPARENT_OP_NAMES or (
            noise_gate_type is not None and isinstance(instr.op, noise_gate_type)
        )

        # ASAP: the earliest layer in which every resource is free.
        t = max(last_time[r] for r in resources)
        if not transparent:
            t += 1
        # Respect fences.  This also applies to transparent instructions, which
        # would otherwise be free to drift back across a barrier on their own
        # qubits.
        t = max(t, max(floor[r] for r in resources))
        layers.append(t)

        if not transparent:
            # Only real instructions occupy their resources; transparent ones
            # leave the clock untouched so several of them can share a layer.
            for r in resources:
                last_time[r] = t

    if noise_gate_type is not None:
        _place_error_channels(qc, layers, noise_gate_type)

    return layers


def intra_layer_substeps(qc: QuantumCircuit, layers: list[int]) -> list[int]:
    """Assign each instruction a drawing sub-step within its layer.

    A layer says "all of this happens in one time step".  Stim, however, renders
    instructions strictly in the order they appear and does not go back to fill
    an earlier column, so the *order within a layer* decides whether the layer
    is drawn as one parallel column or smeared across several.  Emitting
    ``H 0 | X_ERROR 0 | H 5 | X_ERROR 5`` puts the two ``H`` gates in different
    columns even though they share a layer.

    The sub-step is an ASAP schedule *inside* the layer in which every
    instruction consumes its resources — including the ones that are transparent
    to the layer clock.  Sorting a layer by it groups everything that can be
    drawn side by side: all first-use instructions, then all second-use ones,
    and so on.  The example above becomes ``H 0 | H 5 | X_ERROR 0 | X_ERROR 5``,
    which Stim draws as two columns.

    This is purely presentational and never changes semantics: instructions that
    share a resource get strictly increasing sub-steps, so their relative order
    is preserved, and only instructions on disjoint resources can move past one
    another.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit the layers were computed for.
    layers : list[int]
        The layer of every instruction, as returned by :func:`asap_layers`.

    Returns
    -------
    list[int]
        The sub-step of every instruction in ``qc.data``.  Sort the circuit by
        ``(layer, substep)`` — stably — to get a layer-parallel instruction
        order.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from qrisp.circuit.pass_management.scheduling import asap_layers, intra_layer_substeps
    >>> from qrisp.misc.stim_tools import StimNoiseGate
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[0]])
    >>> qc.h(1)
    >>> qc.append(StimNoiseGate("X_ERROR", 0.5), [qc.qubits[1]])
    >>> layers = asap_layers(qc)
    >>> layers
    [0, 0, 0, 0]
    >>> intra_layer_substeps(qc, layers)
    [0, 1, 0, 1]

    """
    # Per layer, the next free sub-step of each resource.
    occupied: dict[int, dict[object, int]] = {}
    substeps: list[int] = []

    for instr, layer in zip(qc.data, layers, strict=True):
        resources = [*instr.qubits, *instr.clbits]
        if not resources:
            substeps.append(0)
            continue

        slots = occupied.setdefault(layer, {})
        substep = max(slots.get(r, 0) for r in resources)
        for r in resources:
            slots[r] = substep + 1
        substeps.append(substep)

    return substeps


def _place_error_channels(qc: QuantumCircuit, layers: list[int], noise_gate_type: type) -> None:
    """Move error channels into the layer they were written into, in place.

    Plain transparency puts an error channel in the layer of the last gate on
    its own qubits.  That is right for a channel annotating a gate, but wrong
    for an *idle* channel: the qubit has no gate in the layer being annotated,
    so the channel drifts back to an earlier one and is drawn in the wrong time
    step.

    The layer a channel was written into is the layer of the nearest preceding
    instruction that occupies a time step.  Two clamps keep the move sound:

    * never earlier than the transparent placement, which is what respects
      barriers and the last gate on the channel's own qubits;
    * never at or past the next instruction occupying a time step on any of the
      channel's own resources, so a channel can never overtake a gate it was
      written in front of.  This matters because layers are not monotone along
      ``qc.data`` (see :func:`asap_layers`), so the nearest preceding time step
      can belong to an unrelated, much later part of the circuit.
    """
    data = qc.data

    # For each instruction, the layer of the next time-step-occupying
    # instruction on any of its own resources - the upper clamp.
    next_step_on: dict[object, int] = {}
    limits: list[int | None] = [None] * len(data)
    for i in range(len(data) - 1, -1, -1):
        instr = data[i]
        resources = [*instr.qubits, *instr.clbits]
        if not is_transparent(instr):
            for r in resources:
                next_step_on[r] = layers[i]
            continue
        following = [next_step_on[r] for r in resources if r in next_step_on]
        limits[i] = min(following) - 1 if following else None

    # The layer each instruction was written into - the target.
    written_into: int | None = None
    for i, instr in enumerate(data):
        if not is_transparent(instr):
            written_into = layers[i]
            continue
        if written_into is None or not isinstance(instr.op, noise_gate_type):
            continue
        target = written_into if limits[i] is None else min(written_into, limits[i])
        layers[i] = max(layers[i], target)
