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
    current layer of their resources without advancing the clock.  They do still
    constrain those resources, though: an instruction written after a transparent
    one is never scheduled ahead of it.  A two-qubit error channel spanning a
    deep and a shallow qubit therefore pulls the shallow qubit's next gate up to
    its own layer instead of letting it slip in front.

    A barrier raises a lower bound ("floor") on the layer of every subsequent
    instruction touching one of **its own** qubits, so it forces those qubits'
    later instructions strictly past its earlier ones.  Qubits the barrier does
    not name are left alone.  The barrier's own reported layer is that floor,
    i.e. the layer it opens.

    Error channels follow two extra rules, because their layer is a statement
    about the noise model rather than about resource availability:

    * They are placed in the layer they were *written into* — that of the most
      recent instruction occupying a time step — rather than in the layer of the
      last gate on their own qubits.  Otherwise an *idle* channel, one on a qubit
      with no gate in the layer it annotates, would drift back to that qubit's
      previous gate and be reported a time step too early.
    * A qubit carries **at most one error channel per layer**.  A second channel
      on the same qubit is a second error, so it moves on to the next layer whose
      slot is free.  This is what keeps two consecutive idle errors from
      collapsing into one time step, and it places readout noise — written
      *before* its measurement, after that layer's gate noise — in the
      measurement's layer without any special case for measurements.

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

    A second error on the same qubit moves on to the next layer, since a qubit
    cannot pick up two errors in one time step:

    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
    >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
    >>> asap_layers(qc)
    [0, 0, 1]

    """
    noise_gate_type = _stim_noise_gate_type()

    # Latest layer occupied by each qubit / clbit.  -1 means "untouched", so the
    # first real instruction on a resource lands in layer 0.
    last_time: dict[object, int] = defaultdict(lambda: -1)

    # Per-resource lower bound on the layer of subsequent instructions.  Barriers
    # and transparent instructions raise it; it is what makes them act as fences.
    # -1 means "no fence yet", a value neither can produce (both fence at layer 0
    # or later), so an unfenced resource is never clamped.
    floor: dict[object, int] = defaultdict(lambda: -1)

    # ``(resource, layer)`` pairs whose error slot is already taken.  A qubit
    # carries at most one error channel per layer.
    error_slots: set[tuple[object, int]] = set()

    # Layer of the most recent instruction that occupied a time step - the layer
    # an error channel written here is annotating.
    current_step = 0

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

        if noise_gate_type is not None and isinstance(instr.op, noise_gate_type):
            # An error channel occupies no time step of its own, but a qubit can
            # only carry *one* error per time step - a second one is a second
            # error, and belongs to the next.  So a channel starts in the layer
            # it was written into (never before its qubits' own last gate or
            # fence) and moves on until it finds a layer whose error slot is
            # free.
            t = max(
                current_step,
                max(last_time[r] for r in resources),
                max(floor[r] for r in resources),
            )
            while any((r, t) in error_slots for r in resources):
                t += 1
            for r in resources:
                error_slots.add((r, t))
                # Fence at the layer the channel actually ended up in, so a gate
                # written after it cannot be scheduled ahead of it.
                floor[r] = max(floor[r], t)
            layers.append(t)
            continue

        transparent = name in TRANSPARENT_OP_NAMES

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
            current_step = t
            for r in resources:
                last_time[r] = t
        else:
            # A transparent instruction occupies no time step, but it is still an
            # ordering constraint: whatever follows it on these resources must not
            # be scheduled ahead of it.  Fencing instead of occupying is what
            # keeps several transparent instructions - and the next real gate -
            # free to share the layer.
            for r in resources:
                floor[r] = max(floor[r], t)

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
