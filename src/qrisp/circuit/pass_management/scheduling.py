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

As-soon-as-possible (ASAP) circuit scheduling.

A circuit *layer* is one time step.  :func:`~qrisp.layerize` reorders instructions
into layers using the schedule computed here.

:func:`asap_schedule` walks the circuit's dependency graph and fills one layer at
a time with everything that fits.  A qubit or clbit admits **one gate**, **one
error channel** and **any number of other annotations** per layer.  That is the
whole rule; ASAP layering is what the gate slot alone produces.

Working on the dependency graph keeps the result faithful to the input: an
instruction is only released once all of its predecessors are out, so it can
never be emitted ahead of one it was written after.

Two consequences are worth stating here, because callers depend on them.

**A barrier only constrains the qubits it names.**  It ends the layer for those
qubits and leaves the rest alone, so an instruction on an unrelated qubit may
move past it.  ``QuantumCircuit.barrier()`` names every qubit, so the usual
full-circuit fence still works, and only such a full-width barrier is a time
boundary — see :func:`~qrisp.circuit.is_full_width_barrier`.

**A gate and the error channel annotating it share a layer, two errors do not.**
The second error on a qubit belongs to the next layer.  This is also what puts
readout noise, written in front of a measurement, in the measurement's layer:
that layer's gate noise has already taken the qubit's error slot.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

from qrisp.circuit.instruction import Instruction
from qrisp.circuit.quantum_circuit import QuantumCircuit

__all__ = [
    "Schedule",
    "asap_layers",
    "asap_schedule",
    "is_error_channel",
    "is_transparent",
]

# Operations that take no time step and cannot be recognised structurally:
#
# * ``qb_alloc`` / ``qb_dealloc`` are bookkeeping markers.
# * ``gphase`` records a global phase, but carries a qubit like a real gate.
# * ``parity`` becomes a Stim ``DETECTOR`` or ``OBSERVABLE_INCLUDE``, i.e.
#   classical post-processing, but carries clbits like a real instruction.
# * ``barrier`` is a scheduling constraint rather than an operation.
#
# Error channels and instructions acting on nothing are recognised in
# :func:`is_transparent` itself.
_TRANSPARENT_OP_NAMES = frozenset({"qb_alloc", "qb_dealloc", "gphase", "parity", "barrier"})


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
    """Check whether *instr* takes no time step of its own.

    Such an instruction rides along in the layer of the gate it belongs to, so it
    does not add to the circuit depth.  This covers barriers, ``qb_alloc`` /
    ``qb_dealloc``, ``gphase``, ``parity``, Stim error channels, and anything
    acting on no resources at all.

    Parameters
    ----------
    instr : Instruction
        The instruction to classify.

    Returns
    -------
    bool
        ``True`` if the instruction takes no time step.

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
    if instr.op.name in _TRANSPARENT_OP_NAMES:
        return True
    if not instr.qubits and not instr.clbits:
        # Nothing to schedule against, so nothing to advance.
        return True
    noise_gate_type = _stim_noise_gate_type()
    return noise_gate_type is not None and isinstance(instr.op, noise_gate_type)


def is_error_channel(instr: Instruction) -> bool:
    """Check whether *instr* is a Stim error channel.

    Error channels are the only instructions the schedule rations: a qubit takes
    at most one per layer.  Allocation markers, ``gphase`` and ``parity`` also
    take no time step but are not rationed, so the two have to be told apart.

    Parameters
    ----------
    instr : Instruction
        The instruction to classify.

    Returns
    -------
    bool
        ``True`` if the operation is a ``StimNoiseGate``.  Always ``False``
        without Stim installed, which is safe: a circuit cannot contain one.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from qrisp.circuit.pass_management.scheduling import is_error_channel
    >>> from qrisp.misc.stim_tools import StimNoiseGate
    >>> qc = QuantumCircuit(1)
    >>> qc.h(0)
    >>> qc.append(StimNoiseGate("X_ERROR", 0.1), [qc.qubits[0]])
    >>> [is_error_channel(instr) for instr in qc.data]
    [False, True]

    """
    noise_gate_type = _stim_noise_gate_type()
    return noise_gate_type is not None and isinstance(instr.op, noise_gate_type)


def _resources(instr: Instruction) -> list:
    """Return the qubits and clbits *instr* acts on.

    Clbits count as resources, so that a ``parity`` follows the measurements it
    reads and two measurements writing one clbit do not share a layer.
    """
    return [*instr.qubits, *instr.clbits]


def _dependency_graph(qc: QuantumCircuit) -> tuple[list[int], list[list[int]]]:
    """Link consecutive instructions on a shared resource.

    That is the whole ordering the schedule has to respect, barrier fences
    included: a barrier shares a qubit with everything it fences.

    Returns
    -------
    tuple[list[int], list[list[int]]]
        The predecessor count and the successors of each instruction.
    """
    data = qc.data
    predecessors: list[set[int]] = [set() for _ in data]
    successors: list[list[int]] = [[] for _ in data]
    last_on: dict[object, int] = {}

    for i, instr in enumerate(data):
        resources = _resources(instr)
        if not resources and i:
            # Nothing to order it against, so pin it behind the instruction it
            # was written after rather than letting it float to the front.
            predecessors[i].add(i - 1)
        for r in resources:
            if r in last_on:
                # A set, so a two-qubit gate following the same instruction on
                # both of its qubits still counts that dependency once.
                predecessors[i].add(last_on[r])
            last_on[r] = i

    for i, preds in enumerate(predecessors):
        for j in preds:
            successors[j].append(i)

    return [len(p) for p in predecessors], successors


class Schedule(NamedTuple):
    """The result of :func:`asap_schedule`."""

    layers: list[int]
    """Layer of every instruction, indexed like ``qc.data``."""

    order: list[int]
    """Indices into ``qc.data``, in the order to emit them in."""


def asap_schedule(qc: QuantumCircuit) -> Schedule:
    """Schedule *qc* into as-soon-as-possible layers.

    The schedule walks the circuit's dependency graph, filling one layer at a
    time with everything that fits.  What fits is decided per qubit and clbit:

    * **one gate**, so instructions sharing a resource land in successive layers
      and instructions on disjoint resources share one;
    * **one error channel**, so a gate and its noise share a layer but a second
      error on the same qubit moves to the next;
    * **any number of other annotations** — ``qb_alloc`` / ``qb_dealloc``,
      ``gphase``, ``parity`` — since they cost no time;
    * **a barrier ends the layer** for the qubits it names.  It needs no slot of
      its own, so it does not add to the depth, but nothing else on those qubits
      may join afterwards.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to schedule.

    Returns
    -------
    Schedule
        The layer of every instruction, and the order to emit them in.

    Notes
    -----
    ``layers`` increases along the instructions of a single resource, but not
    along ``qc.data``: two independent qubit chains both start at layer 0.  A
    time axis therefore only exists once the instructions are put in ``order``,
    which is what :func:`~qrisp.layerize` does.

    ``order`` is a topological order, so emitting in it preserves the circuit
    semantics.  Each group in it acts on disjoint resources, which is what lets
    Stim draw a layer as a single column.

    Stacking more error channels on a qubit than there are layers to hold them
    pushes that qubit's next gate later, since each error takes a time step of
    its own.  That is the honest reading of the input: fifteen errors in a row
    are fifteen errors, not one time step.

    Examples
    --------
    >>> from qrisp import QuantumCircuit
    >>> from qrisp.circuit.pass_management.scheduling import asap_schedule
    >>> qc = QuantumCircuit(4)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> qc.h(2)
    >>> qc.cx(2, 3)
    >>> schedule = asap_schedule(qc)
    >>> schedule.layers
    [0, 1, 0, 1]
    >>> schedule.order
    [0, 2, 1, 3]

    """
    data = qc.data
    remaining, successors = _dependency_graph(qc)

    ready = {i for i, count in enumerate(remaining) if count == 0}
    layers = [0] * len(data)
    order: list[int] = []

    # Per-resource state of the layer being built, all cleared per layer.
    gates: set[object] = set()  # gate slot taken
    errors: set[object] = set()  # error slot taken
    closed: set[object] = set()  # a barrier ended the layer here
    layer = 0

    def fits(i: int) -> bool:
        """Check whether instruction *i* can still join the current layer."""
        instr = data[i]
        resources = _resources(instr)
        if instr.op.name == "barrier":
            # A barrier ends a layer instead of occupying it, so it always fits -
            # including into a layer another barrier has already closed.
            return True
        if not closed.isdisjoint(resources):
            return False
        if is_error_channel(instr):
            return errors.isdisjoint(resources)
        if is_transparent(instr):
            return True
        return gates.isdisjoint(resources)

    def occupy(i: int) -> None:
        """Record what instruction *i* uses up in the current layer."""
        instr = data[i]
        resources = _resources(instr)
        if instr.op.name == "barrier":
            closed.update(resources)
        elif is_error_channel(instr):
            errors.update(resources)
        elif not is_transparent(instr):
            gates.update(resources)

    while ready:
        gates.clear()
        errors.clear()
        closed.clear()

        # Fill the layer in groups.  Every instruction in ``ready`` acts on
        # resources disjoint from the others, since two instructions sharing one
        # are always linked, so a whole group can be admitted at once.  Releasing
        # it unblocks successors, which may still fit - a gate's error channel,
        # say - so keep going until nothing does.
        while True:
            group = sorted(i for i in ready if fits(i))
            if not group:
                break
            for i in group:
                occupy(i)
                layers[i] = layer
                order.append(i)
                ready.discard(i)
            for i in group:
                for j in successors[i]:
                    remaining[j] -= 1
                    if remaining[j] == 0:
                        ready.add(j)

        layer += 1

    return Schedule(layers, order)


def asap_layers(qc: QuantumCircuit) -> list[int]:
    """Return the layer of every instruction in ``qc.data``.

    A wrapper around :func:`asap_schedule` for callers that need the layers but
    not the emission order.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to schedule.

    Returns
    -------
    list[int]
        The layer index of every instruction, indexed like ``qc.data``.

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
    [0, 0, 1, 0]

    A second error on the same qubit moves on to the next layer, since a qubit
    cannot pick up two errors in one time step:

    >>> from qrisp.misc.stim_tools import StimNoiseGate
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
    >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.01), [qc.qubits[1]])
    >>> asap_layers(qc)
    [0, 0, 1]

    """
    return asap_schedule(qc).layers
