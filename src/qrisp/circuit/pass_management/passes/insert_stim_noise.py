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

Insertion of a circuit-level Stim noise model into a QuantumCircuit.

The contract of this pass is a single sentence: *every qubit receives exactly
one noise instruction per physical time step*, with the channel determined by
what that qubit does in the time step (two-qubit gate, one-qubit gate, idling,
measurement, reset).  Everything below exists to make that sentence true
without reordering the user's circuit.

Architecture
------------
The pass runs in three phases.

**Phase 1 - schedule** (:func:`_asap_layers`).  Every instruction is assigned
an as-soon-as-possible layer index.  Instructions on disjoint resources share
a layer; instructions sharing a qubit or clbit land in successive layers.  Two
rules carry most of the weight:

* *Transparent* instructions (see :data:`_TRANSPARENT_OP_NAMES` and
  pre-existing ``StimNoiseGate`` operations) ride along at the current layer of
  their resources without advancing the clock.  This is what keeps a
  user-placed noise instruction attached to the layer of the gate it annotates,
  and what stops classical annotations such as ``parity`` from inventing time
  steps that would then collect a full round of idle noise.
* A ``barrier`` raises a global ``floor``, forcing everything after it into a
  strictly later layer than everything before it.

**Phase 2 - classify.**  Each instruction gets a role (``"barrier"``,
``"transparent"``, ``"noise"``, ``"measure"``, ``"reset"``, ``"gate"``), and
gates on more than two qubits are rejected here - before any output circuit
exists, so the pass either succeeds or leaves nothing half-built.  From the
roles we derive the two facts the emitter needs: which layers are physical time
steps at all (``noise_layers``), and which qubits already draw their noise from
an instruction in a given layer (``handled``).  Every qubit of every
non-transparent instruction is "handled": a measured qubit gets ``X_ERROR``
instead of idle noise, a gate qubit gets its depolarizing channel, and a qubit
carrying user noise has its slot consumed by that noise.

**Phase 3 - emit.**  The circuit is rebuilt by walking ``qc.data`` in its
*original order* and inserting noise between the existing instructions.  The
output therefore differs from the input by insertions only.

Why idle noise is emitted lazily
--------------------------------
A layer is a *set* of instructions, not a contiguous slice of ``qc.data``: a
program that writes two independent qubit chains one after the other has layer
0 instructions at both the start and the middle of the data list.  So "append
the idle noise at the end of the layer" has no single well-defined position
unless the circuit is reordered into layer order first - which is precisely
what this pass must not do in order to not change the measurement record.

The way out is the observation that a noise instruction on qubit ``q`` only has
to be ordered correctly with respect to *other instructions on* ``q``.  Noise
on ``q`` commutes past every instruction that does not touch ``q``, so ``q``'s
idle noise for layer ``L`` may be placed anywhere between ``q``'s last
instruction in a layer ``< L`` and ``q``'s first instruction in a layer
``> L``.  Idle noise is therefore treated as an *obligation* held per qubit and
discharged at the latest still-correct moment:

* immediately before any instruction that touches ``q``,
* at every ``barrier``, which is a global time boundary (it becomes a Stim
  ``TICK``) and therefore flushes the obligations of *all* qubits,
* at the end of the circuit for whatever is left over.

Soundness rests on a monotonicity property of phase 1: for a fixed qubit, the
instructions touching it appear in ``qc.data`` with non-decreasing layer
indices.  A single forward cursor per qubit (``idle_pos``) is hence enough, and
no obligation is ever discharged into the past.  The cursor indexes the shared,
sorted ``noise_layer_list`` rather than a per-qubit list of idle layers, which
keeps the memory cost at one integer per qubit instead of one per
(qubit, layer) pair.

Deliberate non-goals
--------------------
* **Qubit liveness is not tracked.**  ``qb_alloc`` / ``qb_dealloc`` are passed
  through, but idle noise covers the whole register in every time step
  regardless of allocation state, which matches the fixed physical qubit
  register of the intended use case.
* **The timeline is not compacted.**  Producing a layer-monotone instruction
  order (and with it a meaningful global time axis) is the job of the separate
  ``compress_layers`` pass, which can be run afterwards.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.quantum_circuit import QuantumCircuit

# Instructions that do not represent a physical time step.  They neither
# receive a noise instruction nor advance the layer clock:
#
# * ``qb_alloc`` / ``qb_dealloc`` are bookkeeping markers.
# * ``gphase`` is a one-qubit Operation in Qrisp, but it only records a global
#   phase and is therefore not a physical gate.  (It really does carry a qubit,
#   so it cannot be recognised by its qubit count - hence this list.)
# * ``parity`` carries no qubits at all - it becomes a Stim ``DETECTOR`` or
#   ``OBSERVABLE_INCLUDE`` annotation, i.e. classical post-processing.  It does
#   however carry clbits, so without this list it would look like an ordinary
#   resource-consuming instruction and occupy a time step of its own.
#
# Instructions without any qubits are transparent as well, but are recognised
# structurally in the classification phase rather than listed here.
_TRANSPARENT_OP_NAMES = frozenset({"qb_alloc", "qb_dealloc", "gphase", "parity"})


def _asap_layers(qc: QuantumCircuit) -> list[int]:
    """Assign an as-soon-as-possible layer (time step) to every instruction.

    Instructions that act on disjoint resources share a layer; instructions
    that touch a common qubit/clbit are forced into successive layers.
    Instructions that carry no physical time step (see
    ``_TRANSPARENT_OP_NAMES`` and pre-existing ``StimNoiseGate`` operations)
    ride along at the current layer of their resources without advancing the
    clock, which keeps a user-placed noise instruction attached to the layer of
    the gate it annotates.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to schedule.

    Returns
    -------
    list[int]
        The layer index of every instruction in ``qc.data``.

    Notes
    -----
    The returned indices are non-decreasing along the instructions of any
    *single* resource, but not along ``qc.data`` as a whole: two independent
    qubit chains written one after the other both start at layer 0.  The
    emission phase relies on the former and must not assume the latter.

    """
    # Lazy import: ``stim`` is only required when noise is actually inserted.
    from qrisp.misc.stim_tools.error_class import StimNoiseGate

    # Latest layer occupied by each qubit / clbit.  -1 means "untouched", so the
    # first real instruction on a resource lands in layer 0.
    last_time: dict[object, int] = defaultdict(lambda: -1)
    layers: list[int] = []

    # Global lower bound for the layer of subsequent instructions.  Only
    # barriers raise it; it is what makes them act as synchronisation points.
    floor = 0

    for instr in qc.data:
        name = instr.op.name

        # Clbits count as resources: a measurement and a later instruction
        # reading the same clbit must not share a time step.
        resources = [*instr.qubits, *instr.clbits]

        if name == "barrier":
            # A barrier is a synchronization point *between* layers.  It does
            # not occupy a time step of its own (and therefore generates no
            # noise), but it forces every subsequent instruction into a
            # strictly later layer than every instruction before it.  This
            # mirrors the ``TICK`` that barriers become in Stim.
            #
            # The bound is taken over *all* resources rather than only the
            # barrier's own qubits, so a partially applied barrier still
            # synchronises globally - consistent with TICK being global, at the
            # price of over-synchronising the qubits it does not name.
            floor = max(floor, max(last_time.values(), default=-1) + 1)
            layers.append(floor)
            continue

        if not resources:
            # Nothing to schedule against: place the instruction in the current
            # global layer without advancing the clock.  Note the missing "+ 1"
            # compared to the case below - such an instruction joins the layer
            # in progress instead of opening a new one.
            layers.append(max(floor, max(last_time.values(), default=-1)))
            continue

        # Transparent instructions ride along with the layer their resources are
        # already in.  Concretely, a user-placed StimNoiseGate ends up in the
        # same layer as the gate it follows, so it can consume that qubit's
        # noise slot for the layer instead of creating a new one.
        is_transparent = name in _TRANSPARENT_OP_NAMES or isinstance(instr.op, StimNoiseGate)

        # ASAP: the earliest layer in which every resource is free.
        t = max(last_time[r] for r in resources)
        if not is_transparent:
            t += 1
        t = max(t, floor)
        layers.append(t)

        # Only real instructions occupy their resources; transparent ones leave
        # the clock untouched so that several of them can share the layer.
        if not is_transparent:
            for r in resources:
                last_time[r] = t

    return layers


def insert_stim_noise(
    depolarize_1_strength: float = 0.01,
    depolarize_2_strength: float = 0.01,
    X_error_strength: float = 0.01,
    only_necessary: bool = True,
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a pass that inserts a circuit-level Stim noise model into a circuit.

    The circuit is organized into layers (time steps) using an
    as-soon-as-possible (ASAP) schedule: instructions that act on disjoint
    qubits share a layer, instructions that touch a common qubit are placed
    in successive layers.  Every qubit then receives **exactly one** noise
    instruction per layer, according to the following rules:

    .. list-table::
       :widths: 40 60
       :header-rows: 1

       * - Condition
         - Noise inserted
       * - Qubit participates in a two-qubit gate
         - ``DEPOLARIZE2(p2)`` appended **after** the gate.
       * - Qubit participates in a one-qubit gate
         - ``DEPOLARIZE1(p1)`` appended **after** the gate.
       * - Qubit is idle (no gate in the layer)
         - ``DEPOLARIZE1(p1)`` appended at the end of the layer.
       * - Qubit is measured
         - ``X_ERROR(px)`` appended **before** the measurement.
       * - Qubit is reset
         - ``X_ERROR(px)`` appended **after** the reset.

    The layers are only used to decide *what* noise a qubit needs and *how
    often*.  The instructions of the input circuit are **not** reordered: the
    pass walks the original circuit in order and inserts the noise
    instructions between them.  Idle noise for a layer is emitted as late as
    possible, namely right before the next instruction that touches the idle
    qubit (or at the next barrier, or at the end of the circuit).  Since no
    operation on that qubit intervenes, this is equivalent to emitting it at
    the end of its layer.  Use the :ref:`compress_layers` pass afterwards to
    additionally compact the timeline for visualisation.

    This is the standard circuit-level depolarizing noise model used for
    quantum error-correction benchmarks (e.g. surface / repetition codes):
    every gate is followed by a depolarizing channel, idle qubits accumulate
    single-qubit depolarizing noise each layer, and measurements / resets
    are flanked by a bit-flip error.  The resulting :class:`~qrisp.QuantumCircuit`
    can be exported to a Stim circuit via :meth:`~qrisp.QuantumCircuit.to_stim`,
    where the noise instructions behave as genuine noisy channels (see
    :class:`~qrisp.misc.stim_tools.StimNoiseGate`).

    Parameters
    ----------
    depolarize_1_strength : float, default 0.01
        Single-qubit depolarizing error probability ``p1``.
    depolarize_2_strength : float, default 0.01
        Two-qubit depolarizing error probability ``p2``.
    X_error_strength : float, default 0.01
        Pauli-``X`` (bit-flip) error probability ``px`` applied around
        measurements and resets.
    only_necessary : bool, default True
        When ``True``, the pass only inserts noise where the circuit does
        not already contain it: no channel is inserted where the circuit
        already carries an **exactly matching** one, that is a
        :class:`~qrisp.misc.stim_tools.StimNoiseGate` of the same Stim type
        acting on exactly the same qubits, directly after the gate, directly
        before the measurement, or directly after the reset.  The error
        probability is not compared, so a user-placed channel of a different
        strength is respected.  Anything else (a different channel type, a
        different set of qubits, a channel on the other side of the
        instruction) does not count as a match and the noise model is applied
        regardless.  When ``False``, the full noise model is inserted
        unconditionally.

    Returns
    -------
    Callable[[QuantumCircuit], QuantumCircuit]
        A :class:`~qrisp.CircuitPass` that inserts the noise model into a
        given :class:`~qrisp.QuantumCircuit` and returns a new circuit.

    Raises
    ------
    ValueError
        If any of the error probabilities does not lie in ``[0, 1]`` (raised
        immediately by this factory, since Stim requires probabilities to lie
        in this range), or if the circuit contains a gate acting on more than
        two qubits (the noise model only defines channels for one- and
        two-qubit gates).

    Notes
    -----
    * Noise instructions are :class:`~qrisp.misc.stim_tools.StimNoiseGate`
      operations.  They behave as identity gates for the Qrisp simulator but
      become genuine noisy channels when the circuit is converted to Stim.
    * Instructions that carry no physical time step are passed through
      unchanged and generate no noise: the bookkeeping operations
      ``qb_alloc`` / ``qb_dealloc``, the global phase ``gphase`` (which is a
      one-qubit ``Operation`` in Qrisp but not a physical gate), the
      ``parity`` annotation (which becomes a Stim ``DETECTOR`` /
      ``OBSERVABLE_INCLUDE``) and any instruction without qubits.  They also
      do not advance the layer clock, so they never create a noise layer of
      their own.
    * Qubit liveness is not tracked.  ``qb_alloc`` / ``qb_dealloc`` are passed
      through, but idle noise is applied to every qubit of the circuit in
      every layer, irrespective of whether that qubit is currently allocated.
      This matches the physical picture of a fixed qubit register, which is
      the intended use case.
    * The noise model only defines channels for one- and two-qubit gates.
      A gate acting on more than two qubits raises a :class:`ValueError`
      before any noise is inserted (decompose such gates first, e.g. with the
      ``decompose`` pass).
    * A ``barrier`` acts as a synchronization point *between* layers: it still
      enforces that gates before and after it are scheduled into separate
      layers (mirroring the ``TICK`` it becomes in Stim), but it is not a
      noise layer itself — it neither receives a noise instruction nor is a
      time step of its own.  It does act as a global time boundary, so all
      outstanding idle noise is emitted before it.
    * Existing :class:`~qrisp.misc.stim_tools.StimNoiseGate` instructions are
      always left untouched.  With ``only_necessary=True`` (the default) they
      also consume the noise slot of their qubits: no additional idle noise
      is stacked on top of them, and no duplicate channel is inserted
      directly after the gate / before the measurement / after the reset they
      already annotate.  With ``only_necessary=False`` the full noise model is
      applied on top of any user-placed noise.
    * *Reading the output — repeated targets.*  Stim merges consecutive
      identical channels into a single instruction, so a qubit that idles
      through several time steps appears more than once in the target list:
      ``DEPOLARIZE1(0.001) 0 0 0 2 2`` applies the channel three times to qubit
      ``0`` and twice to qubit ``2``.  Each target is an independent
      application, so this is exactly equivalent to five separate instructions,
      one per idle time step.
    * *Reading the output — placement of idle noise.*  Because the instruction
      order of the input is preserved, a qubit's idle noise is emitted at the
      latest still-correct position rather than spread across the round: right
      before the next instruction that touches the qubit, at the next barrier,
      or at the end of the circuit.  Nothing acts on the qubit in between, so
      the resulting channels compose identically — only the layout of the
      printed circuit differs.  Run :ref:`compress_layers` afterwards to
      distribute the instructions over the timeline for visualisation.

    Examples
    --------
    **A first noise model.**  Insert a uniform model into a Bell-state circuit
    and inspect the Stim output::

        >>> from qrisp import QuantumCircuit, insert_stim_noise
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> for q in qc.qubits:
        ...     qc.measure(q)
        >>> noisy = insert_stim_noise(0.01, 0.02, 0.001)(qc)
        >>> print(noisy.to_stim())
        H 0
        DEPOLARIZE1(0.01) 0 1
        CX 0 1
        DEPOLARIZE2(0.02) 0 1
        X_ERROR(0.001) 0
        M 0
        X_ERROR(0.001) 1
        M 1

    Three time steps, three noise instructions per qubit.  In the first one
    ``H`` acts on qubit ``0`` while qubit ``1`` idles, and both therefore get a
    ``DEPOLARIZE1`` - Stim prints them as a single instruction with two targets.
    The ``CX`` gets a correlated ``DEPOLARIZE2`` over both qubits, and each
    measurement is preceded by its readout bit flip.

    **A syndrome-extraction round.**  The typical target of this pass: one round
    of a repetition-code parity check, closed by a detector and a barrier::

        >>> from qrisp import QuantumCircuit, insert_stim_noise
        >>> qc = QuantumCircuit(3)
        >>> qc.cx(0, 1)                        # data 0 -> ancilla 1
        >>> qc.cx(2, 1)                        # data 2 -> ancilla 1
        >>> qc.measure(1)
        >>> _ = qc.parity([qc.clbits[0]])      # becomes a Stim DETECTOR
        >>> qc.reset(1)
        >>> qc.barrier(qc.qubits)              # becomes a Stim TICK
        >>> print(insert_stim_noise(0.001, 0.01, 0.005)(qc).to_stim())
        CX 0 1
        DEPOLARIZE2(0.01) 0 1
        DEPOLARIZE1(0.001) 2
        CX 2 1
        DEPOLARIZE2(0.01) 2 1
        X_ERROR(0.005) 1
        M 1
        DETECTOR rec[-1]
        R 1
        X_ERROR(0.005) 1
        DEPOLARIZE1(0.001) 0 0 0 2 2
        TICK

    The round has four time steps: the two ``CX`` layers, the measurement and
    the reset.  The ancilla ``1`` is busy in all four and collects exactly four
    channels.  The data qubits ``0`` and ``2`` are busy in one each and idle
    through the other three, which is what the two ``DEPOLARIZE1`` instructions
    account for - three applications on qubit ``0`` and, together with the early
    one, three on qubit ``2`` (see the notes on reading the output above).
    Note that the ``DETECTOR`` does not produce a round of idle noise: it is a
    classical annotation, not a time step.

    Repeating the round and rendering the Stim timeline diagram before and after
    the pass shows the model in context — every qubit carries one channel per
    time step, and the rounds are separated by the ``TICK`` of the barrier:

    .. image:: /_static/insert_stim_noise_before.svg
       :alt: Stim timeline diagram of two noiseless syndrome extraction rounds

    .. image:: /_static/insert_stim_noise_after.svg
       :alt: Stim timeline diagram of the same rounds after insert_stim_noise

    **Respecting hand-placed noise.**  By default the pass only adds noise where
    it is missing.  A channel the user already placed is kept as-is::

        >>> from qrisp import QuantumCircuit, insert_stim_noise
        >>> from qrisp.misc.stim_tools import StimNoiseGate
        >>> qc = QuantumCircuit(3)
        >>> qc.cx(0, 1)
        >>> qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), qc.qubits[:2])  # user noise
        >>> noisy = insert_stim_noise(0.01, 0.02, 0.001)(qc)
        >>> print(noisy.to_stim())
        CX 0 1
        DEPOLARIZE2(0.05) 0 1
        DEPOLARIZE1(0.01) 2

    The user's ``DEPOLARIZE2`` is kept at its own strength and no second
    ``DEPOLARIZE2`` is inserted after the ``CX``; only the idle qubit (``2``)
    receives its ``DEPOLARIZE1`` for that layer.  The match has to be exact:
    had the user placed a ``DEPOLARIZE1``, or a ``DEPOLARIZE2`` on a different
    pair of qubits, the model's own ``DEPOLARIZE2`` would still be inserted.
    Pass ``only_necessary=False`` to apply the full model unconditionally.

    **Composition.**  The pass can be used inside a
    :class:`~qrisp.PassManager`.  Running :ref:`compress_layers` afterwards
    reorders the result into layer order, which compacts the Stim timeline
    diagram::

        >>> from qrisp import PassManager, compress_layers
        >>> pm = PassManager()
        >>> pm += insert_stim_noise(depolarize_1_strength=1e-3,
        ...                         depolarize_2_strength=1e-3,
        ...                         X_error_strength=1e-3)
        >>> pm += compress_layers
        >>> qc = pm.run(qc)

    """
    # Type guards on the strengths (match Stim's requirement p in [0, 1]).
    # Checked here in the factory rather than inside the pass, so a bad
    # probability is reported at the point where it was written down - typically
    # while a PassManager is being assembled, long before it is run.
    for label, p in [
        ("depolarize_1_strength", depolarize_1_strength),
        ("depolarize_2_strength", depolarize_2_strength),
        ("X_error_strength", X_error_strength),
    ]:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{label} must be in [0, 1], got {p!r}. Stim requires probabilities to lie in this range.")

    @CircuitPass
    def _insert_stim_noise(qc: QuantumCircuit) -> QuantumCircuit:
        # Lazy import: ``stim`` is only required when noise is actually inserted.
        from qrisp.misc.stim_tools.error_class import StimNoiseGate

        data = qc.data
        layers = _asap_layers(qc)

        # ------------------------------------------------------------------
        # 1. Classify every instruction.  Doing this up front means an
        #    unsupported gate is reported before any work is done.
        # ------------------------------------------------------------------
        # One role per instruction, positionally aligned with ``data`` and
        # ``layers``.  Deciding the role once keeps the two loops below in
        # agreement: phase 2 counts exactly the instructions phase 3 annotates.
        roles: list[str] = []

        for instr in data:
            name = instr.op.name

            # The order of these tests matters.  A barrier may well carry
            # qubits, and a transparent instruction may well carry none, so the
            # structural checks have to come after the name-based ones.
            if name == "barrier":
                roles.append("barrier")
            elif name in _TRANSPARENT_OP_NAMES or not instr.qubits:
                roles.append("transparent")
            elif isinstance(instr.op, StimNoiseGate):
                # Noise the circuit already contains.  Never touched, and with
                # ``only_necessary`` it also occupies its qubits' noise slot.
                roles.append("noise")
            elif name in ("measure", "measurement"):
                # Both spellings occur: ``QuantumCircuit.measure`` emits
                # "measure", while circuits converted from other frameworks may
                # carry "measurement".  The Stim converter accepts both.
                roles.append("measure")
            elif name == "reset":
                roles.append("reset")
            elif len(instr.qubits) > 2:
                # Rejected here, in the middle of the classification loop, so
                # the failure happens before ``new_qc`` is even created.  The
                # noise model simply has no channel for such a gate.
                raise ValueError(
                    f"Gate '{name}' acts on {len(instr.qubits)} qubits, "
                    "which the insert_stim_noise noise model does not "
                    "support (only one- and two-qubit gates can be "
                    "annotated). Decompose multi-qubit gates before "
                    "inserting noise."
                )
            else:
                roles.append("gate")

        # ------------------------------------------------------------------
        # 2. Determine which layers are physical time steps, and which qubits
        #    already get their noise from an instruction in that layer.
        # ------------------------------------------------------------------
        # ``noise_layers``: the layers that are time steps at all.  A layer that
        # contains nothing but barriers and transparent instructions is a
        # scheduling artefact and must not produce a round of idle noise.
        #
        # ``handled[layer]``: the qubits of that layer that draw their noise from
        # an instruction - a measured qubit (X_ERROR), a gate qubit (depolarizing
        # channel) or a qubit carrying user noise (slot consumed).  Every other
        # qubit of the layer is idling and owes a DEPOLARIZE1.
        noise_layers: set[int] = set()
        handled: dict[int, set] = defaultdict(set)

        for idx, instr in enumerate(data):
            if roles[idx] in ("barrier", "transparent"):
                continue
            noise_layers.add(layers[idx])
            handled[layers[idx]].update(instr.qubits)

        # ------------------------------------------------------------------
        # 3. Rebuild the circuit in its original order, inserting the noise.
        # ------------------------------------------------------------------
        new_qc = qc.clearcopy()

        # The time steps in ascending order, plus a per-qubit cursor into them.
        # Every qubit walks this list exactly once over the course of the
        # emission, picking up a DEPOLARIZE1 for each layer it idles through.
        # The list is shared and the per-qubit state is a single integer, which
        # avoids materialising one idle-layer list per qubit (that would cost
        # O(qubits * layers) memory on a large code).
        noise_layer_list = sorted(noise_layers)
        idle_pos = dict.fromkeys(qc.qubits, 0)

        def _flush_idle(qubit, before_layer: int) -> None:
            """Emit the outstanding idle noise of *qubit* for all layers before *before_layer*.

            Discharges the idle-noise obligations of *qubit* up to (excluding)
            *before_layer* and advances its cursor past them.  The cursor only
            ever moves forward, so no layer is served twice and none is served
            after the circuit has already moved past it.
            """
            pos = idle_pos[qubit]
            while pos < len(noise_layer_list) and noise_layer_list[pos] < before_layer:
                if qubit not in handled[noise_layer_list[pos]]:
                    new_qc.append(StimNoiseGate("DEPOLARIZE1", depolarize_1_strength), [qubit])
                pos += 1
            idle_pos[qubit] = pos

        def _already_annotated(idx: int, stim_name: str, qubits, before: bool) -> bool:
            """Return whether the instruction at *idx* already carries the expected channel.

            The instruction must be directly preceded (``before=True``) or
            followed (``before=False``) by a ``StimNoiseGate`` of Stim type
            *stim_name* acting on exactly *qubits*.  The error probability is
            deliberately not compared, so user-placed noise of a different
            strength is respected.
            """
            # The match has to be exact in both the channel type and the qubit
            # set, because the rules of the noise model overlap in position: the
            # instruction after a gate is also the instruction before whatever
            # comes next.  A loose match would let a single user-placed
            # X_ERROR count both as the gate's depolarizing channel and as the
            # following measurement's bit-flip error, dropping one of the two
            # channels the model calls for.
            #
            # Adjacency is evaluated on the *input* circuit.  Since the emitter
            # only inserts, and inserts nothing between a gate and a matching
            # neighbour, input adjacency implies output adjacency.
            j = idx - 1 if before else idx + 1
            if not 0 <= j < len(data):
                return False
            adj = data[j]
            return (
                isinstance(adj.op, StimNoiseGate) and adj.op.stim_name == stim_name and set(adj.qubits) == set(qubits)
            )

        for idx, instr in enumerate(data):
            role = roles[idx]
            layer = layers[idx]
            qubits = instr.qubits

            if role == "barrier":
                # A barrier becomes a Stim TICK, i.e. a global time boundary, so
                # it flushes *every* qubit rather than just its own: no idle
                # noise belonging to an earlier time step may cross a TICK.
                # Barriers are also the reason the flush never falls arbitrarily
                # far behind in a QEC circuit with one barrier per round.
                for q in qc.qubits:
                    _flush_idle(q, layer)
                new_qc.append(instr)
                continue

            # This instruction is about to occupy its qubits, which is the last
            # moment at which their outstanding idle noise from earlier layers
            # can still be emitted in the right place.
            for q in qubits:
                _flush_idle(q, layer)

            if role in ("transparent", "noise"):
                # Copied through verbatim.  Existing noise is never rewritten,
                # and transparent instructions get no channel of their own.
                new_qc.append(instr)
                continue

            if role == "measure":
                # --- X_ERROR BEFORE the measurement ---
                # The bit flip has to happen before the qubit is read out, so
                # that it can actually corrupt the outcome.
                if not (only_necessary and _already_annotated(idx, "X_ERROR", qubits, before=True)):
                    # One channel per qubit: X_ERROR is single-qubit, while a
                    # measurement instruction may in principle carry several.
                    for q in qubits:
                        new_qc.append(StimNoiseGate("X_ERROR", X_error_strength), [q])
                new_qc.append(instr)
                continue

            if role == "reset":
                # --- X_ERROR AFTER the reset ---
                # Mirror image of the measurement: a reset defines the state, so
                # the imperfection has to be applied to the fresh state.
                new_qc.append(instr)
                if not (only_necessary and _already_annotated(idx, "X_ERROR", qubits, before=False)):
                    for q in qubits:
                        new_qc.append(StimNoiseGate("X_ERROR", X_error_strength), [q])
                continue

            # --- Regular gate: depolarizing channel after the gate ---
            # Gates are noisy in the sense that they leave the state slightly
            # wrong, hence the channel follows the gate.  Two-qubit gates get a
            # correlated DEPOLARIZE2 over both of their qubits rather than two
            # independent single-qubit channels.
            new_qc.append(instr)

            if len(qubits) == 2:
                stim_name, strength = "DEPOLARIZE2", depolarize_2_strength
            else:
                stim_name, strength = "DEPOLARIZE1", depolarize_1_strength

            if not (only_necessary and _already_annotated(idx, stim_name, qubits, before=False)):
                new_qc.append(StimNoiseGate(stim_name, strength), qubits)

        # Whatever is still owed belongs to trailing layers in which the qubit
        # never got touched again.  Passing one past the last time step empties
        # every cursor.
        for q in qc.qubits:
            _flush_idle(q, max(noise_layers, default=-1) + 1)

        return new_qc

    return _insert_stim_noise
