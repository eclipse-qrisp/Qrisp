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

**Phase 1 - schedule.**  The time steps come from
:func:`~qrisp.circuit.pass_management.scheduling.asap_layers`, the same
scheduler :func:`~qrisp.layerize` uses.  Sharing it is the whole point:  the
layers this pass inserts noise *for* are then exactly the layers ``layerize``
later draws, so the pipeline

.. code-block::

    insert_stim_noise -> layerize(insert_barriers=True) -> to_stim

renders one column and one ``TICK`` per time step with each qubit's single
channel visible in it - which is how the inserted model is checked.  A private
scheduler here would put the two passes into disagreement and the picture would
stop verifying anything.

Two properties of that scheduler carry most of the weight:

* Error channels and annotations (``qb_alloc`` / ``qb_dealloc``, ``gphase``,
  ``parity``, barriers) take no time step of their own.  This is what keeps a
  user-placed noise instruction attached to the layer of the gate it annotates,
  and what stops classical annotations such as ``parity`` from inventing time
  steps that would then collect a full round of idle noise.
* A ``barrier`` ends the layer of the qubits it *names* and leaves the rest
  alone.  Only a full-width barrier is therefore a global time boundary - which
  is also exactly when it becomes a Stim ``TICK``.  Use
  :func:`~qrisp.promote_barriers` to widen local fences beforehand if global
  boundaries are what is wanted.

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
* at a ``barrier`` naming ``q``, so that no idle noise crosses a fence declared
  over that qubit.  A full-width barrier is a global time boundary (it becomes a
  Stim ``TICK``) and hence flushes *every* qubit; a partial barrier flushes only
  the qubits it names, since it says nothing about the others,
* at the end of the circuit for whatever is left over.

Soundness rests on a monotonicity property of the schedule: consecutive
instructions on a resource are linked in the scheduler's dependency graph, so
for a fixed qubit the instructions touching it appear in ``qc.data`` with
non-decreasing layer indices.  Emitting ``q``'s noise for layer ``L`` at some
position is therefore correct as soon as ``q``'s next instruction lies in a
layer ``> L``, which is what both flush sites establish - the second one because
a barrier is itself part of the chain of every qubit it names, so nothing on
those qubits can follow it in an earlier layer.  This is also why the barrier
flush has to be width-aware: a partial barrier is *not* in the chain of the
qubits it omits, and flushing those would emit their noise ahead of instructions
that still belong to earlier layers.

A single forward cursor per qubit (``idle_pos``) is hence enough, and no
obligation is ever discharged into the past.  The cursor indexes the shared,
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
  :func:`~qrisp.layerize` pass, which can be run afterwards.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.pass_management.scheduling import (
    asap_layers,
    is_error_channel,
    is_transparent,
)
from qrisp.circuit.quantum_circuit import QuantumCircuit, is_full_width_barrier


def insert_stim_noise(
    depolarize_1_strength: float = 0.01,
    depolarize_2_strength: float = 0.01,
    X_error_strength: float = 0.01,
    only_necessary: bool = True,
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    """Create a pass that inserts a circuit-level Stim noise model into a circuit.

    The circuit is organized into layers (time steps) using the same
    as-soon-as-possible (ASAP) schedule as :func:`~qrisp.layerize`: instructions
    that act on disjoint qubits share a layer, instructions that touch a common
    qubit are placed in successive layers.  Every qubit then receives **exactly
    one** noise instruction per layer, according to the following rules:

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
    qubit (or at the next barrier naming it, or at the end of the circuit).
    Since no operation on that qubit intervenes, this is equivalent to emitting
    it at the end of its layer.  Because this pass and :func:`~qrisp.layerize`
    share their scheduler, running ``layerize`` afterwards puts every channel
    back into the time step it was inserted for:

    .. code-block::

        insert_stim_noise -> layerize(insert_barriers=True) -> to_stim

    gives one ``TICK`` per time step with each qubit's single channel in it,
    which is the recommended way to look at the result.

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
        When ``True``, the pass only inserts noise where the circuit does not
        already contain it: no channel is inserted where the circuit already
        carries an **exactly matching** one, that is a
        :class:`~qrisp.misc.stim_tools.StimNoiseGate` of the same Stim type
        acting on exactly the same qubits, directly after the gate, directly
        before the measurement, or directly after the reset.  "Directly" is
        meant per qubit: an instruction on an unrelated qubit may sit in between
        without breaking the match, since it says nothing about the qubits the
        channel annotates.  The error probability is not compared, so a
        user-placed channel of a different strength is respected.  Anything else
        (a different channel type, a different set of qubits, a channel on the
        other side of the instruction) does not count as a match and the noise
        model is applied regardless — a ``DEPOLARIZE1`` after an entangling gate
        earns no special treatment and the ``DEPOLARIZE2`` goes in anyway.  Each
        channel annotates at most one instruction; see the note below on
        ``reset`` / ``X_ERROR`` / ``measure``.  When ``False``, the full noise
        model is inserted unconditionally.

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
    * A ``barrier`` acts as a synchronization point *between* layers: it
      enforces that instructions before and after it are scheduled into separate
      layers, but it is not a noise layer itself — it neither receives a noise
      instruction nor is a time step of its own.  It constrains exactly the
      qubits it names, so all outstanding idle noise of *those* qubits is
      emitted before it.  A full-width barrier names the whole register and is
      therefore a global time boundary (this is also precisely when it becomes a
      Stim ``TICK``); a partial barrier is a local fence and lets an instruction
      on an unnamed qubit stay in an earlier layer.  Use
      :func:`~qrisp.promote_barriers` to widen local fences into global
      boundaries before inserting noise — note that this widens the schedule and
      hence inflates the idle-noise budget.
    * Existing :class:`~qrisp.misc.stim_tools.StimNoiseGate` instructions are
      always left untouched.  With ``only_necessary=True`` (the default) they
      also consume the noise slot of their qubits: no additional idle noise
      is stacked on top of them, and no duplicate channel is inserted
      directly after the gate / before the measurement / after the reset they
      already annotate.  A fully hand-annotated circuit therefore comes out
      unchanged.  With ``only_necessary=False`` the full noise model is applied
      on top of any user-placed noise.
    * *A channel annotates at most one instruction.*  The rules of the model
      overlap in position — the channel after a reset is also the channel before
      whatever reads the qubit next — so ``reset``, ``X_ERROR``, ``measure``
      written in that order has both instructions expecting exactly that
      ``X_ERROR``.  Letting them share it would leave one of their two time steps
      with no noise at all, so it is attributed to the instruction whose time step
      it was scheduled into (the ``reset``) and the other one is amended with a
      channel of its own.
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
      before the next instruction that touches the qubit, at the next barrier
      naming it, or at the end of the circuit.  Nothing acts on the qubit in
      between, so the resulting channels compose identically — only the layout
      of the printed circuit differs.  Running ``layerize(insert_barriers=True)``
      afterwards distributes the channels back over the timeline, one per time
      step, which is easier to read and to check.

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

    Repeating the round and rendering the Stim timeline diagram shows the model
    in context.  The "after" diagram is the output of
    ``layerize(insert_barriers=True)``, so it carries one ``TICK`` per time step
    and the model can be read off column by column — every qubit holding exactly
    one channel in each:

    .. image:: /_static/insert_stim_noise_before.svg
       :alt: Stim timeline diagram of two noiseless syndrome extraction rounds

    .. image:: /_static/insert_stim_noise_after.svg
       :alt: Stim timeline diagram of the same rounds after insert_stim_noise and layerize

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
    receives its ``DEPOLARIZE1`` for that layer.

    "Directly after" is meant per qubit, not per list position, so an instruction
    on an unrelated qubit may sit in between without breaking the match::

        >>> qc = QuantumCircuit(3)
        >>> qc.cx(0, 1)
        >>> qc.h(2)                                                    # unrelated
        >>> qc.append(StimNoiseGate("DEPOLARIZE2", 0.05), qc.qubits[:2])
        >>> print(insert_stim_noise(0.01, 0.02, 0.001)(qc).to_stim())
        CX 0 1
        H 2
        DEPOLARIZE1(0.01) 2
        DEPOLARIZE2(0.05) 0 1

    The match does have to be exact, though.  Had the user placed a
    ``DEPOLARIZE1``, or a ``DEPOLARIZE2`` on a different pair of qubits, the
    model's own ``DEPOLARIZE2`` would still be inserted::

        >>> qc = QuantumCircuit(2)
        >>> qc.cx(0, 1)
        >>> qc.append(StimNoiseGate("DEPOLARIZE1", 0.05), [qc.qubits[0]])
        >>> print(insert_stim_noise(0.01, 0.02, 0.001)(qc).to_stim())
        CX 0 1
        DEPOLARIZE2(0.02) 0 1
        DEPOLARIZE1(0.05) 0

    Pass ``only_necessary=False`` to apply the full model unconditionally.

    **Composition — reading the model off the output.**  The pass belongs in a
    :class:`~qrisp.PassManager` together with :func:`~qrisp.layerize`, which
    reorders the result into layer order and marks each time step with a
    ``TICK``::

        >>> from qrisp import PassManager, layerize
        >>> pm = PassManager()
        >>> pm += insert_stim_noise(depolarize_1_strength=1e-3,
        ...                         depolarize_2_strength=1e-3,
        ...                         X_error_strength=1e-3)
        >>> pm += layerize(insert_barriers=True)

    Because both passes share their scheduler, the layers ``layerize`` draws are
    the layers the noise was inserted for.  The output therefore shows the model
    one time step at a time, and every qubit appearing exactly once per step is
    the statement of the contract::

        >>> qc = QuantumCircuit(3)
        >>> qc.cx(0, 1)
        >>> qc.cx(2, 1)
        >>> qc.measure(1)
        >>> qc.reset(1)
        >>> print(pm.run(qc).to_stim())
        CX 0 1
        DEPOLARIZE2(0.001) 0 1
        DEPOLARIZE1(0.001) 2
        TICK
        CX 2 1
        DEPOLARIZE2(0.001) 2 1
        DEPOLARIZE1(0.001) 0
        TICK
        X_ERROR(0.001) 1
        DEPOLARIZE1(0.001) 0 2
        M 1
        TICK
        R 1
        X_ERROR(0.001) 1
        DEPOLARIZE1(0.001) 0 2
        TICK

    Four time steps, four ``TICK``\\ s, and in each of them qubits ``0``, ``1``
    and ``2`` carry one channel each — the ancilla's gate or readout error, the
    data qubits' idle noise.  A miscounted layer would show up here as a qubit
    missing from a step or appearing twice in one.

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
        layers = asap_layers(qc)

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

            # The order of these tests matters.  ``is_transparent`` is true for
            # barriers and for error channels as well - both take no time step -
            # so the two roles that need telling apart from "rides along at no
            # cost" have to be recognised first.
            if name == "barrier":
                roles.append("barrier")
            elif is_error_channel(instr):
                # Noise the circuit already contains.  Never touched, and with
                # ``only_necessary`` it also occupies its qubits' noise slot.
                roles.append("noise")
            elif is_transparent(instr):
                roles.append("transparent")
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
        #
        # ``chain[qubit]``: the instructions acting on that qubit, in order, with
        # the ones that take no time step left out.  This is the per-qubit thread
        # of the scheduler's dependency graph, and it is what "the channel right
        # after this gate" has to mean: positions in ``qc.data`` say nothing, since
        # an instruction on an unrelated qubit may sit between a gate and the
        # channel annotating it.  ``chain_pos`` records where each instruction sits
        # in the thread of each of its qubits, so a neighbour is an index lookup.
        noise_layers: set[int] = set()
        handled: dict[int, set] = defaultdict(set)
        chain: dict[object, list[int]] = defaultdict(list)
        chain_pos: list[dict] = [{} for _ in data]

        for idx, instr in enumerate(data):
            if roles[idx] == "transparent":
                # Skipped rather than threaded, so that a ``gphase`` or an
                # allocation marker between a gate and its channel does not hide
                # the one from the other.  Barriers *are* threaded: a fence ends
                # the time step, so nothing across it annotates anything.
                continue
            for q in instr.qubits:
                chain_pos[idx][q] = len(chain[q])
                chain[q].append(idx)
            if roles[idx] == "barrier":
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

        # Channels of the input that have been accepted as some instruction's
        # annotation.  A channel annotates one instruction, never two - see
        # ``_claim``.
        claimed: set[int] = set()

        def _neighbour(idx: int, qubit, before: bool) -> int | None:
            """Return the instruction next to *idx* on *qubit*'s thread, if any."""
            pos = chain_pos[idx][qubit] + (-1 if before else 1)
            if 0 <= pos < len(chain[qubit]):
                return chain[qubit][pos]
            return None

        def _claim(idx: int, stim_name: str, qubits, before: bool) -> bool:
            """Accept an existing channel as the annotation of *idx*, if there is one.

            The channel has to sit directly before (``before=True``) or directly
            after the instruction on the thread of every one of *qubits*, be of
            Stim type *stim_name*, and act on exactly *qubits*.  The error
            probability is deliberately not compared, so a hand-placed channel of
            a different strength is respected.
            """
            # The type has to match exactly.  A ``DEPOLARIZE1`` after an
            # entangling gate is not the channel the model calls for, so it earns
            # no special treatment and the ``DEPOLARIZE2`` goes in regardless.
            #
            # The claim is exclusive because the rules overlap in position: the
            # channel after a reset is also the channel before whatever reads the
            # qubit next.  Written out, ``reset -> X_ERROR -> measure`` has both
            # instructions expecting exactly this channel, and letting them share
            # it would leave one of their two time steps with no noise at all.
            # Whoever comes first on the thread takes it - which is the one the
            # channel shares a layer with, since a channel joins the layer of the
            # instruction it follows - and the other one is amended below.
            neighbours = {_neighbour(idx, q, before) for q in qubits}
            if len(neighbours) != 1:
                # The qubits disagree about what sits next to them, so whatever is
                # there does not annotate this instruction as a whole.
                return False
            (j,) = neighbours
            if j is None or j in claimed or roles[j] != "noise":
                return False
            adj = data[j]
            if adj.op.stim_name != stim_name or set(adj.qubits) != set(qubits):
                return False
            claimed.add(j)
            return True

        for idx, instr in enumerate(data):
            role = roles[idx]
            layer = layers[idx]
            qubits = instr.qubits

            if role == "barrier":
                # A barrier flushes the qubits it names: it ends their layer, so
                # this is the last position at which their outstanding noise can
                # still be emitted on the correct side of the fence.  A
                # full-width barrier names the whole register and is a global
                # time boundary (it becomes a Stim TICK), so it flushes
                # everything - which is also what keeps the flush from falling
                # arbitrarily far behind in a QEC circuit with one barrier per
                # round.
                #
                # The width matters for correctness, not just for tidiness.  A
                # partial barrier is no constraint on the qubits it omits, and
                # those may well carry later instructions belonging to *earlier*
                # layers; flushing them here would emit their noise ahead of
                # those instructions.
                # Note the ``+ 1``: a barrier sits *in* the layer it ends, unlike
                # every other instruction here, which is flushed up to the layer
                # it is about to occupy.  The obligations for the barrier's own
                # layer therefore have to go out too, or they would be emitted on
                # the far side of a fence that was meant to close them in.
                flush_qubits = qc.qubits if is_full_width_barrier(instr, qc) else qubits
                for q in flush_qubits:
                    _flush_idle(q, layer + 1)
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
                #
                # One channel per qubit: X_ERROR is single-qubit, while a
                # measurement instruction may in principle carry several, so each
                # is claimed and amended on its own.
                for q in qubits:
                    if not (only_necessary and _claim(idx, "X_ERROR", [q], before=True)):
                        new_qc.append(StimNoiseGate("X_ERROR", X_error_strength), [q])
                new_qc.append(instr)
                continue

            if role == "reset":
                # --- X_ERROR AFTER the reset ---
                # Mirror image of the measurement: a reset defines the state, so
                # the imperfection has to be applied to the fresh state.
                new_qc.append(instr)
                for q in qubits:
                    if not (only_necessary and _claim(idx, "X_ERROR", [q], before=False)):
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

            if not (only_necessary and _claim(idx, stim_name, qubits, before=False)):
                new_qc.append(StimNoiseGate(stim_name, strength), qubits)

        # Whatever is still owed belongs to trailing layers in which the qubit
        # never got touched again.  Passing one past the last time step empties
        # every cursor.
        for q in qc.qubits:
            _flush_idle(q, max(noise_layers, default=-1) + 1)

        return new_qc

    return _insert_stim_noise
