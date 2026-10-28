# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Reorder QuantumCircuit operations based on their causal dependencies.

Circuit Reordering
==================

The functions in this module reorder circuits so that measurements, resets,
and disentanglers are performed as early as possible.

The benefit is the width of the tensor factors the simulator carries. A factor
spanning n qubits holds 2**n amplitudes, so keeping the simulation small means
splitting qubits off into their own factors as soon as they become separable.

Measurements, resets and disentanglers mark where a qubit is *likely* to be
separable, not where it is guaranteed to be. A measurement often indicates that
the algorithm has brought the qubit into a computational basis state, but
nothing about measuring a qubit forces that to be the case. Splitting there is
therefore speculative: the simulator attempts to factor the qubit out and leaves
the state untouched if it cannot (see
:mod:`~qrisp.simulator.preprocessing.disentangling`). For the many algorithms
that do measure computational basis states the assumption holds often enough to
be very effective, so performing these operations at the earliest point their
prerequisites allow keeps the factors narrower for everything that follows.
Where a split does collapse a factor onto a single outcome, the alternative
drops out of the simulation altogether.

This applies to measurements even though :func:`~qrisp.simulator.simulator.run`
does not execute them in place: the multiverse rewrite in
:mod:`~qrisp.simulator.preprocessing.measurement_handling` defers the outcome
onto an ancilla and leaves a disentangler behind at that point, so an early
measurement still becomes an early attempt to split.
:func:`~qrisp.simulator.simulator.single_shot_sim` is the one case where nothing
is speculative: it measures inline, samples an outcome and collapses the state,
so the measured qubit really is separable afterwards.

Causal Graph
------------

The ordering is achieved by converting the circuit into a directed acyclic
graph (the causal graph), where two successive operations with overlapping
qubits are represented by two nodes connected by a directed edge. The edge
points in the *opposite* direction of the sequential order of the operations.
This way, to evaluate which gates are necessary to perform a measurement,
disentangling, or reset, we simply look at the set of gates reachable from
that node in the causal graph.

Consider the following circuit::

              ┌───┐
     qubit_9: ┤ X ├──■─────
              ├───┤┌─┴─┐┌─┐
    qubit_10: ┤ Y ├┤ x ├┤M├
              ├───┤└───┘└╥┘
    qubit_11: ┤ H ├──────╫─
              └───┘      ║
     clbit_0: ═══════════╩═

In order to perform the measurement, we have to execute the X and CX gates.
The corresponding causal graph is::

    (measure) -> (CX) -> (X)
                   L---> (Y)
    (H)

We see that CX, X, and Y are reachable from the measurement. The gates required
to reach a given node are therefore emitted by collecting that node's
descendants, ordering them by descending topological index, and appending the
node itself.

Ordering by topological index is what keeps the result a legitimate reordering.
Two operations sharing a qubit stand in a causal relationship, and the later of
the two always carries the greater topological index, so replaying a descendant
set from the highest index downwards respects every such relationship. The
particular topological order consequently determines the circuit that comes out:
a layered, generation-by-generation order is used, and substituting a different
valid order (node indices in descending order would be one) changes the output
of this pass.

Representation
--------------

The graph is held as a CSR adjacency structure rather than as a graph object:
two arrays ``indptr`` and ``indices``, where the successors of node ``i`` are
``indices[indptr[i]:indptr[i + 1]]``. Neighbour lookup is then a slice of a
contiguous array, which lets the topological sort, the descendant counting and
the traversal itself all run as Numba-jitted loops over plain integer arrays.

Consumed nodes are flagged in a ``consumed`` array instead of being deleted.
Consuming a node always consumes its whole descendant set as well, so skipping
consumed nodes during a walk is equivalent to having removed them from the
graph, and no adjacency data ever has to be rebuilt. Filtering *after* a walk
would not do: it is the walk itself that costs, and leaving consumed nodes
reachable turns the loop from O(V + E) overall into O(len(preferential nodes) *
(V + E)).
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange

from qrisp.circuit import Operation, QuantumCircuit

# Counting descendants dominates this pass, and the work it costs is roughly the number
# of preferential nodes times the size of the graph. Above this many of those units the
# parallel kernel is used, below it the serial one.
#
# The threshold exists because the first prange call in a process spins up Numba's
# thread pool, which costs on the order of 0.3s -- far more than the whole pass takes on
# a small circuit. A single call only repays that spin-up somewhere around 1e8 units, so
# below the threshold the serial kernel runs and the thread pool is never touched at
# all. Note that the cost is per process, not per call, so a session doing many
# reorderings comes out ahead sooner than this bound suggests.
_PARALLEL_WORK_THRESHOLD = 10**8


def _build_causal_csr(
    qc: QuantumCircuit, preferential_gates: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the causal graph of a circuit as a CSR adjacency structure.

    Edges run against circuit order, so the successors of a node are the
    instructions that have to be executed *before* it and its descendant set is
    exactly the set of gates needed to reach it.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit whose causal graph is built. One node per instruction, numbered
        by position in ``qc.data``.
    preferential_gates : list[str]
        Gate names that should be pulled forward. Collected together with the
        ``final_op`` sentinels that :func:`_reorder_circuit` appends.

    Returns
    -------
    indptr : np.ndarray
        Offsets into ``indices``, of length ``len(qc.data) + 1``.
    indices : np.ndarray
        Concatenated successor lists.
    pref_nodes : np.ndarray
        The preferential nodes, in circuit order.
    pref_is_final : np.ndarray
        Boolean mask marking which of ``pref_nodes`` are ``final_op`` sentinels.

    """
    n = len(qc.data)
    indptr = np.zeros(n + 1, dtype=np.int64)
    successors: list[int] = []
    pref_nodes: list[int] = []
    pref_is_final: list[bool] = []

    # The most recent instruction to touch each qubit and each clbit. A new
    # instruction depends exactly on those, since any earlier dependency is already
    # implied transitively.
    current_node_qubits: dict = {}
    current_node_clbits: dict = {}

    preferential = list(preferential_gates) + ["final_op"]

    for node, instruction in enumerate(qc.data):
        node_set = []
        for qb in instruction.qubits:
            if qb in current_node_qubits:
                node_set.append(current_node_qubits[qb])
            current_node_qubits[qb] = node
        for cb in instruction.clbits:
            if cb in current_node_clbits:
                node_set.append(current_node_clbits[cb])
            current_node_clbits[cb] = node

        # An instruction whose operands were all last touched by the same
        # instruction would otherwise record that successor more than once.
        successors.extend(set(node_set))
        indptr[node + 1] = len(successors)

        if instruction.op.name in preferential:
            pref_nodes.append(node)
            pref_is_final.append(instruction.op.name == "final_op")

    return (
        indptr,
        np.array(successors, dtype=np.int64) if successors else np.empty(0, dtype=np.int64),
        np.array(pref_nodes, dtype=np.int64),
        np.array(pref_is_final, dtype=np.bool_),
    )


@njit(cache=True)
def _topological_order(indptr: np.ndarray, indices: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Return a layered topological order of the causal graph, plus how many nodes it covers.

    Kahn's algorithm, emitting one generation of zero-in-degree nodes at a time. The
    returned count is less than ``n`` only if the graph contains a cycle, which a
    causal graph cannot -- every edge points from a later instruction to an earlier
    one -- so the caller treats a short order as a malformed circuit.
    """
    # In-degree of every node, obtained by sweeping the whole adjacency array once.
    # CSR gives us out-edges, so this is the only way round to get in-degrees, and it
    # costs one linear pass over indices.
    indeg = np.zeros(n, dtype=np.int64)
    for node in range(n):
        for edge in range(indptr[node], indptr[node + 1]):
            indeg[indices[edge]] += 1

    # `current` and `following` hold the generation being emitted and the one being
    # accumulated. A single generation can in principle contain every node, so both are
    # allocated at full width rather than grown.
    order = np.empty(n, dtype=np.int64)
    current = np.empty(n, dtype=np.int64)
    following = np.empty(n, dtype=np.int64)

    # Seed with the nodes nothing points at. Edges run against circuit order, so these
    # are the instructions no later instruction depends on -- the trailing ones.
    n_current = 0
    for node in range(n):
        if indeg[node] == 0:
            current[n_current] = node
            n_current += 1

    pos = 0
    while n_current > 0:
        n_following = 0
        for i in range(n_current):
            node = current[i]
            order[pos] = node
            pos += 1
            # Releasing a node's edges may expose successors whose last remaining
            # dependency it was; those form the next generation.
            for edge in range(indptr[node], indptr[node + 1]):
                child = indices[edge]
                indeg[child] -= 1
                if indeg[child] == 0:
                    following[n_following] = child
                    n_following += 1
        for i in range(n_following):
            current[i] = following[i]
        n_current = n_following

    # Every node is emitted exactly once, at the moment its in-degree reaches zero. A
    # node on a cycle never gets there, which is why a short `pos` signals one.
    return order, pos


@njit(cache=True)
def _descendant_counts(
    indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, n: int, skip: np.ndarray
) -> np.ndarray:
    """Count the descendants of each given node, leaving entries flagged in ``skip`` at zero.

    One iterative depth-first walk per node. ``mark`` is stamped with a per-node
    counter rather than being cleared between walks, so the whole routine allocates
    nothing inside the loop.
    """
    # `mark` records which walk last reached a node. Clearing an n-wide array before
    # every walk would cost more than the walks themselves on a sparse graph, so each
    # walk gets a fresh stamp instead and a node counts as unseen whenever its mark
    # differs from the current one. Zero-initialised means "never reached", which no
    # stamp equals because stamps start at one.
    mark = np.zeros(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    counts = np.zeros(len(nodes), dtype=np.int64)
    stamp = 0

    for i in range(len(nodes)):
        if skip[i]:
            continue

        stamp += 1
        start = nodes[i]
        mark[start] = stamp
        stack[0] = start
        n_stack = 1
        count = 0

        # An explicit stack rather than recursion: the walk can be as deep as the
        # circuit is long, which would overflow the call stack on a real circuit. Only
        # the size of the reachable set matters here, not the order it is visited in,
        # so depth-first with a plain LIFO stack is fine.
        #
        # Each node is pushed at most once per stamp, so the stack cannot exceed n.
        while n_stack > 0:
            n_stack -= 1
            node = stack[n_stack]
            for edge in range(indptr[node], indptr[node + 1]):
                child = indices[edge]
                if mark[child] != stamp:
                    mark[child] = stamp
                    count += 1
                    stack[n_stack] = child
                    n_stack += 1

        counts[i] = count

    return counts


@njit(cache=True, parallel=True)
def _descendant_counts_parallel(  # noqa: PLR0913, PLR0917
    indptr: np.ndarray,
    indices: np.ndarray,
    nodes: np.ndarray,
    n: int,
    skip: np.ndarray,
    n_threads: int,
) -> np.ndarray:
    """Parallel counterpart of :func:`_descendant_counts`, producing identical counts.

    The walks are independent, so each thread strides through its own share of the node
    list with private ``mark`` and ``stack`` buffers and writes only its own entries of
    ``counts``. Nothing is shared, so no synchronisation is needed.

    ``n_threads`` is a parameter rather than a ``get_num_threads()`` call inside the
    body on purpose: that call is a ctypes entry point into the threading layer, which
    Numba classifies as a dynamic global and which would make this function
    uncacheable, costing a recompile in every process.
    """
    counts = np.zeros(len(nodes), dtype=np.int64)

    # prange over threads rather than over nodes, so the scratch buffers below are
    # allocated once per thread instead of once per walk.
    for thread in prange(n_threads):
        # Per-thread scratch: 2 * n int64 each, so the memory cost of going parallel
        # scales with the thread count. Keeping them private is what removes the need
        # for any synchronisation -- the stamping trick from the serial kernel would
        # otherwise be a data race.
        mark = np.zeros(n, dtype=np.int64)
        stack = np.empty(n, dtype=np.int64)
        stamp = 0

        # Strided rather than split into contiguous blocks. `nodes` is in circuit
        # order, and a node's descendant set is its causal past, so the walks get
        # steadily more expensive towards the end of the circuit. Interleaving gives
        # every thread a mix of cheap and expensive walks; contiguous blocks would
        # leave the thread holding the tail of the circuit running alone.
        for i in range(thread, len(nodes), n_threads):
            if skip[i]:
                continue

            stamp += 1
            start = nodes[i]
            mark[start] = stamp
            stack[0] = start
            n_stack = 1
            count = 0

            # Identical walk to the serial kernel; see there for the stamping and the
            # bound on the stack.
            while n_stack > 0:
                n_stack -= 1
                node = stack[n_stack]
                for edge in range(indptr[node], indptr[node + 1]):
                    child = indices[edge]
                    if mark[child] != stamp:
                        mark[child] = stamp
                        count += 1
                        stack[n_stack] = child
                        n_stack += 1

            counts[i] = count

    return counts


def _count_descendants(
    indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray, n: int, skip: np.ndarray
) -> np.ndarray:
    """Count descendants, choosing the serial or the parallel kernel by problem size.

    Both kernels return the same counts; see :data:`_PARALLEL_WORK_THRESHOLD` for why
    the small cases are deliberately kept off the parallel path.
    """
    work = len(nodes) * (n + len(indices))
    if work < _PARALLEL_WORK_THRESHOLD:
        return _descendant_counts(indptr, indices, nodes, n, skip)

    # Only reached for large problems, so the threading layer is not touched at all by
    # the serial path above. More threads than nodes to walk would just allocate
    # scratch buffers that never get used.
    n_threads = min(get_num_threads(), max(1, len(nodes)))
    if n_threads == 1:
        return _descendant_counts(indptr, indices, nodes, n, skip)

    return _descendant_counts_parallel(indptr, indices, nodes, n, skip, n_threads)


@njit(cache=True)
def _traverse(
    indptr: np.ndarray, indices: np.ndarray, tp_index: np.ndarray, eval_nodes: np.ndarray, n: int
) -> tuple[np.ndarray, int]:
    """Emit the reordered node sequence, plus its length.

    For each preferential node in turn, collect the descendants that have not been
    consumed yet, emit them by descending topological index, then emit the node
    itself. A node already consumed by an earlier traversal is skipped: it was part
    of that node's causal past and has been emitted there.
    """
    # Two separate bookkeeping arrays, with distinct lifetimes:
    #
    #   `consumed` is global to the whole routine and permanent -- it says a node has
    #   already been emitted, and stands in for deleting it from the graph.
    #   `mark` is per-traversal and only deduplicates within one walk, since the causal
    #   graph is not a tree and a node is reachable by several paths.
    consumed = np.zeros(n, dtype=np.bool_)
    emission = np.empty(n, dtype=np.int64)
    n_emitted = 0

    mark = np.zeros(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    descendants = np.empty(n, dtype=np.int64)

    for i in range(len(eval_nodes)):
        start = eval_nodes[i]
        if consumed[start]:
            continue

        # The loop index doubles as the stamp: it is unique per iteration, and it is
        # never zero, which is the "never marked" value mark was initialised to. Skipped
        # iterations simply leave unused stamps behind.
        stamp = i + 1
        mark[start] = stamp
        stack[0] = start
        n_stack = 1
        n_desc = 0

        while n_stack > 0:
            n_stack -= 1
            node = stack[n_stack]
            for edge in range(indptr[node], indptr[node + 1]):
                child = indices[edge]
                # Consumed nodes are neither collected nor walked through. Not walking
                # through them is the half that matters: it is what makes the flag
                # equivalent to having removed the node, since a path that only reached
                # further nodes via this one is genuinely gone.
                if mark[child] != stamp and not consumed[child]:
                    mark[child] = stamp
                    descendants[n_desc] = child
                    n_desc += 1
                    stack[n_stack] = child
                    n_stack += 1

        # Descending topological index, obtained by sorting the negated indices
        # ascending. tp_index is a permutation, so no two descendants compare equal and
        # the sort needs no tie-breaking.
        keys = np.empty(n_desc, dtype=np.int64)
        for j in range(n_desc):
            keys[j] = -tp_index[descendants[j]]

        for j in np.argsort(keys):
            node = descendants[j]
            emission[n_emitted] = node
            n_emitted += 1
            consumed[node] = True

        # The preferential node goes last: everything it depends on has just been
        # emitted, which is the whole point of pulling it forward.
        emission[n_emitted] = start
        n_emitted += 1
        consumed[start] = True

    # Every node belongs to at most one traversal, so nothing is emitted twice. It
    # reaches n as long as every node is reachable from some preferential node, which
    # the final_op sentinels appended by the caller guarantee for any instruction acting
    # on at least one qubit. An instruction acting on no qubit at all is reachable from
    # nothing and is therefore dropped -- the pass has always behaved this way, and the
    # only such operation, "barrier", is removed before this point by
    # _count_measurements_and_treat_alloc.
    return emission, n_emitted


def _reorder_circuit(qc: QuantumCircuit, preferential_gates: list[str] | None = None) -> QuantumCircuit:
    """
    Reorder the given quantum circuit based on a topological sorting of its causal graph.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to be reordered.
    preferential_gates : list[str], optional
        List of gate names to prioritize during reordering. Defaults to None.

    Returns
    -------
    QuantumCircuit
        The reordered quantum circuit.

    """
    if preferential_gates is None:
        preferential_gates = []

    # A sentinel per qubit, so that the gates trailing each qubit are covered by some
    # preferential node and therefore end up in the output. They are stripped again
    # below, and removed from the incoming circuit before returning.
    for qb in qc.qubits:
        qc.append(Operation("final_op", num_qubits=1), [qb])

    n = len(qc.data)
    indptr, indices, pref_nodes, pref_is_final = _build_causal_csr(qc, preferential_gates)

    # Preferential nodes are processed cheapest first: a measurement with few
    # descendants needs only a few gates simulated before it can be executed. The
    # final_op sentinels are not real operations and carry no such cost, so they sort
    # last on a sentinel key -- which also means their descendant sets never have to
    # be computed. The sort is stable to keep sentinels in qubit order.
    counts = _count_descendants(indptr, indices, pref_nodes, n, pref_is_final)
    keys = np.where(pref_is_final, np.iinfo(np.int64).max, counts)
    eval_nodes = pref_nodes[np.argsort(keys, kind="stable")]

    order, covered = _topological_order(indptr, indices, n)
    if covered != n:
        raise RuntimeError("Causal graph of the given circuit is not acyclic")
    tp_index = np.empty(n, dtype=np.int64)
    tp_index[order] = np.arange(n, dtype=np.int64)

    emission, n_emitted = _traverse(indptr, indices, tp_index, eval_nodes, n)

    data = qc.data
    new_qc = qc.clearcopy()
    new_qc.data = [data[int(node)] for node in emission[:n_emitted] if data[int(node)].op.name != "final_op"]

    # Remove the sentinels appended above
    for _ in range(len(qc.qubits)):
        qc.data.pop(-1)

    return new_qc
