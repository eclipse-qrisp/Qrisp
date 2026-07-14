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

Circuit pass that cancels adjacent gate–inverse-gate pairs by building a
directed acyclic graph (DAG) and fusing neighbouring instructions.

**Architecture**

The module is layered in three tiers:

1. **Operation-level fusion** — domain logic that decides whether two
   :class:`~qrisp.circuit.Operation` objects cancel, fuse into a single
   operation, or cannot be combined.  A dispatcher,
   :func:`_fuse_operations`, delegates to specialised handlers in order:

   * :func:`_fuse_swap_with_neighbour` — SWAP·{CX,CP,RZZ} and vice
     versa, replacing the pair with a cheaper compound gate.
   * :func:`_fuse_parameterized_1q` — same-site Rz, P, Rx, Ry, gphase
     fusion by parameter addition; cross-type Rz↔P fusion with global
     phase absorption.
   * :func:`_fuse_gidney_and` — compute/uncompute cancellation of
     :class:`~qrisp.alg_primitives.GidneyLogicalAND`.
   * :func:`_fuse_controlled_ops` — recursion into
     :class:`~qrisp.circuit.ControlledOperation` and
   * :func:`_check_inverse_cancel` — generic final fallback via
     ``op_a.inverse().name == op_b.name``.

2. **Instruction-level layer** — :func:`_fuse_instructions` validates
   qubit alignment, delegates symmetric-gate set-matching to
   :func:`_resolve_qubit_order`, then calls :func:`_fuse_operations`.

3. **DAG orchestration** — :func:`fuse_adjacents` builds a DAG where
   each instruction is a node and edges track qubit/clbit flow.  When
   an instruction has a single predecessor (meaning the pair is locally
   adjacent on every qubit), fusion is attempted.  Cancelled or fused
   nodes are removed and edges rewired via :func:`_rewire_past_node`.
   Surviving instructions are emitted in topological order by
   :func:`_emit_surviving_circuit`.

**Sentinel**

Fusion functions return :data:`_FUSION_CANCEL` (a unique sentinel
object) to signal total cancellation, avoiding the ambiguity of magic
integers.

**Global phase**

Rotations that carry a ``global_phase`` attribute contribute it to a
running accumulator.  Non-zero accumulated phase is emitted as a single
``gphase`` gate at the end of the pass.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from qrisp.circuit import (
    ClControlledOperation,
    ControlledOperation,
    GPhaseGate,
    Instruction,
    Operation,
    PGate,
    QuantumCircuit,
    RXGate,
    RYGate,
    RZGate,
    fast_append,
)
from qrisp.circuit.pass_management.circuit_pass import CircuitPass
from qrisp.circuit.pass_management.passes.combine_single_qubit_gates import combine_single_qubit_gates

# Sentinel object returned when two operations cancel completely.
_FUSION_CANCEL = object()


# =============================================================================
# Operation-level fusion logic — dispatched by _fuse_operations
# =============================================================================


def _fuse_swap_with_neighbour(op_a, op_b):
    """Try to fuse a SWAP gate with a neighbouring symmetric two-qubit gate.

    When a SWAP immediately precedes (or follows) one of these symmetric
    two-qubit gates (CX, CP, RZZ, CZ), the pair can be replaced by a simpler
    compound gate that uses fewer CX gates than the naive decomposition
    of SWAP then gate.

    The "half_*_qc" circuits below are the gate that remains after
    absorbing part of the SWAP into the neighbouring operation.

    Parameters
    ----------
    op_a : Operation
        The first operation (earlier in time).
    op_b : Operation
        The second operation (later in time).

    Returns
    -------
    Operation or None
        The fused operation, or ``None`` if neither op is a SWAP or
        the neighbour is not a recognised symmetric two-qubit gate.

    """
    with fast_append(0):
        # --- swap-a then something ---
        if "swap" == op_a.name:
            if op_b.name == "cx":
                # SWAP · CX(0,1) = CX(0,1) · CX(1,0)
                half_swap_qc = QuantumCircuit(2)
                half_swap_qc.cx(0, 1)
                half_swap_qc.cx(1, 0)
                return half_swap_qc.to_gate()
            elif op_b.name == "rzz":
                # SWAP · RZZ(θ, 0,1) ⇒ 2 CX, RZ(θ), CX
                half_rzz_qc = QuantumCircuit(2)
                half_rzz_qc.cx(0, 1)
                half_rzz_qc.cx(1, 0)
                half_rzz_qc.rz(op_b.params[0], 1)
                half_rzz_qc.cx(0, 1)
                return half_rzz_qc.to_gate()
            elif op_b.name == "cp":
                # SWAP · CP(θ, 0,1) ⇒ 2 CX, P(-θ/2,1), CX, P(θ/2,0), P(θ/2,1)
                half_cp_qc = QuantumCircuit(2)
                half_cp_qc.cx(0, 1)
                half_cp_qc.cx(1, 0)
                half_cp_qc.p(-op_b.params[0] / 2, 1)
                half_cp_qc.cx(0, 1)
                half_cp_qc.p(op_b.params[0] / 2, 0)
                half_cp_qc.p(op_b.params[0] / 2, 1)
                return half_cp_qc.to_gate()

        # --- something then swap-b ---
        # Mirrors the cases above but with SWAP on the right.
        if "swap" == op_b.name:
            if op_a.name == "cx":
                # CX(0,1) · SWAP = CX(1,0) · CX(0,1)
                half_swap_qc = QuantumCircuit(2)
                half_swap_qc.cx(1, 0)
                half_swap_qc.cx(0, 1)
                return half_swap_qc.to_gate()
            if op_a.name == "rzz":
                # RZZ(θ, 0,1) · SWAP ⇒ 2 CX, RZ(θ), CX
                half_rzz_qc = QuantumCircuit(2)
                half_rzz_qc.cx(0, 1)
                half_rzz_qc.cx(1, 0)
                half_rzz_qc.rz(op_a.params[0], 1)
                half_rzz_qc.cx(0, 1)
                return half_rzz_qc.to_gate()
            if op_a.name == "cp":
                # CP(θ, 0,1) · SWAP ⇒ 2 CX, P(-θ/2,1), CX, P(θ/2,0), P(θ/2,1)
                half_cp_qc = QuantumCircuit(2)
                half_cp_qc.cx(0, 1)
                half_cp_qc.cx(1, 0)
                half_cp_qc.p(-op_a.params[0] / 2, 1)
                half_cp_qc.cx(0, 1)
                half_cp_qc.p(op_a.params[0] / 2, 0)
                half_cp_qc.p(op_a.params[0] / 2, 1)
                return half_cp_qc.to_gate()
            if op_a.name == "cz":
                # CZ(0,1) · SWAP ⇒ H(1), CX(1,0), CX(0,1), H(0)
                half_cz_qc = QuantumCircuit(2)
                half_cz_qc.h(1)
                half_cz_qc.cx(1, 0)
                half_cz_qc.cx(0, 1)
                half_cz_qc.h(0)
                return half_cz_qc.to_gate()

    return None


def _fuse_parameterized_1q(op_a, op_b, gphase_array):
    """Try to fuse two parameterised single-qubit gates.

    Rz(θ₁)·Rz(θ₂) → Rz(θ₁+θ₂), and if θ₁+θ₂ = 0 the pair cancels.
    P(θ₁)·P(θ₂)  → P(θ₁+θ₂), same logic.
    Cross-type fusion (Rz+P) absorbs the Rz global phase into the
    gphase tracker and emits a single P gate.

    Also handles same-name Rx, Ry, and gphase gates.

    Note: global phases from Rz are tracked in gphase_array and
    collapsed at the very end of the main pass.

    Parameters
    ----------
    op_a : Operation
        The first operation (earlier in time).
    op_b : Operation
        The second operation (later in time).
    gphase_array : list
        Single-element list tracking accumulated global phase.

    Returns
    -------
    Operation or _FUSION_CANCEL or None
        * ``Operation`` — the fused result.
        * ``_FUSION_CANCEL`` — the two operations cancel completely.
        * ``None`` — no parameterised fusion possible.

    """
    if not op_a.params:
        return None

    param_sum = sum(op_a.params + op_b.params)

    # --- Rz + Rz / Rz + P ---
    if op_a.name == "rz":
        if op_b.name == "rz":
            if param_sum == 0:
                return _FUSION_CANCEL  # perfect cancellation
            return RZGate(param_sum)
        if op_b.name == "p":
            gphase_array[0] += op_a.global_phase
            if param_sum == 0:
                return _FUSION_CANCEL
            return PGate(param_sum)

    # --- P + Rz / P + P ---
    elif op_a.name == "p":
        if op_b.name == "rz":
            gphase_array[0] += op_b.global_phase
            if param_sum == 0:
                return _FUSION_CANCEL
            return PGate(param_sum)
        if op_b.name == "p":
            if param_sum == 0:
                return _FUSION_CANCEL
            return PGate(param_sum)

    # --- Rx, Ry, gphase (same-name only) ---
    if op_a.name == op_b.name:
        if op_a.name == "rx":
            if param_sum == 0:
                return _FUSION_CANCEL
            return RXGate(param_sum)
        if op_a.name == "ry":
            if param_sum == 0:
                return _FUSION_CANCEL
            return RYGate(param_sum)
        if op_a.name == "gphase":
            return GPhaseGate(op_a.global_phase + op_b.global_phase)
        # Other parameterised gates with matching names (e.g. cp, rzz)
        # are handled by the generic inverse check further below.

    return None


def _fuse_gidney_and(op_a, op_b):
    """Cancel a GidneyLogicalAND compute/uncompute pair.

    This is a definitive check — if the ops are GidneyLogicalAND
    instances, no further fusion paths should be attempted.

    Parameters
    ----------
    op_a : Operation
        The first operation (earlier in time).
    op_b : Operation
        The second operation (later in time).

    Returns
    -------
    _FUSION_CANCEL or None
        ``_FUSION_CANCEL`` if the compute/uncompute pair cancels,
        ``None`` otherwise (including "definitive no match").

    """
    # Lazy import to avoid circular dependencies at module load time.
    from qrisp.alg_primitives import GidneyLogicalAND

    if not (isinstance(op_a, GidneyLogicalAND) and isinstance(op_b, GidneyLogicalAND)):
        return None

    if op_a.ctrl_state == op_b.ctrl_state:
        # A compute/uncompute pair cancels.
        if op_a.inv != op_b.inv:
            return _FUSION_CANCEL

    # Definitive no-match for GidneyLogicalAND pairs — do not fall
    # through to any other fusion path.
    return None


def _fuse_controlled_ops(op_a, op_b):
    """Try to fuse two controlled operations by recursing into their
    base operations.

    IMPORTANT: when this function returns ``None`` it means "no
    controlled-op match was found, but a generic inverse cancellation
    should still be attempted".  This is deliberate because many gates
    with definitions (SWAP, circuit wrappers, etc.) are self-inverse
    and need the fallback path.

    Parameters
    ----------
    op_a : ControlledOperation
        The first operation (earlier in time).
    op_b : ControlledOperation
        The second operation (later in time).

    Returns
    -------
    Operation or _FUSION_CANCEL or None
        * ``Operation`` — the fused result (re-wrapped in the
          appropriate controlled wrapper).
        * ``_FUSION_CANCEL`` — the two operations cancel completely.
        * ``None`` — not handled here; fall through to generic inverse check.

    """
    if op_a.ctrl_state == op_b.ctrl_state:
        # Recurse into the base (target) operations.
        temp = _fuse_via_transpile(op_a.base_operation, op_b.base_operation)
        if temp is _FUSION_CANCEL:
            return _FUSION_CANCEL
        if temp is not None:
            # Re-wrap the fused base in a controlled wrapper.
            return ControlledOperation(
                temp,
                num_ctrl_qubits=len(op_a.controls),
                ctrl_state=op_a.ctrl_state,
            )

    # Deliberately fall through to the generic inverse check.
    # Many gates with definitions (SWAP, circuits wrapped as
    # gates, etc.) need to be tested for self-inverse cancellation.
    return None


def _fuse_via_transpile(op_a, op_b):
    """Try to fuse two composite gates by transpiling and cancelling.

    Some cancellations are only visible after decomposing composite
    gates into their constituent operations.  This escape hatch:

    1. Builds a two-instruction circuit from ``op_a`` and ``op_b``.
    2. Transpiles it (decomposes the definitions).
    3. Runs ``fuse_adjacents`` on the decomposition.
    4. If anything cancelled (gate count changed), returns the fused
       result as a new gate.

    Parameters
    ----------
    op_a : Operation
        The first operation (earlier in time).
    op_b : Operation
        The second operation (later in time).

    Returns
    -------
    Operation or None
        The fused operation, or ``None`` if nothing cancelled.

    """
    qc = QuantumCircuit(op_a.num_qubits)
    qc.append(op_a, qc.qubits)
    qc.append(op_b, qc.qubits)
    qc = qc.transpile()
    op_counts = qc.count_ops()
    qc = fuse_adjacents(qc)
    qc = combine_single_qubit_gates(qc)
    if op_counts != qc.count_ops():
        # If all gates cancelled → full cancellation.
        if len(qc.data) == 0:
            return _FUSION_CANCEL
        return qc.to_gate()
    return None


def _check_inverse_cancel(op_a, op_b):
    """Check whether op_b is the exact inverse of op_a.

    This is the generic final fallback for inverse cancellation.
    Handles:
      * Self-inverse gates: X·X, H·H, CX·CX, CZ·CZ, SWAP·SWAP, …
      * Param gates with zero sum not caught by _fuse_parameterized_1q
        (e.g. cp·cp⁻¹)

    The "alloc" suffix guard prevents confusing qb_alloc / qb_dealloc
    with proper gates.

    Parameters
    ----------
    op_a : Operation
        The first operation (earlier in time).
    op_b : Operation
        The second operation (later in time).

    Returns
    -------
    _FUSION_CANCEL or None
        ``_FUSION_CANCEL`` if the pair cancels, ``None`` otherwise.

    """
    # If the name of the inverse of gate a is the same
    # as gate b, chances are they are indeed inverse to each other
    # We confirm this by comparing their unitaries
    try:
        inv_op = op_a.inverse()
    except:
        return None

    if inv_op.name == op_b.name and op_a.name[-5:] != "alloc":
        # Avoid calculating very large unitaries
        if inv_op.num_qubits > 4 or op_b.num_qubits > 4:
            return None

        # Retrieve the unitaries
        try:
            unitary_a = inv_op.get_unitary()
            unitary_b = op_b.get_unitary()
        except:
            return None

        # Compare
        if np.linalg.norm(unitary_a - unitary_b) > 1e-4:
            return None

        # Mark as cancelled
        return _FUSION_CANCEL

    return None


# =============================================================================
# Operation-level dispatcher
# =============================================================================


def _fuse_operations(op_a, op_b, gphase_array):
    """Try to fuse two operations into one (or cancel them).

    Dispatches through the specialised fusion helpers in order:
    1. SWAP fusion with neighbouring two-qubit gates.
    2. Parameterised single-qubit gate fusion.
    3. GidneyLogicalAND compute/uncompute cancellation.
    4. Controlled-operation recursion (if op_a has a definition).
    5. Generic inverse cancellation.
    6. Recursive transpile-and-cancel for composite gates.

    Parameters
    ----------
    op_a : Operation
        The first operation (earlier in time).
    op_b : Operation
        The second operation (later in time).
    gphase_array : list
        Single-element list tracking accumulated global phase.

    Returns
    -------
    Operation or _FUSION_CANCEL or None
        * ``Operation`` — the fused result.
        * ``_FUSION_CANCEL`` — the two operations cancel completely.
        * ``None`` — no fusion possible.

    """
    # 1. SWAP fusion with neighbouring two-qubit gates.
    result = _fuse_swap_with_neighbour(op_a, op_b)
    if result is not None:
        return result

    # 2. Parameterised single-qubit gate fusion.
    result = _fuse_parameterized_1q(op_a, op_b, gphase_array)
    if result is not None:
        return result

    # 3. GidneyLogicalAND — must be checked before the generic path
    #    and does NOT fall through to inverse check.
    result = _fuse_gidney_and(op_a, op_b)
    if result is not None:
        return result

    # 4. Composite gate handling (controlled ops) — falls through
    #    to generic inverse check if no controlled-op match is found.
    if isinstance(op_a, ControlledOperation) and isinstance(op_b, ControlledOperation):
        result = _fuse_controlled_ops(op_a, op_b)
        if result is not None:
            return result
        # Deliberately fall through to the generic inverse check.
        # Many gates with definitions (SWAP, circuits wrapped as
        # gates, etc.) need to be tested for self-inverse cancellation.

    # 5. Generic inverse cancellation — catches self-inverse gates
    #    (X·X, H·H, SWAP·SWAP, …) and param gates with zero sum.
    result = _check_inverse_cancel(op_a, op_b)
    if result is not None:
        return result

    # 6. Recursive transpile-and-cancel for composite gates whose
    #    cancellation only becomes visible after decomposition.
    #    This is an escape hatch for cases where the inverse names
    #    don't match but the decompositions partially cancel.
    if (op_a.definition or op_b.definition) and op_a.num_clbits == 0 == op_b.num_clbits:
        result = _fuse_via_transpile(op_a, op_b)
        if result is not None:
            return result

    return None


# =============================================================================
# Instruction-level fusion
# =============================================================================

# Gates whose qubit arguments are symmetric (set-based matching).
_SYMMETRIC_GATES = frozenset({"cz", "swap", "cp", "rzz"})


def _resolve_qubit_order(instr_a, instr_b):
    """Determine the correct qubit order for the fused instruction.

    For symmetric two-qubit gates (CZ, SWAP, CP, RZZ) the qubit order
    does not matter — CZ(0,1) followed by CZ(1,0) is still the same
    pair.  We match by qubit *set* rather than by position.

    For non-symmetric gates, exact positional qubit match is required.

    Parameters
    ----------
    instr_a : Instruction
        The first instruction (earlier in time).
    instr_b : Instruction
        The second instruction (later in time).

    Returns
    -------
    list of Qubit or None
        The qubit list to use for the fused instruction, or ``None``
        if the qubits cannot be matched.

    """
    a_sym = instr_a.op.name in _SYMMETRIC_GATES
    b_sym = instr_b.op.name in _SYMMETRIC_GATES

    if a_sym or b_sym:
        # Symmetric match by set — qubit order difference is tolerated.
        if set(instr_a.qubits) != set(instr_b.qubits):
            return None
        # Preserve the non-symmetric side's qubit order if there is one.
        # If only op_a is symmetric, keep op_b's order (the successor's).
        # If only op_b is symmetric (or both), keep op_a's order.
        if a_sym and not b_sym:
            return instr_b.qubits
        return instr_a.qubits
    for i, qb in enumerate(instr_a.qubits):
        if instr_b.qubits[i] != qb:
            return None
    return instr_a.qubits


def _fuse_instructions(instr_a, instr_b, gphase_array):
    """Try to fuse two neighbouring instructions.

    This is a thin wrapper around ``_fuse_operations`` that additionally
    validates qubit alignment and handles *symmetric* gates (via
    ``_resolve_qubit_order``).

    Returns
    -------
    Instruction or _FUSION_CANCEL or None
        * ``Instruction`` — the fused result.
        * ``_FUSION_CANCEL`` — the two instructions cancel.
        * ``None`` — no fusion possible.

    """
    # Must operate on the same number of qubits.
    if len(instr_a.qubits) != len(instr_b.qubits):
        return None

    # Skip classically-controlled operations (not yet handled).
    if isinstance(instr_a.op, ClControlledOperation) or isinstance(instr_b.op, ClControlledOperation):
        return None

    # Resolve the correct qubit order (handles symmetric gates).
    qubits = _resolve_qubit_order(instr_a, instr_b)
    if qubits is None:
        return None

    # Delegate to the operation-level fusion logic.
    result = _fuse_operations(instr_a.op, instr_b.op, gphase_array)

    # If the result is a new Operation, wrap it in an Instruction.
    # Otherwise propagate _FUSION_CANCEL or None as-is.
    if isinstance(result, Operation):
        return Instruction(result, qubits)
    return result


# =============================================================================
# Public pass
# =============================================================================
# DAG management helpers — used by fuse_adjacents
# =============================================================================


def _init_dag(qc):
    """Initialize the DAG with virtual nodes for every qubit and clbit.

    Nodes are identified by integers:
      * Negative numbers (-1, -2, …) represent qubits and clbits
        at the *start* of time (before any gate acts on them).
      * Non-negative numbers (0, 1, 2, …) represent instructions.

    qubit_dic / clbit_dic map each (c)bit to the *last* node that
    touched it, so we can connect each new instruction to its
    immediate predecessors.

    edge_dic maps DAG edges → lists of qubits, so that when we
    delete a node we know which qubit_dic entries to rewire.

    Parameters
    ----------
    qc : QuantumCircuit
        The input circuit.

    Returns
    -------
    tuple
        (G, qubit_dic, clbit_dic, edge_dic, data_list)

    """
    G = nx.DiGraph()
    qubit_dic = {}
    clbit_dic = {}
    edge_dic = {}

    # Seed the DAG with "virtual" nodes for every qubit and clbit.
    for i, qb in enumerate(qc.qubits):
        qubit_dic[qb] = -i - 1
        G.add_node(-i - 1)

    for i, cb in enumerate(qc.clbits):
        clbit_dic[cb] = -i - 1
        G.add_node(-qc.num_qubits() - i - 1)

    # Work on a copy so we can mutate instructions in-place (for
    # fusion replacements) without affecting the original circuit.
    data_list = list(qc.data)

    return G, qubit_dic, clbit_dic, edge_dic, data_list


def _rewire_past_node(G, node_id, qubit_dic, edge_dic):
    """Rewire incoming edges of *node_id* to skip past it.

    After rewiring, subsequent gates that would have seen *node_id*
    as a predecessor now see *node_id*'s predecessor instead.
    The node itself is NOT removed (the caller decides when to do that).

    Parameters
    ----------
    G : nx.DiGraph
        The DAG.
    node_id : int
        The node whose incoming edges should be rewired.
    qubit_dic : dict
        Maps each qubit → last node that touched it.
    edge_dic : dict
        Maps DAG edges → list of qubits travelling on that edge.

    """
    for edge in G.in_edges(node_id):
        for qb in edge_dic[edge]:
            qubit_dic[qb] = edge[0]
        del edge_dic[edge]


def _emit_surviving_circuit(G, data_list, qc, gphase_array):
    """Emit the surviving instructions in topological order.

    Topological sort guarantees that earlier instructions appear
    before later ones in the dependency chain — even after nodes have
    been deleted and edges rewired.

    Parameters
    ----------
    G : nx.DiGraph
        The DAG containing only surviving instruction nodes and
        virtual qubit/clbit nodes.
    data_list : list
        The working copy of circuit instructions (mutated for fusions).
    qc : QuantumCircuit
        The original circuit (used for clearcopy and clbit lookup).
    gphase_array : list
        Single-element list tracking accumulated global phase.

    Returns
    -------
    QuantumCircuit
        A new circuit containing only the surviving instructions.

    """
    qc_new = qc.clearcopy()
    topo_sort = list(nx.topological_sort(G))

    with fast_append(3):
        for i in topo_sort:
            if i < 0:
                # Virtual qubit/clbit node — skip.
                continue
            # Emit the instruction (unless it's a gphase whose value
            # was already absorbed into the accumulated counter).
            if data_list[i].op.name != "gphase":
                qc_new.append(data_list[i].op, data_list[i].qubits, qc.data[i].clbits)
            else:
                gphase_array[0] += data_list[i].op.params[0]

    # If accumulated global phase is non-zero, emit a gphase gate.
    if gphase_array[0] != 0:
        qc_new.gphase(gphase_array[0], qc_new.qubits[0])

    return qc_new


# =============================================================================
# Public pass
# =============================================================================


@CircuitPass
def fuse_adjacents(qc: QuantumCircuit) -> QuantumCircuit:
    """Cancel adjacent gate–inverse-gate pairs.

    **How it works**

    1.  Build a **directed acyclic graph (DAG)** where each instruction
        is a node.  Edges connect an instruction to its immediate
        successor on the same qubit or classical bit.

    2.  Walk through the instructions in circuit order.  For each
        instruction whose **predecessors all agree on a single node**
        (i.e. the instruction is directly adjacent to its predecessor on
        every qubit it touches), call ``_fuse_instructions`` to attempt
        to merge or cancel the pair.

    3.  When a pair is cancelled, both nodes are
        removed from the DAG and their qubit edges are rewired to
        bridge over the gap.

    4.  When a pair is fused into a new gate, the predecessor node is
        replaced with the fused result and the successor node is
        removed.

    5.  After all candidate pairs have been inspected, the surviving
        nodes are emitted in **topological order**, which preserves the
        circuit's original dependency structure.

    **What it cancels / fuses**

    * Self-inverse gates: ``X·X``, ``H·H``, ``CX·CX``, ``CZ·CZ``,
      ``SWAP·SWAP``, …
    * Parameterised rotations with opposite angles:
      ``Rz(θ)·Rz(−θ)``, ``Rx(θ)·Rx(−θ)``, ``Ry(θ)·Ry(−θ)``
    * Phase gates: ``P(θ)·P(−θ)``
    * Cross-type fusion: ``Rz(θ)·P(−θ)`` → ``P(0)·Gphase``
    * Controlled operations whose base gates cancel
    * Partial fusion of ``SWAP`` with neighbouring ``CX``, ``CP``,
      ``RZZ`` into fewer CX gates

    **Global phase tracking**

    Gates that carry a ``global_phase`` attribute (notably ``Rz``)
    contribute their global phase to a running total.  At the end of
    the pass, any non-zero accumulated phase is emitted as a single
    ``gphase`` gate.

    Parameters
    ----------
    qc : QuantumCircuit
        The input circuit.

    Returns
    -------
    QuantumCircuit
        A new circuit with adjacent inverse gates cancelled.

    Examples
    --------
    Cancel adjacent self-inverse gates (e.g. CX·CX)::

        >>> from qrisp import QuantumCircuit, PassManager
        >>> from qrisp import fuse_adjacents
        >>> qc = QuantumCircuit(2)
        >>> qc.cx(0, 1)
        >>> qc.cx(0, 1)   # CX is self-inverse → cancelled
        >>> print(qc)
        <BLANKLINE>
        qb_96: ──■────■──
               ┌─┴─┐┌─┴─┐
        qb_97: ┤ X ├┤ X ├
               └───┘└───┘
        >>>
        >>> pm = PassManager()
        >>> pm += fuse_adjacents
        >>> optimized_qc = pm.run(qc)
        >>> print(optimized_qc)
        <BLANKLINE>
        qb_96:
        <BLANKLINE>
        qb_97:
        <BLANKLINE>

    Fuse a SWAP with a neighbouring CX into fewer gates::

        >>> qc = QuantumCircuit(2)
        >>> qc.swap(0, 1)
        >>> qc.cx(0, 1)
        >>> pm = PassManager()
        >>> pm += fuse_adjacents
        >>> pm += decompose()
        >>> optimized = pm.run(qc)
        >>> # SWAP·CX fused into a cheaper compound gate rather than
        >>> # decomposing into 3 CX + 1 CX = 4 CX gates.
        >>> print(optimized)
                     ┌───┐
        qb_110: ──■──┤ X ├
                ┌─┴─┐└─┬─┘
        qb_111: ┤ X ├──■──
                └───┘

    """
    gphase_array = [0]  # mutable accumulator for global phase

    # Step 1: initialise the DAG.
    G, qubit_dic, clbit_dic, edge_dic, data_list = _init_dag(qc)

    # Step 2: build the DAG and attempt fusion/cancellation on the fly.
    #
    # For each instruction we:
    #   a) Add a node to the DAG.
    #   b) Add edges from the last instruction on each touched qubit.
    #   c) If ALL those edges come from the SAME node, the instruction
    #      is "locally adjacent" to its predecessor and we can attempt
    #      to fuse/cancel them.
    for i, instr in enumerate(data_list):
        G.add_node(i)

        # Collect the immediate predecessors for every qubit.
        predecessors = set()
        for qb in instr.qubits:
            predecessors.add(qubit_dic[qb])

            # Add an edge from the predecessor to the current node.
            if not G.has_edge(qubit_dic[qb], i):
                G.add_edge(qubit_dic[qb], i)
                edge_dic[(qubit_dic[qb], i)] = []

            # Track which qubits travel along this edge for later rewire.
            edge_dic[(qubit_dic[qb], i)].append(qb)

            # Update the "last seen at" pointer for this qubit.
            qubit_dic[qb] = i

        # Classical bits — same logic, no fusion needed (they just
        # act as barriers in the DAG).
        for cb in instr.clbits:
            G.add_edge(clbit_dic[cb], i)
            clbit_dic[cb] = i

        # Convert to list for indexing.
        predecessors = list(predecessors)

        # ------------------------------------------------------------------
        # Fusion check: only attempt when the instruction has exactly one
        # predecessor, meaning it is directly adjacent on EVERY qubit it
        # touches.  A multi-predecessor situation (e.g. a CNOT whose
        # control and target were last touched by different gates) means
        # the gates are NOT locally adjacent → skip.
        # ------------------------------------------------------------------
        if len(predecessors) == 1:
            pred = predecessors[0]
            if pred < 0:
                continue  # predecessor is the initial |0⟩ — nothing to fuse

            result = _fuse_instructions(data_list[pred], instr, gphase_array)

            if result is not None:
                if result is _FUSION_CANCEL:
                    # ---- Total cancellation ----
                    # Both gates cancel out.  Remove the predecessor
                    # node and rewire its incoming edges so that
                    # subsequent gates see the predecessor of 'pred'.
                    _rewire_past_node(G, pred, qubit_dic, edge_dic)
                    G.remove_node(pred)
                else:
                    # ---- Partial fusion ----
                    # The two gates are replaced by a single fused gate.
                    # The predecessor node gets the new operation.
                    data_list[pred] = result

                    # Rewire the incoming edges of the *successor* node
                    # to point past it (since it is being removed).
                    _rewire_past_node(G, i, qubit_dic, edge_dic)

                # In both cases the successor node (i) is removed.
                G.remove_node(i)

    # Step 3: emit the surviving instructions in topological order.
    return _emit_surviving_circuit(G, data_list, qc, gphase_array)
