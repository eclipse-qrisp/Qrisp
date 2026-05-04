"""
********************************************************************************
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

import networkx as nx

from qrisp.circuit import (
    QuantumCircuit,
    Operation,
    Instruction,
    fast_append,
    RXGate,
    RYGate,
    RZGate,
    RZZGate,
    PGate,
    GPhaseGate,
    ClControlledOperation,
    ControlledOperation,
)


# ---------------------------------------------------------------------------
# Private helpers — instruction fusion for cancel_inverses
# ---------------------------------------------------------------------------


def _fuse_operations(op_a, op_b, gphase_array):
    """Try to fuse two operations into one (or return 1 for cancellation).

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
    Operation or int or None
        * ``Operation`` — the fused result.
        * ``1`` — the two operations cancel completely.
        * ``None`` — no fusion possible.
    """

    # Lazy imports to avoid circular dependencies at module load time.
    from qrisp.alg_primitives import GidneyLogicalAND
    from qrisp.environments import CustomControlOperation

    # ------------------------------------------------------------------
    # Special-case: SWAP fused with a following CP, CZ, CX or RZZ gate.
    #
    # When a SWAP immediately precedes one of these symmetric two-qubit
    # gates, the pair can be replaced by a simpler compound gate that
    # uses fewer CX gates than the naive decomposition of SWAP then gate.
    #
    # The "half_*_qc" circuits below are the gate that remains after
    # absorbing part of the SWAP into the neighbouring operation.
    # ------------------------------------------------------------------
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
            elif op_a.name == "rzz":
                # RZZ(θ, 0,1) · SWAP ⇒ 2 CX, RZ(θ), CX
                half_rzz_qc = QuantumCircuit(2)
                half_rzz_qc.cx(0, 1)
                half_rzz_qc.cx(1, 0)
                half_rzz_qc.rz(op_a.params[0], 1)
                half_rzz_qc.cx(0, 1)
                return half_rzz_qc.to_gate()
            elif op_a.name == "cp":
                # CP(θ, 0,1) · SWAP ⇒ 2 CX, P(-θ/2,1), CX, P(θ/2,0), P(θ/2,1)
                half_cp_qc = QuantumCircuit(2)
                half_cp_qc.cx(0, 1)
                half_cp_qc.cx(1, 0)
                half_cp_qc.p(-op_a.params[0] / 2, 1)
                half_cp_qc.cx(0, 1)
                half_cp_qc.p(op_a.params[0] / 2, 0)
                half_cp_qc.p(op_a.params[0] / 2, 1)
                return half_cp_qc.to_gate()

    # ------------------------------------------------------------------
    # Parameterised single-qubit gates: fuse by adding parameters.
    #
    # Rz(θ₁) · Rz(θ₂) → Rz(θ₁+θ₂), and if θ₁+θ₂ = 0 the pair cancels.
    # P(θ₁)  · P(θ₂)  → P(θ₁+θ₂), same logic.
    # Cross-type fusion (Rz+P) absorbs the Rz global phase into the
    # gphase tracker and emits a single P gate.
    #
    # Note: global phases from Rz are tracked in gphase_array and
    # collapsed at the very end of the main pass.
    # ------------------------------------------------------------------
    if op_a.params:
        param_sum = sum(op_a.params + op_b.params)

        # --- Rz + Rz / Rz + P ---
        if op_a.name == "rz":
            if op_b.name == "rz":
                gphase_array[0] += op_a.global_phase + op_b.global_phase
                if param_sum == 0:
                    return 1  # perfect cancellation
                else:
                    return RZGate(param_sum)
            if op_b.name == "p":
                gphase_array[0] += op_a.global_phase
                if param_sum == 0:
                    return 1
                else:
                    return PGate(param_sum)

        # --- P + Rz / P + P ---
        elif op_a.name == "p":
            if op_b.name == "rz":
                gphase_array[0] += op_b.global_phase
                if param_sum == 0:
                    return 1
                else:
                    return PGate(param_sum)
            if op_b.name == "p":
                if param_sum == 0:
                    return 1
                else:
                    return PGate(param_sum)

        # --- Rx, Ry, gphase (same-name only) ---
        if op_a.name == op_b.name:
            if op_a.name == "rx":
                if param_sum == 0:
                    return 1
                else:
                    return RXGate(param_sum)
            elif op_a.name == "ry":
                if param_sum == 0:
                    return 1
                else:
                    return RYGate(param_sum)
            if op_a.name == "gphase":
                return GPhaseGate(op_a.global_phase + op_b.global_phase)
            # Other parameterised gates with matching names (e.g. cp, rzz)
            # are handled by the generic inverse check further below.

    # ------------------------------------------------------------------
    # Gates that carry a *definition* (decomposable composite gates)
    # or are explicitly known to be CX, CY, CZ.
    #
    # We try to recurse into the definition to fuse inner operations.
    # For ControlledOperations we recurse into the base operation;
    # for GidneyLogicalAND we check the inv flag;
    # for CustomControlOperation we unwrap one level.
    #
    # IMPORTANT: this block does NOT return None when no special case
    # matches — it falls through to the generic inverse check below.
    # This is necessary because SWAP and many other gates have a
    # definition (their CX decomposition) yet are still their own
    # inverse and should cancel via the generic path.
    # ------------------------------------------------------------------
    if op_a.definition or op_a.name in ["cx", "cy", "cz"]:

        # -- Controlled operations with matching control states --
        if isinstance(op_a, ControlledOperation) and isinstance(
            op_b, ControlledOperation
        ):
            if op_a.ctrl_state == op_b.ctrl_state:
                # Recurse into the base (target) operations.
                temp = _fuse_operations(
                    op_a.base_operation, op_b.base_operation, gphase_array
                )
                if temp == 1:
                    return 1
                elif temp is not None:
                    # Re-wrap the fused base in a controlled wrapper.
                    return ControlledOperation(
                        temp,
                        num_ctrl_qubits=len(op_a.controls),
                        ctrl_state=op_a.ctrl_state,
                    )

        # -- Gidney logical-AND pairs (measurement-based Toffoli) --
        elif isinstance(op_a, GidneyLogicalAND) and isinstance(
            op_b, GidneyLogicalAND
        ):
            if op_a.ctrl_state == op_b.ctrl_state:
                # A compute/uncompute pair cancels.
                if op_a.inv != op_b.inv:
                    return 1
            return None

        # -- Custom control operations (unusual control schemes) --
        if isinstance(op_a, CustomControlOperation) and isinstance(
            op_b, CustomControlOperation
        ):
            # Unwrap one level of the custom control and recurse.
            temp = _fuse_operations(
                op_a.definition.data[0].op,
                op_b.definition.data[0].op,
                gphase_array,
            )
            if temp == 1:
                return 1
            elif temp is not None:
                return CustomControlOperation(temp, op_a.targeting_control)

        # Deliberately fall through to the generic inverse check.
        # Many gates with definitions (SWAP, circuits wrapped as
        # gates, etc.) need to be tested for self-inverse cancellation.

    # ------------------------------------------------------------------
    # Generic inverse cancellation — the final fallback.
    #
    # If op_b's name equals the inverse of op_a's name, the pair
    # cancels.  This handles:
    #   * Self-inverse gates: X·X, H·H, CX·CX, CZ·CZ, SWAP·SWAP, …
    #   * Param gates with zero sum not caught above (e.g. cp·cp⁻¹)
    #
    # The "alloc" suffix guard prevents confusing qb_alloc / qb_dealloc
    # with proper gates.
    # ------------------------------------------------------------------
    if op_a.inverse().name == op_b.name and op_a.name[-5:] != "alloc":
        return 1

    # No fusion possible.
    return None


# =============================================================================
# Private helper: instruction-level fusion
# =============================================================================


def _fuse_instructions(instr_a, instr_b, gphase_array):
    """Try to fuse two neighbouring instructions.

    This is a thin wrapper around ``_fuse_operations`` that additionally
    validates qubit alignment and handles *symmetric* gates whose qubit
    order can be freely permuted (CZ, SWAP, CP, RZZ).

    Returns
    -------
    Instruction or int or None
        * ``Instruction`` — the fused result.
        * ``1`` — the two instructions cancel.
        * ``None`` — no fusion possible.
    """
    # Must operate on the same number of qubits.
    if len(instr_a.qubits) != len(instr_b.qubits):
        return None

    # Skip classically-controlled operations (not yet handled).
    if isinstance(instr_a.op, ClControlledOperation) or isinstance(
        instr_b.op, ClControlledOperation
    ):
        return None

    # ------------------------------------------------------------------
    # Determine whether either instruction is a *symmetric* two-qubit
    # gate.  For symmetric gates the qubit order does not matter —
    # CZ(0,1) followed by CZ(1,0) is still the same pair.  We match
    # by qubit *set* rather than by position.
    #
    # Flag which side (if any) is symmetric so we know which qubit
    # order to preserve in the output instruction.
    # ------------------------------------------------------------------
    symmetric_instruction = None
    if instr_a.op.name in ["cz", "swap", "cp", "rzz"]:
        symmetric_instruction = "a"
    if instr_b.op.name in ["cz", "swap", "cp", "rzz"]:
        symmetric_instruction = "b"

    if symmetric_instruction and set(instr_a.qubits) == set(instr_b.qubits):
        # Symmetric match by set — qubit order difference is tolerated.
        pass
    else:
        # Non-symmetric gate: require exact positional qubit match.
        for i, qb in enumerate(instr_a.qubits):
            if instr_b.qubits[i] != qb:
                return None

    # Delegate to the operation-level fusion logic.
    temp = _fuse_operations(instr_a.op, instr_b.op, gphase_array)

    # If the result is a new Operation, wrap it in an Instruction with
    # the correct qubit ordering:
    #   * No symmetric gate involved → use op_a's qubit order.
    #   * Only op_a is symmetric   → use op_b's qubit order (op_a was
    #     the predecessor; after fusion the order from op_b wins).
    #   * Only op_b is symmetric   → use op_a's qubit order.
    if isinstance(temp, Operation):
        if symmetric_instruction is None:
            return Instruction(temp, instr_a.qubits)
        elif symmetric_instruction == "a":
            return Instruction(temp, instr_b.qubits)
        elif symmetric_instruction == "b":
            return Instruction(temp, instr_a.qubits)
    else:
        # Propagate 1 (cancellation) or None (no fusion) as-is.
        return temp


# =============================================================================
# Public pass
# =============================================================================


def cancel_inverses(qc: QuantumCircuit) -> QuantumCircuit:
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

    3.  When a pair is cancelled (return value ``1``), both nodes are
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
    * Cross-type fusion: ``Rz(θ)·P(−θ)`` → ``P(0)`` → cancelled
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

    Example
    -------
    >>> from qrisp import PassManager, cancel_inverses
    >>> from qrisp import QuantumCircuit
    >>>
    >>> qc = QuantumCircuit(2)
    >>> qc.cx(0, 1)
    >>> qc.cx(0, 1)   # CX is self-inverse → cancelled
    >>>
    >>> pm = PassManager()
    >>> pm.add_pass(cancel_inverses)
    >>> optimized_qc = pm.run(qc)   # empty circuit
    """
    # ------------------------------------------------------------------
    # Step 1: initialise the DAG.
    #
    # Nodes are identified by integers:
    #   * Negative numbers (-1, -2, …) represent qubits and clbits
    #     at the *start* of time (before any gate acts on them).
    #   * Non-negative numbers (0, 1, 2, …) represent instructions.
    #
    # qubit_dic / clbit_dic map each (c)bit to the *last* node that
    # touched it, so we can connect each new instruction to its
    # immediate predecessors.
    #
    # edge_dic maps DAG edges → lists of qubits, so that when we
    # delete a node we know which qubit_dic entries to rewire.
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    qubit_dic = {}
    clbit_dic = {}
    edge_dic = {}

    gphase_array = [0]  # mutable accumulator for global phase

    # Seed the DAG with "virtual" nodes for every qubit and clbit.
    for i in range(qc.num_qubits()):
        qubit_dic[qc.qubits[i]] = -i - 1
        G.add_node(-i - 1)

    for i in range(len(qc.clbits)):
        clbit_dic[qc.clbits[i]] = -i - 1
        G.add_node(-qc.num_qubits() - i - 1)

    # Work on a copy so we can mutate instructions in-place (for
    # fusion replacements) without affecting the original circuit.
    data_list = list(qc.data)

    # ------------------------------------------------------------------
    # Step 2: build the DAG and attempt fusion/cancellation on the fly.
    #
    # For each instruction we:
    #   a) Add a node to the DAG.
    #   b) Add edges from the last instruction on each touched qubit.
    #   c) If ALL those edges come from the SAME node, the instruction
    #      is "locally adjacent" to its predecessor and we can attempt
    #      to fuse/cancel them.
    # ------------------------------------------------------------------
    for i in range(len(data_list)):

        G.add_node(i)
        instr = data_list[i]

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

            fused_gate = _fuse_instructions(
                data_list[pred], data_list[i], gphase_array
            )

            if fused_gate is not None:
                if fused_gate == 1:
                    # ---- Total cancellation ----
                    # Both gates cancel out.  Remove the predecessor
                    # node and rewire its incoming edges so that
                    # subsequent gates see the predecessor of 'pred'.
                    for edge in G.in_edges(pred):
                        for qb in edge_dic[edge]:
                            qubit_dic[qb] = edge[0]
                        del edge_dic[edge]
                    G.remove_node(pred)
                else:
                    # ---- Partial fusion ----
                    # The two gates are replaced by a single fused gate.
                    # The predecessor node gets the new operation.
                    data_list[pred] = fused_gate

                    # Rewire the incoming edges of the *successor* node
                    # to point past it (since it is being removed).
                    for edge in G.in_edges(i):
                        for qb in edge_dic[edge]:
                            qubit_dic[qb] = edge[0]
                        del edge_dic[edge]

                # In both cases the successor node (i) is removed.
                G.remove_node(i)

    # ------------------------------------------------------------------
    # Step 3: emit the surviving instructions in topological order.
    #
    # Topological sort guarantees that earlier instructions appear
    # before later ones in the dependency chain — even after we've
    # deleted nodes and rewired edges.
    # ------------------------------------------------------------------
    qc_new = qc.clearcopy()

    topo_sort = list(nx.topological_sort(G))
    with fast_append(3):
        for i in topo_sort:
            if i < 0:
                # Virtual qubit/clbit node — skip.
                continue
            else:
                # If this is the very last instruction and accumulated
                # global phase is non-zero, emit a gphase gate first.
                if i == topo_sort[-1]:
                    if data_list[i].op.name == "gphase":
                        gphase_array[0] += data_list[i].op.params[0]
                    if gphase_array[0] != 0:
                        qc_new.gphase(gphase_array[0], data_list[i].qubits[0])

                # Emit the instruction (unless it's a gphase whose value
                # was already absorbed into the accumulated counter).
                if data_list[i].op.name != "gphase":
                    qc_new.append(
                        data_list[i].op, data_list[i].qubits, qc.data[i].clbits
                    )
                else:
                    gphase_array[0] += data_list[i].op.params[0]

    return qc_new

