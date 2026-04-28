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

"""
PASS 2 – SCF → CC dialect lowering.

Replaces structured control-flow ops (``scf.if``, ``scf.for``, ``scf.while``)
with their CC-dialect equivalents (``cc.if``, ``cc.loop``).

Strategy
--------
Loop-carried SSA values are promoted to stack memory via a
:class:`_MemorySlot` abstraction (``cc.alloca`` / ``cc.store`` /
``cc.load``), so the resulting ``cc.loop`` has **no** SSA results and no
block arguments in its regions.  This matches the style emitted by the
CUDA-Q compiler.

Memory slot creation automatically **unwraps rank-0 tensors** to scalars
(``tensor<T>`` → ``T``) because CUDA-Q's ``cc.alloca`` only supports scalar
types.  The wrappers (``tensor.extract`` / ``tensor.from_elements``) are
inserted transparently and cleaned up by Pass 3 (tensor unwrap).

For ``scf.while``, an **invariant-detection** step classifies each
loop-carried value as *invariant* (captured from the enclosing scope,
no alloca) or *variant* (promoted to a memory slot).  This avoids illegal
``cc.alloca !quake.veq<?>`` ops that would crash the CUDA-Q runtime.

Architecture
------------
Both ``scf.for`` and ``scf.while`` lowerings share the same memory-slot
infrastructure:

* ``_MemorySlot`` – data class encapsulating an alloca + tensor-wrapping
  metadata
* ``_create_slot`` – alloca + init store (with tensor unwrap)
* ``_load_from_slot`` – ``cc.load`` [+ ``tensor.from_elements``]
* ``_store_to_slot`` – [``tensor.extract`` +] ``cc.store``
* ``_rewrite_block_to_region`` – replace block args with loads / captures,
  move ops to a new region

Important
---------
Ops that are **permanently removed** (``scf.condition``, ``scf.yield``,
``scf.for``, ``scf.while``) must use ``Rewriter.erase_op(op,
safe_erase=False)`` — **not** ``op.detach()``.  The ``detach()`` method
removes the op from its parent block but does **not** release its operand
references, leaving phantom uses that prevent downstream dead-code
elimination from cleaning up dead ``tensor.from_elements`` ops.

``op.detach()`` is only correct when **moving** an op between blocks
(followed immediately by ``new_block.add_op(op)``).
"""

import warnings

from xdsl.dialects import scf, arith, tensor
from xdsl.dialects.builtin import i1, TensorType
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.quake_lowering.cc_dialect_old import (
    CcConditionOp,
    CcContinueOp,
    CcIfOp,
    CcLoopOp,
    CcAllocaOp,
    CcStoreOp,
    CcLoadOp,
)
from qrisp.jasp.mlir.quake_lowering.quake_dialect import (
    QuakeVeqType,
    QuakeRefType,
)


# ===================================================================
# Public entry point
# ===================================================================

def lower_scf_to_cc(module) -> None:
    """In-place PASS 2: lower SCF structured control flow to the CC dialect."""
    for op in list(module.body.blocks[0].ops):
        if op.name == "func.func":
            _process_region(op.body)


# ===================================================================
# xDSL compatibility shims
# ===================================================================

def _replace_all_uses_with(val: SSAValue, new_val: SSAValue) -> None:
    if hasattr(val, "replace_all_uses_with"):
        val.replace_all_uses_with(new_val)
    else:
        val.replace_by(new_val)


def _get_arith_addi():
    cls = getattr(arith, "Addi", None) or getattr(arith, "AddiOp", None)
    if cls is None:
        raise ImportError(
            "Cannot find arith.Addi or arith.AddiOp in this xDSL version"
        )
    return cls


def _get_results(op) -> list:
    return list(getattr(op, "results", getattr(op, "res", [])))


# ===================================================================
# Type classification
# ===================================================================

def _is_rank0_tensor(ty) -> bool:
    """Return True if *ty* is a rank-0 tensor (e.g. ``tensor<i64>``)."""
    return isinstance(ty, TensorType) and not ty.get_shape()


def _is_alloca_safe(ty) -> bool:
    """Return True if *ty* can be used with ``cc.alloca``.

    Only scalar types (``i1``, ``i64``, ``f64``, …) are supported.
    Quantum handles and tensor types are rejected.
    """
    if isinstance(ty, TensorType):
        return False
    if isinstance(ty, (QuakeVeqType, QuakeRefType)):
        return False
    return True


# ===================================================================
# Memory slot abstraction
# ===================================================================

class _MemorySlot:
    """A loop-carried value promoted to ``cc.alloca`` stack memory.

    Encapsulates the alloca op and metadata about whether the original
    SSA type was a rank-0 tensor that was unwrapped to a scalar for
    the alloca.  All load/store helpers use this metadata to
    transparently insert ``tensor.from_elements`` / ``tensor.extract``
    wrappers as needed.
    """

    __slots__ = ("alloca", "is_tensor_wrapped", "original_type")

    def __init__(self, alloca: CcAllocaOp, is_tensor_wrapped: bool,
                 original_type):
        self.alloca = alloca
        self.is_tensor_wrapped = is_tensor_wrapped
        self.original_type = original_type


def _create_slot(outer_block: Block, init_val: SSAValue,
                 insert_before) -> _MemorySlot:
    """Create a memory slot: ``cc.alloca`` + initial ``cc.store``.

    If *init_val* is a rank-0 ``tensor<T>``, a ``tensor.extract`` is
    emitted to unwrap it to scalar ``T`` before the alloca.

    Raises :class:`TypeError` if the (unwrapped) type is not
    alloca-safe (quantum handles, higher-rank tensors).
    """
    original_type = init_val.type
    store_val = init_val
    alloca_type = original_type
    is_wrapped = False

    if _is_rank0_tensor(alloca_type):
        scalar_type = alloca_type.element_type
        extract = tensor.ExtractOp(init_val, [], scalar_type)
        outer_block.insert_ops_before([extract], insert_before)
        store_val = extract.result
        alloca_type = scalar_type
        is_wrapped = True

    if not _is_alloca_safe(alloca_type):
        raise TypeError(
            f"Cannot promote value of type {alloca_type} to cc.alloca.  "
            f"Quantum types and non-scalar tensor types must be "
            f"loop-invariant."
        )

    alloca = CcAllocaOp(alloca_type)
    store = CcStoreOp(store_val, alloca.result)
    outer_block.insert_ops_before([alloca, store], insert_before)
    return _MemorySlot(alloca, is_wrapped, original_type)


def _load_from_slot(slot: _MemorySlot, block: Block,
                    insert_before=None, *, raw: bool = False) -> SSAValue:
    """Emit ``cc.load`` [+ ``tensor.from_elements``] and return the value.

    Parameters
    ----------
    slot : _MemorySlot
        The slot to load from.
    block : Block
        Target block for the new ops.
    insert_before : Operation or None
        If given, ops are inserted before this op.  Otherwise they are
        appended to the end of *block*.
    raw : bool
        If True, return the raw scalar even if the slot was
        tensor-wrapped.  Useful for arithmetic (``arith.cmpi``,
        ``arith.addi``) that requires scalar operands.
    """
    load = CcLoadOp(slot.alloca.result)
    ops = [load]
    result = load.result

    if not raw and slot.is_tensor_wrapped:
        wrap = tensor.FromElementsOp(
            operands=[[result]],
            result_types=[slot.original_type],
        )
        ops.append(wrap)
        result = wrap.result

    if insert_before is not None:
        block.insert_ops_before(ops, insert_before)
    else:
        for op in ops:
            block.add_op(op)
    return result


def _store_to_slot(slot: _MemorySlot, value: SSAValue, block: Block,
                   insert_before) -> None:
    """Emit [``tensor.extract`` +] ``cc.store``.

    If the slot was tensor-wrapped and *value* is still a rank-0 tensor,
    a ``tensor.extract`` is inserted first to unwrap it.  If *value* is
    already scalar, the store is emitted directly.
    """
    store_val = value
    ops = []

    if slot.is_tensor_wrapped and _is_rank0_tensor(value.type):
        extract = tensor.ExtractOp(value, [], value.type.element_type)
        ops.append(extract)
        store_val = extract.result

    ops.append(CcStoreOp(store_val, slot.alloca.result))
    block.insert_ops_before(ops, insert_before)


# ===================================================================
# Block rewriting (shared by scf.for and scf.while)
# ===================================================================

def _rewrite_block_to_region(old_block: Block,
                             slots_and_captures: list) -> tuple:
    """Create a new ``Region``, replace block args, and move all ops.

    Parameters
    ----------
    old_block : Block
        The original block whose ops will be moved.
    slots_and_captures : list
        One entry per block arg.  Each entry is either:

        * ``_MemorySlot`` → emit ``cc.load`` [+ from_elements] as
          replacement (variant value).
        * ``SSAValue`` → use directly as replacement (invariant capture
          from the enclosing scope).

    Returns
    -------
    (new_region, new_block, replacement_values)
        ``replacement_values[i]`` is the SSA value that replaced the
        *i*-th block arg (either the load result or the captured value).

    Notes
    -----
    ``_replace_all_uses_with`` is called while the consuming ops still
    reside in *old_block*.  This is safe because it only rewires
    use-def pointers; the ops are moved immediately afterwards.

    Ops are moved via ``op.detach()`` + ``new_block.add_op(op)``
    (correct for **moving**; see module docstring for the ``detach``
    vs ``erase_op`` distinction).
    """
    new_region = Region([Block()])
    new_block = new_region.blocks[0]

    replacement_values = []
    for barg, item in zip(list(old_block.args), slots_and_captures):
        if isinstance(item, _MemorySlot):
            val = _load_from_slot(item, new_block)  # appends to end
        else:
            val = item  # invariant capture
        _replace_all_uses_with(barg, val)
        replacement_values.append(val)

    # Move all ops (including terminators) to the new block.
    for op in list(old_block.ops):
        op.detach()
        new_block.add_op(op)

    return new_region, new_block, replacement_values


# ===================================================================
# IR search helpers
# ===================================================================

def _find_condition_op(block: Block):
    """Return the ``scf.condition`` op in *block*, or None."""
    for op in block.ops:
        if op.name == "scf.condition":
            return op
    return None


def _find_trailing_yield(block: Block):
    """Return the trailing ``scf.yield`` in *block*, or None."""
    ops = list(block.ops)
    if ops and ops[-1].name == "scf.yield":
        return ops[-1]
    return None


# ===================================================================
# Traversal
# ===================================================================

def _process_region(region: Region) -> None:
    for block in region.blocks:
        _process_block(block)


def _process_block(block: Block) -> None:
    for op in list(block.ops):
        _process_op(op, block)


def _process_op(op, block: Block) -> None:
    if op.name == "scf.if":
        _convert_scf_if(op, block)
    elif op.name == "scf.while":
        _convert_scf_while(op, block)
    elif op.name == "scf.for":
        _convert_scf_for(op, block)
    else:
        for region in op.regions:
            _process_region(region)


# ===================================================================
# scf.if → cc.if
# ===================================================================

def _convert_scf_if(if_op, outer_block: Block) -> None:
    """Convert ``scf.if`` → ``cc.if``, including the case with results."""
    for region in if_op.regions:
        _process_region(region)

    result_types = list(if_op.result_types)
    cond = if_op.cond

    # --- No results: simple conversion (unchanged) ------------------------
    if not result_types:
        true_region = if_op.detach_region(if_op.regions[0])
        false_region = if_op.detach_region(if_op.regions[0])

        _replace_yield_with_continue(true_region)
        _replace_yield_with_continue(false_region)

        cc_if = CcIfOp(cond, true_region, false_region)
        Rewriter.replace_op(if_op, cc_if, [], safe_erase=False)
        return

    # --- Has results: promote to memory slots -----------------------------

    # 1. Create one uninitialized alloca per result (reuse _MemorySlot directly)
    slots = []
    for rt in result_types:
        original_type = rt
        alloca_type = rt
        is_wrapped = False

        if _is_rank0_tensor(alloca_type):
            alloca_type = alloca_type.element_type
            is_wrapped = True

        alloca = CcAllocaOp(alloca_type)
        outer_block.insert_ops_before([alloca], if_op)
        slots.append(_MemorySlot(alloca, is_wrapped, original_type))

    # 2. Detach regions
    true_region = if_op.detach_region(if_op.regions[0])
    false_region = if_op.detach_region(if_op.regions[0])

    # 3. In each branch: replace scf.yield with stores + cc.continue
    for region in (true_region, false_region):
        for block in region.blocks:
            for op in list(block.ops):
                if op.name == "scf.yield":
                    for operand, slot in zip(list(op.operands), slots):
                        _store_to_slot(slot, operand, block, op)
                    block.insert_ops_before([CcContinueOp()], op)
                    Rewriter.erase_op(op, safe_erase=False)

    # 4. Build cc.if (result-free)
    cc_if = CcIfOp(cond, true_region, false_region)
    outer_block.insert_ops_before([cc_if], if_op)

    # 5. After cc.if, load from each slot to replace original results
    for slot, res in zip(slots, _get_results(if_op)):
        final_val = _load_from_slot(slot, outer_block, if_op)
        _replace_all_uses_with(res, final_val)

    # 6. Erase original scf.if
    Rewriter.erase_op(if_op, safe_erase=False)


def _replace_yield_with_continue(region: Region) -> None:
    """Replace no-operand ``scf.yield`` → ``cc.continue`` (for cc.if)."""
    for block in region.blocks:
        for op in list(block.ops):
            if op.name == "scf.yield" and not list(op.operands):
                block.insert_ops_before([CcContinueOp()], op)
                Rewriter.erase_op(op, safe_erase=False)
            elif op.name == "scf.yield":
                warnings.warn(
                    "scf.yield with operands inside cc.if region; "
                    "left in place.",
                    stacklevel=4,
                )


# ===================================================================
# scf.for → cc.loop
# ===================================================================

def _convert_scf_for(for_op, outer_block: Block) -> None:
    """Convert ``scf.for`` → ``cc.loop`` via memory-slot promotion.

    The induction variable (IV) and every iter-arg are promoted to
    ``_MemorySlot``s.  A synthetic condition region (``iv < ub``) is
    built; the body region is moved from the original ``scf.for``.

    Steps
    -----
    1. Create memory slots for IV + iter-args.
    2. Build synthetic condition region: ``while (iv < ub)``.
    3. Rewrite body: replace block args with loads, move ops.
    4. Replace ``scf.yield``: store iter-args + IV step + ``cc.continue``.
    5. Build ``cc.loop``.
    6. Final loads → replace original results.
    7. Erase original ``scf.for``.
    """
    # Recurse first.
    for region in for_op.regions:
        _process_region(region)

    lb = for_op.operands[0]
    ub = for_op.operands[1]
    step = for_op.operands[2]
    iter_args = list(for_op.operands[3:])
    body_block = for_op.regions[0].blocks[0]

    # ---- 1. Create memory slots ------------------------------------------
    iv_slot = _create_slot(outer_block, lb, for_op)
    iter_slots = [_create_slot(outer_block, v, for_op) for v in iter_args]

    # ---- 2. Synthetic condition: while (iv < ub) -------------------------
    cond_region = Region([Block()])
    cond_block = cond_region.blocks[0]

    iv_cond = _load_from_slot(iv_slot, cond_block, raw=True)
    cmp = arith.CmpiOp(iv_cond, ub, "slt")
    cond_block.add_op(cmp)
    cond_block.add_op(CcConditionOp(cmp.result))

    # ---- 3. Body region: replace block args → loads, move ops ------------
    all_slots = [iv_slot] + iter_slots
    body_region, new_body, _ = _rewrite_block_to_region(body_block, all_slots)

    # ---- 4. Replace yield: stores + IV step + continue -------------------
    yield_op = _find_trailing_yield(new_body)
    if yield_op is not None:
        # Store yielded iter-arg values back to their slots.
        for operand, slot in zip(yield_op.operands, iter_slots):
            _store_to_slot(slot, operand, new_body, yield_op)

        # IV step: load scalar IV, add step, store back.
        addi_cls = _get_arith_addi()
        iv_scalar = _load_from_slot(iv_slot, new_body, yield_op, raw=True)
        next_iv = addi_cls(iv_scalar, step)
        new_body.insert_ops_before([next_iv], yield_op)
        _store_to_slot(iv_slot, next_iv.result, new_body, yield_op)

        new_body.insert_ops_before([CcContinueOp()], yield_op)
        Rewriter.erase_op(yield_op, safe_erase=False)

    # ---- 5. Build cc.loop ------------------------------------------------
    cc_loop = CcLoopOp(
        arguments=[],
        result_types=[],
        while_region=cond_region,
        body_region=body_region,
    )
    outer_block.insert_ops_before([cc_loop], for_op)

    # ---- 6. Final loads → replace results --------------------------------
    for slot, res in zip(iter_slots, _get_results(for_op)):
        final_val = _load_from_slot(slot, outer_block, for_op)
        _replace_all_uses_with(res, final_val)

    # ---- 7. Erase original (releases operand references) -----------------
    Rewriter.erase_op(for_op, safe_erase=False)


# ===================================================================
# scf.while → cc.loop (with invariant detection)
# ===================================================================

def _analyse_while_variance(before_block: Block, after_block: Block,
                            n: int) -> tuple:
    """Classify each loop-carried value as invariant or variant.

    A value at index *i* is **invariant** if:

    * The ``scf.condition`` forwards the *before* block arg unchanged, AND
    * The ``scf.yield`` yields the *after* block arg unchanged.

    Returns ``(is_variant, cond_stores_needed, body_stores_needed)`` —
    three lists of booleans, each of length *n*.
    """
    cond_op = _find_condition_op(before_block)
    yield_op = _find_trailing_yield(after_block)

    before_args = list(before_block.args)
    after_args = list(after_block.args)
    cond_forwarded = list(cond_op.operands)[1:] if cond_op else []
    yield_operands = list(yield_op.operands) if yield_op else []

    is_variant = []
    cond_stores_needed = []
    body_stores_needed = []

    for i in range(n):
        c_mod = (i < len(cond_forwarded) and i < len(before_args)
                 and cond_forwarded[i] is not before_args[i])
        b_mod = (i < len(yield_operands) and i < len(after_args)
                 and yield_operands[i] is not after_args[i])
        is_variant.append(c_mod or b_mod)
        cond_stores_needed.append(c_mod)
        body_stores_needed.append(b_mod)

    return is_variant, cond_stores_needed, body_stores_needed


def _convert_scf_while(while_op, outer_block: Block) -> None:
    """Convert ``scf.while`` → ``cc.loop`` with invariant detection.

    Steps
    -----
    1. **Analyse** — classify each loop-carried value (before any IR
       modifications).
    2. **Create slots** — ``cc.alloca`` + init store for variant values
       only.
    3. **Rewrite before (condition) region** — block args → loads /
       captures; ``scf.condition`` → stores + ``cc.condition``.
    4. **Rewrite after (body) region** — block args → loads / captures;
       ``scf.yield`` → stores + ``cc.continue``.
    5. **Build** ``cc.loop``.
    6. **Final loads** → replace original SSA results.
    7. **Erase** original ``scf.while``.
    """
    # Recurse first.
    for region in while_op.regions:
        _process_region(region)

    init_args = list(while_op.operands)
    before_block = while_op.regions[0].blocks[0]
    after_block = while_op.regions[1].blocks[0]
    n = len(init_args)

    # ---- 1. Analyse (before any IR modifications!) -----------------------
    cond_op = _find_condition_op(before_block)
    if cond_op is None:
        warnings.warn(
            "scf.while has no scf.condition in before region; "
            "left in SCF form.",
            stacklevel=4,
        )
        return

    is_variant, cond_stores_needed, body_stores_needed = \
        _analyse_while_variance(before_block, after_block, n)

    # ---- 2. Create memory slots for variant values -----------------------
    slots = [None] * n  # None ⇒ invariant
    for i in range(n):
        if is_variant[i]:
            slots[i] = _create_slot(outer_block, init_args[i], while_op)

    # Build the mixed list for _rewrite_block_to_region:
    #   _MemorySlot for variant, SSAValue for invariant.
    slots_and_captures = [
        slots[i] if slots[i] is not None else init_args[i]
        for i in range(n)
    ]

    # ---- 3. Rewrite before (condition) region ----------------------------
    before_region, new_before, _ = _rewrite_block_to_region(
        before_block, slots_and_captures
    )

    # Replace scf.condition: store variant forwarded values + cc.condition.
    cond_op = _find_condition_op(new_before)
    cond_val = cond_op.operands[0]
    forwarded = list(cond_op.operands)[1:]
    for i, fwd_val in enumerate(forwarded):
        if i < n and slots[i] is not None and cond_stores_needed[i]:
            _store_to_slot(slots[i], fwd_val, new_before, cond_op)
    new_before.insert_ops_before([CcConditionOp(cond_val)], cond_op)
    Rewriter.erase_op(cond_op, safe_erase=False)

    # ---- 4. Rewrite after (body) region ----------------------------------
    after_region, new_after, _ = _rewrite_block_to_region(
        after_block, slots_and_captures
    )

    # Replace scf.yield: store variant yielded values + cc.continue.
    yield_op = _find_trailing_yield(new_after)
    if yield_op is not None:
        for i, operand in enumerate(yield_op.operands):
            if i < n and slots[i] is not None and body_stores_needed[i]:
                _store_to_slot(slots[i], operand, new_after, yield_op)
        new_after.insert_ops_before([CcContinueOp()], yield_op)
        Rewriter.erase_op(yield_op, safe_erase=False)

    # ---- 5. Build cc.loop ------------------------------------------------
    cc_loop = CcLoopOp(
        arguments=[],
        result_types=[],
        while_region=before_region,
        body_region=after_region,
    )
    outer_block.insert_ops_before([cc_loop], while_op)

    # ---- 6. Final loads → replace results --------------------------------
    loop_results = _get_results(while_op)
    for i, res in enumerate(loop_results):
        if slots[i] is not None:
            final_val = _load_from_slot(slots[i], outer_block, while_op)
            _replace_all_uses_with(res, final_val)
        else:
            # Invariant: result equals the initial value.
            _replace_all_uses_with(res, init_args[i])

    # ---- 7. Erase original (releases operand references) -----------------
    Rewriter.erase_op(while_op, safe_erase=False)
