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
PASS 4 – CC Array Pointer → CC StdVec Lowering (Entrypoint + Propagation)
===========================================================================

Rewrites ``!cc.ptr<!cc.array<T x N>>`` function parameters for CUDA-Q
runtime compatibility.

**Entrypoint function** (marked with ``cudaq.entrypoint = "true"``):
  - Parameter type → ``!cc.stdvec<T>``
  - Inserts ``cc.stdvec_data`` to extract ``!cc.ptr<!cc.array<T x ?>>``
  - Selective rewiring ensures array accesses use the extracted pointer, 
    while downstream helper calls pass the stdvec.

**Internal helper functions**:
  - Inter-procedural analysis (IPA) traces parameters receiving dynamic-ptr 
    values from entrypoints and upgrades them to ``!cc.stdvec<T>`` to match.
"""

from typing import Any
from collections import defaultdict

from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import ModuleOp, FunctionType, Attribute
from xdsl.ir import SSAValue

from xdsl.rewriter import Rewriter, InsertPoint
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)

from qrisp.jasp.mlir.quake_lowering.cc_dialect import (
    CcArrayType,
    CcPtrType,
    CcStdVecDataOp,
    CcStdVecType,
    CcCastOp,
)

_MLIR_DYNAMIC = -9223372036854775808

# ===================================================================
# Public entry point
# ===================================================================

# NOTE: Can likely be further simplified: MLIR from JASP assumes statically sized arrays. 
# The CUDA-Q entrypoint assumes a dynamically sized stdvec. 
# We can just replace the array in the entrypoint with stdvec and then immediately cast to a statically sized cc array.
# This would eliminate the need for call graph analysis and selective rewiring, as all internal functions would continue to use the original static array type.

def lower_array_to_stdvec(module: ModuleOp) -> None:
    """In-place pass: Propagate stdvec types through call graphs and rewrite."""
    
    # Step 1: Inter-procedural analysis to find all required argument rewrites
    rewrites = _analyze_call_graph(module)
    if not rewrites:
        return

    # Step 2: Declarative rewriting of function signatures and selective rewiring
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([
            StdVecFuncSignaturePattern(rewrites)
        ]),
        walk_regions_first=False
    )
    walker.rewrite_module(module)


# ===================================================================
# Phase 1: Call Graph Analysis
# ===================================================================


def _analyze_call_graph(module: ModuleOp) -> dict[str, set[int]]:
    """Return a mapping of {func_name: set(arg_indices)} needing stdvec conversion."""
    rewrites: dict[str, set[int]] = defaultdict(set)
    func_map: dict[str, func_dialect.FuncOp] = {}

    # 1. Map entrypoints and cache functions
    for op in module.body.blocks[0].ops:
        if not isinstance(op, func_dialect.FuncOp):
            continue
            
        func_name = op.sym_name.data
        func_map[func_name] = op
        
        if _is_entrypoint(op):
            for i, arg in enumerate(op.function_type.inputs):
                if _is_array_ptr(arg):
                    rewrites[func_name].add(i)

    # 2. Propagate down the call graph via BFS worklist
    worklist = [(name, idx) for name, indices in rewrites.items() for idx in indices]
    
    while worklist:
        curr_func_name, curr_arg_idx = worklist.pop(0)
        curr_func = func_map.get(curr_func_name)
        if not curr_func or not curr_func.body.blocks:
            continue

        curr_arg = curr_func.body.blocks[0].args[curr_arg_idx]

        for use in curr_arg.uses:
            if isinstance(use.operation, func_dialect.CallOp):
                callee_name = use.operation.callee.string_value()
                operand_idx = use.index
                
                # If this is the first time we've seen this argument in the callee, queue it
                if callee_name not in rewrites or operand_idx not in rewrites[callee_name]:
                    rewrites[callee_name].add(operand_idx)
                    worklist.append((callee_name, operand_idx))

    return dict(rewrites)


# ===================================================================
# Phase 2: Rewrite Patterns
# ===================================================================


class StdVecFuncSignaturePattern(RewritePattern):
    """Rewrites target function signatures and inserts selective array pointer extraction."""
    
    def __init__(self, rewrites: dict[str, set[int]]):
        self.rewrites = rewrites

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.FuncOp, rewriter: PatternRewriter) -> None:
        func_name = op.sym_name.data
        if func_name not in self.rewrites:
            return

        arg_indices = self.rewrites[func_name]
        block = op.body.blocks[0]

        # 1. Break the greedy loop if already converted
        if all(isinstance(block.args[idx].type, CcStdVecType) for idx in arg_indices):
            return

        new_inputs = list(op.function_type.inputs)

        for idx in arg_indices:
            arg = block.args[idx]
            old_type = arg.type
            
            # Ensure we are dealing with the expected pointer type
            assert isinstance(old_type, CcPtrType) and isinstance(old_type.element_type, CcArrayType)
            elem_type = old_type.element_type.element_type

            stdvec_type = CcStdVecType(elem_type)
            # The dynamic pointer extracted from stdvec
            dyn_ptr_type = CcPtrType(CcArrayType(elem_type, _MLIR_DYNAMIC))

            # 2. Extract array pointer from stdvec
            data_op = CcStdVecDataOp(arg, dyn_ptr_type)
            rewriter.insert_op(data_op, InsertPoint.at_start(block))

            # 3. FIX: Create a cast bridge if the original IR expected a static size
            # This converts !cc.ptr<!cc.array<T x ?>> to !cc.ptr<!cc.array<T x N>>
            if old_type != dyn_ptr_type:
                cast_op = CcCastOp(data_op.result, old_type)
                rewriter.insert_op(cast_op, InsertPoint.after(data_op))
                replacement_val = cast_op.result
            else:
                replacement_val = data_op.result

            # 4. Selectively rewire existing uses
            for use in list(arg.uses):
                if use.operation == data_op:
                    continue

                # Pass-through to helpers expecting stdvec are NOT rewired
                is_pass_through_call = False
                if isinstance(use.operation, func_dialect.CallOp):
                    callee = use.operation.callee.string_value()
                    if callee in self.rewrites and use.index in self.rewrites[callee]:
                        is_pass_through_call = True

                if not is_pass_through_call:
                    _replace_specific_use(use, replacement_val)

            # 5. Update the block argument type safely
            Rewriter.replace_value_with_new_type(arg, stdvec_type)
            new_inputs[idx] = stdvec_type

        # 6. Commit signature change
        op.function_type = FunctionType.from_lists(new_inputs, list(op.function_type.outputs))


# ===================================================================
# Helpers
# ===================================================================


def _replace_specific_use(use: Any, new_val: SSAValue) -> None:
    """Safely replace a specific operand use in an xDSL operation."""
    if hasattr(use.operation, "replace_operand"):
        use.operation.replace_operand(use.index, new_val)
    else:
        # Fallback for immutable operation lists
        new_ops = list(use.operation.operands)
        new_ops[use.index] = new_val
        use.operation.operands = tuple(new_ops)


def _is_entrypoint(func_op: func_dialect.FuncOp) -> bool:
    """Check if the function has the CUDA-Q entrypoint attribute."""
    if "cudaq.entrypoint" in func_op.attributes:
        val = func_op.attributes["cudaq.entrypoint"]
        return getattr(val, "data", str(val)).strip('"') == "true"
    return False


def _is_array_ptr(t: Attribute) -> bool:
    """Return True if the type is !cc.ptr<!cc.array<...>>."""
    return isinstance(t, CcPtrType) and isinstance(t.element_type, CcArrayType)
