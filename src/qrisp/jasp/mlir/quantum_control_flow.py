"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
Rewrite StableHLO control flow to SCF to support quantum types.

StableHLO control-flow ops like ``stablehlo.case`` and ``stablehlo.while`` do
not permit unregistered/foreign types (such as JASP quantum types). Here we
replace them with structurally equivalent SCF ops:

- stablehlo.case     -> scf.index_switch
- stablehlo.while    -> scf.while
- stablehlo.return   -> scf.yield or scf.condition (depending on region)

This pass operates on an xDSL module that contains unregistered ops for the
JASP dialect, so we explicitly allow unregistered ops in the xDSL context
outside of this file.
"""

from xdsl.dialects import builtin, tensor, arith, scf
from xdsl.pattern_rewriter import (
    PatternRewriter, 
    RewritePattern, 
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier
)

def fix_quantum_control_flow(xdsl_module: builtin.ModuleOp) -> None:
    """Rewrite StableHLO control-flow ops to SCF equivalents in-place.

    Parameters
    ----------
    xdsl_module:
        The xDSL module to be rewritten. The transformation is applied
        greedily and recursively over the whole module.
    """
    # Build the pattern set. Keep it small and focused so the greedy rewriter
    # converges quickly.
    patterns = [HLOControlFlowReplacement()]

    # Apply patterns using a greedy rewriter over the entire module.
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=True,
        walk_reverse=False,
    )

    walker.rewrite_module(xdsl_module)
    

class HLOControlFlowReplacement(RewritePattern):
    """Replace StableHLO control-flow ops with SCF equivalents.

    Specifically handles three cases:
    - "stablehlo.case"     -> scf.index_switch with i32->index cast for selector
    - "stablehlo.while"    -> scf.while (regions are cloned as-before/after)
    - "stablehlo.return"   -> scf.yield or scf.condition depending on parent
    """
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.UnregisteredOp, rewriter: PatternRewriter):
        """Pattern callback that matches unregistered StableHLO ops and rewrites them."""
        # Unregistered ops keep their name as a string attribute, including quotes.
        op_name = op.op_name.data

        # stablehlo.case -> scf.index_switch
        if op_name == "stablehlo.case":
            
            # If no quantum types are involved, we don't need to transform
            for tp in op.result_types + op.operand_types:
                if "jasp" in str(tp):
                    break
            else:
                return
            
            # StableHLO case selector is an i32 scalar; SCF switch expects an index.
            # We also enumerate cases 0..N-1 and use the last region as default.
            case_values = builtin.DenseArrayBase.from_list(
                builtin.i64, list(range(len(op.regions) - 1))
            )

            # Extract scalar from a tensor value (StableHLO convention) as i32 then cast to index.
            i32_case_indicator = tensor.ExtractOp(op.operands[0], [], builtin.i32)
            selector_index = arith.IndexCastOp(i32_case_indicator.result, builtin.IndexType())

            switch_op = scf.IndexSwitchOp(
                arg=selector_index,
                cases=case_values,
                default_region=op.regions[-1].clone(),
                case_regions=[region.clone() for region in op.regions[:-1]],
                result_types=op.result_types,
            )

            rewriter.replace_matched_op([i32_case_indicator, selector_index, switch_op])
            return

        # stablehlo.while -> scf.while
        if op_name == "stablehlo.while":
            
            # If no quantum types are involved, we don't need to transform
            for tp in op.result_types + op.operand_types:
                if "jasp" in str(tp):
                    break
            else:
                return
            
            while_op = scf.WhileOp(
                arguments=op.operands,
                result_types=op.result_types,
                before_region=op.regions[0].clone(),
                after_region=op.regions[1].clone(),
            )

            rewriter.replace_matched_op(while_op)
            return

        # stablehlo.return inside case/while regions
        if op_name == "stablehlo.return":
            parent_op = op.parent_op()

            # In switch arms: just yield the carried values
            if isinstance(parent_op, scf.IndexSwitchOp):
                yield_op = scf.YieldOp(*op.operands)
                rewriter.replace_matched_op(yield_op)
                return

            # In while regions: condition in before-region, yield in after-region
            if isinstance(parent_op, scf.WhileOp):
                # before-region (region 0) produces the loop condition and carried args
                if parent_op.regions[0].blocks[0] == op.parent:
                    # Extract i1 from tensor<i1> and build scf.condition
                    loop_cancellation = tensor.ExtractOp(op.operands[0], [], builtin.i1)
                    cond_op = scf.ConditionOp(loop_cancellation.result, *op.parent.args)
                    rewriter.replace_matched_op([loop_cancellation, cond_op])
                    return

                # after-region (region 1) yields carried values back to the loop
                if parent_op.regions[1].blocks[0] == op.parent:
                    yield_op = scf.YieldOp(*op.operands)
                    rewriter.replace_matched_op(yield_op)
                    return

        # No match: leave op unchanged
        return