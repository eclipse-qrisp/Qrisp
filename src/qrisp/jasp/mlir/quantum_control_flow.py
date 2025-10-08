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

from xdsl.dialects import builtin, tensor, arith, scf
from xdsl.pattern_rewriter import (
    PatternRewriter, 
    RewritePattern, 
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier
)

def fix_quantum_control_flow(xdsl_module):
    
    # This function fixes the problem that the control flow from stablehlo ("stablehlo.case", "stablehlo.while")
    # doesn't allow foreign types such as Qubits or QuantumStates.
    # We replace the control flow with the more general scf based control flow.
    
    # Create pattern list
    patterns = [
        HLOControlFlowReplacement(),
    ]

    # Apply patterns using greedy rewriter
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=True,
        walk_reverse=False
    )

    walker.rewrite_module(xdsl_module)
    

class HLOControlFlowReplacement(RewritePattern):
    """
    Pattern that folds arithmetic operations with constant operands.
    Example: %1 = arith.addi %c5, %c3 : i32  →  %1 = arith.constant 8 : i32
    """
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.UnregisteredOp, rewriter: PatternRewriter):
        """Fold addition of two constants"""
        # Check if both operands are constants
        
        # print(op.dialect_name())
        
        if str(op.op_name) == "\"stablehlo.case\"":
            
            case_values = builtin.DenseArrayBase.from_list(builtin.i64, list(range(len(op.regions)-1)))
            
            i32_case_indicator = tensor.ExtractOp(op.operands[0], [], builtin.i32)

            # Cast to index
            selector_index = arith.IndexCastOp(i32_case_indicator.result, builtin.IndexType())
            
            # The IndexSwitchOp returns an i32 selected from the matched region.
            switch_op = scf.IndexSwitchOp(
                arg = selector_index,
                cases = case_values,
                default_region = op.regions[-1].clone(),
                case_regions = [region.clone() for region in op.regions[:-1]],
                result_types = op.result_types,
            )
            
            rewriter.replace_matched_op([i32_case_indicator, selector_index, switch_op])
            
        if str(op.op_name) == "\"stablehlo.while\"":
            
            while_op = scf.WhileOp(arguments = op.operands,
                                   result_types = op.result_types,
                                   before_region = op.regions[0].clone(),
                                   after_region = op.regions[1].clone())
            
            rewriter.replace_matched_op(while_op)
            
        if str(op.op_name) == "\"stablehlo.return\"":
            
            parent_op = op.parent_op()
            
            if isinstance(parent_op, scf.IndexSwitchOp):
                
                yield_op = scf.YieldOp(*op.operands)
                rewriter.replace_matched_op(yield_op)
                
            elif isinstance(parent_op, scf.WhileOp):
                
                if parent_op.regions[0].blocks[0] == op.parent:
                    
                    loop_cancellation = tensor.ExtractOp(op.operands[0], [], builtin.i1)
                    
                    cond_op = scf.ConditionOp(loop_cancellation.result, *op.parent.args)
                    rewriter.replace_matched_op([loop_cancellation, cond_op])
                    
                elif parent_op.regions[1].blocks[0] == op.parent:
                    yield_op = scf.YieldOp(*op.operands)
        
        return