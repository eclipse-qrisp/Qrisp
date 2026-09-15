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

"""Lower CC arrays to CUDA-Q sequences."""

# CC Array Pointer → CC Sequence Lowering (Entrypoint Only)
# =========================================================
#
# Rewrites !cc.ptr<!cc.array<T x N>> parameters in the CUDA-Q entrypoint
# to !cc.sequence<T> for runtime compatibility.
#
# Approach
# --------
# - Replace the array pointer parameter with a sequence in the entrypoint
#   signature.
# - Immediately extract the data pointer and cast it back to the original
#   statically-sized array pointer type.
# - All internal functions continue to use the original static array type
#   unchanged, requiring no call graph analysis or propagation.

from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import Attribute, FunctionType, ModuleOp
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter

from qrisp.jasp.cudaq_interface.quake_lowering.dialects.cc_dialect import (
    CcArrayType,
    CcCastOp,
    CcPtrType,
    CcSequenceDataOp,
    CcSequenceType,
)

_MLIR_DYNAMIC = -9223372036854775808


# ===================================================================
# Public entry point
# ===================================================================


def _lower_array_to_sequence(module: ModuleOp) -> None:
    """Rewrite entrypoint array pointer args to sequences with immediate cast-back."""
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([EntrypointArrayToSequencePattern()]),
        walk_regions_first=False,
    )
    walker.rewrite_module(module)


# ===================================================================
# Rewrite Pattern
# ===================================================================


class EntrypointArrayToSequencePattern(RewritePattern):
    """Rewrite entrypoint array params to sequences with an immediate cast to the static array pointer."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.FuncOp, rewriter: PatternRewriter) -> None:
        """Convert entrypoint array parameters to CUDA-Q sequences."""
        if not _is_entrypoint(op):
            return

        block = op.body.blocks[0]
        new_inputs = list(op.function_type.inputs)
        modified = False

        for idx, arg in enumerate(block.args):
            old_type = arg.type
            if not _is_array_ptr(old_type):
                continue

            # Already converted — break greedy loop
            if isinstance(old_type, CcSequenceType):
                continue

            modified = True
            elem_type = old_type.element_type.element_type
            sequence_type = CcSequenceType(elem_type)
            dyn_ptr_type = CcPtrType(CcArrayType(elem_type, _MLIR_DYNAMIC))

            # Extract a dynamic-sized array pointer from the sequence.
            data_op = CcSequenceDataOp(arg, dyn_ptr_type)
            rewriter.insert_op(data_op, InsertPoint.at_start(block))

            # Cast dynamic pointer back to the original static pointer type
            cast_op = CcCastOp(data_op.result, old_type)
            rewriter.insert_op(cast_op, InsertPoint.after(data_op))

            # Replace all uses of the old arg (except the data_op itself) with the cast result
            for use in list(arg.uses):
                if use.operation is data_op:
                    continue
                use.operation.operands[use.index] = cast_op.result

            # Update block argument type to sequence.
            Rewriter.replace_value_with_new_type(arg, sequence_type)
            new_inputs[idx] = sequence_type

        if modified:
            op.function_type = FunctionType.from_lists(new_inputs, list(op.function_type.outputs))


# ===================================================================
# Helpers
# ===================================================================


def _is_entrypoint(func_op: func_dialect.FuncOp) -> bool:
    """Check if the function has the CUDA-Q entrypoint attribute."""
    if "cudaq.entrypoint" in func_op.attributes:
        val = func_op.attributes["cudaq.entrypoint"]
        return getattr(val, "data", str(val)).strip('"') == "true"
    return False


def _is_array_ptr(t: Attribute) -> bool:
    """Return True if the type is !cc.ptr<!cc.array<...>>."""
    return isinstance(t, CcPtrType) and isinstance(t.element_type, CcArrayType)
