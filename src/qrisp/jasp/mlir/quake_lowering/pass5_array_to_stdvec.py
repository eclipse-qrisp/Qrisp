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
"""

"""
PASS 5 – CC Array Pointer → CC StdVec Lowering (Entrypoint Only)
=================================================================

Rewrites ``!cc.ptr<!cc.array<T x N>>`` parameters in the CUDA-Q entrypoint
to ``!cc.stdvec<T>`` for runtime compatibility.

Strategy:
  - Replace the array pointer parameter with a stdvec in the entrypoint signature.
  - Immediately extract the data pointer and cast it back to the original
    statically-sized array pointer type.
  - All internal functions continue to use the original static array type
    unchanged, requiring no call graph analysis or propagation.
"""

from xdsl.dialects import func as func_dialect
from xdsl.dialects.builtin import ModuleOp, FunctionType, Attribute

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


def lower_array_to_stdvec(module: ModuleOp) -> None:
    """In-place pass: Rewrite entrypoint array pointer args to stdvec with immediate cast-back."""
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([EntrypointArrayToStdVecPattern()]),
        walk_regions_first=False,
    )
    walker.rewrite_module(module)


# ===================================================================
# Rewrite Pattern
# ===================================================================


class EntrypointArrayToStdVecPattern(RewritePattern):
    """Rewrites entrypoint array pointer params to stdvec with immediate cast to static array pointer."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func_dialect.FuncOp, rewriter: PatternRewriter) -> None:
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
            if isinstance(old_type, CcStdVecType):
                continue

            modified = True
            elem_type = old_type.element_type.element_type
            stdvec_type = CcStdVecType(elem_type)
            dyn_ptr_type = CcPtrType(CcArrayType(elem_type, _MLIR_DYNAMIC))

            # Insert stdvec_data to extract a dynamic-sized array pointer
            data_op = CcStdVecDataOp(arg, dyn_ptr_type)
            rewriter.insert_op(data_op, InsertPoint.at_start(block))

            # Cast dynamic pointer back to the original static pointer type
            cast_op = CcCastOp(data_op.result, old_type)
            rewriter.insert_op(cast_op, InsertPoint.after(data_op))

            # Replace all uses of the old arg (except the data_op itself) with the cast result
            for use in list(arg.uses):
                if use.operation is data_op:
                    continue
                use.operation.operands[use.index] = cast_op.result

            # Update block argument type to stdvec
            Rewriter.replace_value_with_new_type(arg, stdvec_type)
            new_inputs[idx] = stdvec_type

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
