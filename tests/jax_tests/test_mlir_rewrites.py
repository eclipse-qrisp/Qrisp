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

import pytest
from qrisp.jasp.mlir.mlir_rewrites import cmpi_extui_folding, scalar_linalg_folding, scalar_tensor_folding
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap


@pytest.fixture
def ctx() -> Context:
    """Provides an xDSL context with the necessary dialects registered."""
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(tensor.Tensor)
    return ctx


class TestFoldScalarLinalgGeneric:
    """Unit tests for the scalar_linalg_folding rewrite pattern."""

    @staticmethod
    def build_test_ir(is_0d: bool = True, num_outputs: int = 1):
        """Constructs a test IR inside a func.FuncOp to prevent DCE."""

        f32 = builtin.f32
        tensor_type = builtin.TensorType(f32, [])

        block = Block(arg_types=[tensor_type, tensor_type])
        in_tensor = block.args[0]
        out_tensor = block.args[1]

        inner_block = Block(arg_types=[f32, f32])
        add_op = arith.AddfOp(inner_block.args[0], inner_block.args[0])
        yield_op = linalg.YieldOp(add_op.result)
        inner_block.add_ops([add_op, yield_op])

        map_0d = builtin.AffineMapAttr(AffineMap(0, 0, []))
        maps = [map_0d, map_0d]
        if num_outputs == 2:
            maps.append(map_0d)
        maps_attr = builtin.ArrayAttr(maps)

        if is_0d:
            iter_types = builtin.ArrayAttr([])
        else:
            iter_types = builtin.ArrayAttr([builtin.StringAttr("parallel")])

        inputs = [in_tensor]
        outputs = [out_tensor] if num_outputs == 1 else [out_tensor, out_tensor]

        generic_op = linalg.GenericOp(
            inputs=inputs,
            outputs=outputs,
            result_types=[out.type for out in outputs],
            body=Region([inner_block]),
            indexing_maps=maps_attr,
            iterator_types=iter_types,
        )

        return_op = func.ReturnOp(*generic_op.results)
        block.add_ops([generic_op, return_op])

        func_op = func.FuncOp(
            "test_wrapper", ([tensor_type, tensor_type], [tensor_type] * num_outputs), Region([block])
        )

        module = builtin.ModuleOp([func_op])
        return module, block

    def test_successful_fold(self, ctx):
        """Tests that a valid 0D generic is properly unpacked."""
        module, block = self.build_test_ir(is_0d=True, num_outputs=1)
        scalar_linalg_folding(ctx, module)

        ops = list(block.ops)

        assert not any(isinstance(op, linalg.GenericOp) for op in ops)
        assert any(isinstance(op, tensor.ExtractOp) for op in ops)
        assert any(isinstance(op, arith.AddfOp) for op in ops)
        assert any(isinstance(op, tensor.FromElementsOp) for op in ops)

    @pytest.mark.parametrize(
        "is_0d, num_outputs, reason",
        [
            (False, 1, "The generic op is not 0-dimensional (has iterator_types)"),
            (True, 2, "The generic op produces multiple outputs"),
        ],
    )
    def test_no_folds(self, ctx, is_0d, num_outputs, reason):
        """Tests edge cases that violate preconditions and should trigger early returns."""
        module, block = self.build_test_ir(is_0d=is_0d, num_outputs=num_outputs)
        scalar_linalg_folding(ctx, module)

        ops = list(block.ops)

        assert any(isinstance(op, linalg.GenericOp) for op in ops), f"Failed on: {reason}"
        assert not any(isinstance(op, tensor.ExtractOp) for op in ops)
        assert not any(isinstance(op, tensor.FromElementsOp) for op in ops)


class TestScalarTensorFolding:
    """Unit tests for the scalar_tensor_folding rewrite pattern."""

    @staticmethod
    def build_extract_test_ir():
        """
        Constructs a standard, perfectly valid 0-D scalar tensor extraction sequence:
          %tensor = tensor.from_elements %arg0
          %extracted = tensor.extract %tensor
          func.return %extracted
        """
        f32 = builtin.f32
        block = Block(arg_types=[f32])
        scalar_input = block.args[0]

        tensor_type = builtin.TensorType(f32, [])
        from_elements_op = tensor.FromElementsOp(operands=[scalar_input], result_types=[tensor_type])
        defining_val = from_elements_op.result
        ops_to_add = [from_elements_op]

        extract_op = tensor.ExtractOp.create(operands=[defining_val], result_types=[f32])
        ops_to_add.append(extract_op)

        return_op = func.ReturnOp(extract_op.result)
        ops_to_add.append(return_op)

        block.add_ops(ops_to_add)

        func_op = func.FuncOp("test_extract_wrapper", ([f32], [f32]), Region([block]))

        return builtin.ModuleOp([func_op]), block

    @staticmethod
    def build_dead_from_elements_ir(has_uses: bool = False):
        """
        Constructs a test function containing an orphaned tensor.from_elements
        to verify custom targeted Dead Code Elimination (DCE).
        """
        f32 = builtin.f32
        block = Block(arg_types=[f32])
        scalar_input = block.args[0]

        tensor_type = builtin.TensorType(f32, [])

        from_elements_op = tensor.FromElementsOp(operands=[scalar_input], result_types=[tensor_type])

        ops = [from_elements_op]

        if has_uses:
            return_op = func.ReturnOp(from_elements_op.result)
            ops.append(return_op)
            ret_types = [tensor_type]
        else:
            return_op = func.ReturnOp(scalar_input)
            ops.append(return_op)
            ret_types = [f32]

        block.add_ops(ops)

        func_op = func.FuncOp("test_dce_wrapper", ([f32], ret_types), Region([block]))

        return builtin.ModuleOp([func_op]), block

    def test_successful_bypass_extract(self, ctx):
        """Tests that tensor.extract on a 0-D tensor bypasses to the raw element."""
        module, block = self.build_extract_test_ir()
        scalar_tensor_folding(ctx, module)

        ops = list(block.ops)

        # 1. The tensor.extract operation should be completely bypassed and erased
        assert not any(isinstance(op, tensor.ExtractOp) for op in ops)

        # 2. EraseDeadFromElements should clean up the stranded tensor.from_elements op
        assert not any(isinstance(op, tensor.FromElementsOp) for op in ops)

        # 3. The return should point directly to the function block input argument
        return_op = next(op for op in ops if isinstance(op, func.ReturnOp))
        assert return_op.operands[0] == block.args[0]

    def test_erase_dead_from_elements(self, ctx):
        """Tests that EraseDeadFromElements successfully purges an abandoned tensor."""
        module, block = self.build_dead_from_elements_ir(has_uses=False)
        scalar_tensor_folding(ctx, module)

        ops = list(block.ops)

        # The abandoned from_elements operation must be erased
        assert not any(isinstance(op, tensor.FromElementsOp) for op in ops)

    def test_preserve_live_from_elements(self, ctx):
        """Tests that EraseDeadFromElements preserves tensors that actually have users."""
        module, block = self.build_dead_from_elements_ir(has_uses=True)
        scalar_tensor_folding(ctx, module)

        ops = list(block.ops)

        # It has a user (the ReturnOp), so it must be preserved
        assert any(isinstance(op, tensor.FromElementsOp) for op in ops)


class TestCmpiExtUIFolding:
    """Unit tests for the cmpi_extui_folding rewrite pattern."""

    @staticmethod
    def build_test_ir(pred_int: int, rhs_val: int, lhs_is_extui: bool = True):
        """Constructs a test IR inside a func.FuncOp to prevent DCE."""

        # 1. Create a block that takes exactly one i1 argument.
        # This block argument (block.args[0]) acts as our abstract, un-foldable value!
        block = Block(arg_types=[builtin.i1])
        orig_bool = block.args[0]

        ops = []

        # Use the abstract block argument instead of a constant
        if lhs_is_extui:
            lhs_op = arith.ExtUIOp(orig_bool, builtin.i32)
        else:
            lhs_op = arith.ConstantOp.from_int_and_width(0, 32)
        ops.append(lhs_op)

        rhs_op = arith.ConstantOp.from_int_and_width(rhs_val, 32)
        ops.append(rhs_op)

        cmp_op = arith.CmpiOp(lhs_op.result, rhs_op.result, pred_int)
        ops.append(cmp_op)

        # Consumes the output to strictly prevent Dead Code Elimination
        return_op = func.ReturnOp(cmp_op.result)
        ops.append(return_op)

        # Add all built operations into the block
        block.add_ops(ops)

        # 2. Wrap the block in a Function
        # The signature ([builtin.i1], [builtin.i1]) means: takes i1, returns i1
        func_op = func.FuncOp("test_wrapper", ([builtin.i1], [builtin.i1]), Region([block]))

        # 3. Place the function inside the MLIR Module
        module = builtin.ModuleOp([func_op])

        return module, block

    @pytest.mark.parametrize(
        "pred_int, rhs_val, expected_result",
        [
            # ======= vs 0 =======
            (0, 0, "NOT"),  # eq  0 → NOT %x
            (1, 0, "IDENTITY"),  # ne  0 → %x
            (2, 0, "FALSE"),  # slt 0 → false
            (3, 0, "NOT"),  # sle 0 → NOT %x
            (4, 0, "IDENTITY"),  # sgt 0 → %x
            (5, 0, "TRUE"),  # sge 0 → true
            (6, 0, "FALSE"),  # ult 0 → false
            (7, 0, "NOT"),  # ule 0 → NOT %x
            (8, 0, "IDENTITY"),  # ugt 0 → %x
            (9, 0, "TRUE"),  # uge 0 → true
            # ======= vs 1 =======
            (0, 1, "IDENTITY"),  # eq  1 → %x
            (1, 1, "NOT"),  # ne  1 → NOT %x
            (2, 1, "NOT"),  # slt 1 → NOT %x
            (3, 1, "TRUE"),  # sle 1 → true
            (4, 1, "FALSE"),  # sgt 1 → false
            (5, 1, "IDENTITY"),  # sge 1 → %x
            (6, 1, "NOT"),  # ult 1 → NOT %x
            (7, 1, "TRUE"),  # ule 1 → true
            (8, 1, "FALSE"),  # ugt 1 → false
            (9, 1, "IDENTITY"),  # uge 1 → %x
        ],
    )
    def test_all_fold_cases(self, ctx, pred_int, rhs_val, expected_result):
        """Exhaustively tests every predicate vs 0 and 1 from the _FOLD_TABLE."""
        module, block = self.build_test_ir(pred_int=pred_int, rhs_val=rhs_val)
        cmpi_extui_folding(ctx, module)

        ops = list(block.ops)

        # 1. The original cmpi must ALWAYS be erased
        assert not any(isinstance(op, arith.CmpiOp) for op in ops)

        # 2. Verify the correct outcome
        if expected_result == "IDENTITY":
            # No new operations should be inserted (just uses original i1)
            assert not any(isinstance(op, arith.XOrIOp) for op in ops)

        elif expected_result == "NOT":
            # An XOR with 1 should be inserted
            assert any(isinstance(op, arith.XOrIOp) for op in ops)
            xor_op = next(op for op in ops if isinstance(op, arith.XOrIOp))

            # Verify the RHS of the XOR comes from a ConstantOp of 1:i1
            rhs_op = xor_op.rhs.owner
            assert isinstance(rhs_op, arith.ConstantOp)
            assert rhs_op.value.value.data in [-1, 1]
            assert xor_op.rhs.type == builtin.i1

        elif expected_result == "TRUE":
            # Verify that a constant true (1 : i1) exists in the block
            true_consts = [
                op
                for op in ops
                if isinstance(op, arith.ConstantOp) and op.value.type == builtin.i1 and op.value.value.data in [-1, 1]
            ]
            assert len(true_consts) >= 1, "Expected a constant true (1 : i1) but found none."

        elif expected_result == "FALSE":
            # Verify that a constant false (0 : i1) exists in the block
            false_consts = [
                op
                for op in ops
                if isinstance(op, arith.ConstantOp) and op.value.type == builtin.i1 and op.value.value.data == 0
            ]
            assert len(false_consts) >= 1, "Expected a constant false (0 : i1) but found none."

    @pytest.mark.parametrize(
        "pred_int, rhs_val, lhs_is_extui, reason",
        [
            (0, 0, False, "LHS is a constant, not an ExtUIOp"),
            (0, 2, True, "RHS is 2, not 0 or 1"),
            (10, 0, True, "Predicate 10 is out of bounds/unsupported"),
        ],
    )
    def test_no_folds(self, ctx, pred_int, rhs_val, lhs_is_extui, reason):
        """Test edge cases that should trigger the early returns (no rewrite)."""
        module, block = self.build_test_ir(pred_int=pred_int, rhs_val=rhs_val, lhs_is_extui=lhs_is_extui)
        cmpi_extui_folding(ctx, module)

        ops = list(block.ops)
        # The rewrite should have aborted, leaving the CmpiOp untouched
        assert any(isinstance(op, arith.CmpiOp) for op in ops), f"Failed on: {reason}"
