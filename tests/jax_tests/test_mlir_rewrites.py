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

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, llvm, tensor
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from qrisp.jasp.mlir.mlir_rewrites import cmpi_extui_folding, scalar_linalg_folding, scalar_tensor_folding


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
        """Constructs a standard, perfectly valid 0-D scalar tensor extraction sequence:
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
        """Constructs a test function containing an orphaned tensor.from_elements
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


# ==============================================================================
# Classical lowering to LLVM dialect
# ==============================================================================


# ==============================================================================
# Helpers
# ==============================================================================


def _run_classical_lowering(module, ctx):
    from qrisp.jasp.mlir.mlir_rewrites.classical_lowering import (
        classical_to_llvm,
    )
    classical_to_llvm(ctx, module)


def _make_binary_func(arg_type, op_builder):
    block = Block(arg_types=[arg_type, arg_type])
    result = op_builder(block.args[0], block.args[1])
    ret = func.ReturnOp(result)
    block.add_ops([result] if not isinstance(result, list) else result)
    block.add_op(ret)
    func_op = func.FuncOp(
        "test", ([arg_type, arg_type], [arg_type]), Region([block])
    )
    return builtin.ModuleOp([func_op]), block


# -- arith integer binary ops --------------------------------------------


@pytest.fixture
def llvm_ctx() -> Context:
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(llvm.LLVM)
    return ctx


def test_arith_addi_lowering(llvm_ctx):
    """``arith.addi`` (N-bit add, modulo 2^N) → ``llvm.add``."""
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.AddiOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.AddiOp) for op in ops)
    assert any(isinstance(op, llvm.AddOp) for op in ops)


def test_arith_subi_lowering(llvm_ctx):
    """``arith.subi`` (N-bit subtract, modulo 2^N) → ``llvm.sub``."""
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.SubiOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.SubiOp) for op in ops)
    assert any(isinstance(op, llvm.SubOp) for op in ops)


def test_arith_muli_lowering(llvm_ctx):
    """``arith.muli`` (N-bit multiply, low N bits of product) → ``llvm.mul``."""
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.MuliOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.MuliOp) for op in ops)
    assert any(isinstance(op, llvm.MulOp) for op in ops)


def test_arith_divui_lowering(llvm_ctx):
    """``arith.divui`` (unsigned division, trunc toward 0) → ``llvm.udiv``."""
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.DivUIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.DivUIOp) for op in ops)
    assert any(isinstance(op, llvm.UDivOp) for op in ops)


def test_arith_divsi_lowering(llvm_ctx):
    """``arith.divsi`` (signed division, trunc toward 0) → ``llvm.sdiv``.

    ``INT_MIN / -1`` and division by zero are UB in both dialects.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.DivSIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.DivSIOp) for op in ops)
    assert any(isinstance(op, llvm.SDivOp) for op in ops)


def test_arith_remui_lowering(llvm_ctx):
    """``arith.remui`` (unsigned remainder) → ``llvm.urem``.

    Both operands are treated as unsigned bitvectors — the result magnitude
    is ``|lhs| % |rhs|`` interpreted as unsigned.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.RemUIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.RemUIOp) for op in ops)
    assert any(isinstance(op, llvm.URemOp) for op in ops)


def test_arith_remsi_lowering(llvm_ctx):
    """``arith.remsi`` (signed remainder) → ``llvm.srem``.

    The result sign matches the dividend (``lhs``): ``-7 rem 3 = -1``,
    ``7 rem -3 = 1``.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.RemSIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.RemSIOp) for op in ops)
    assert any(isinstance(op, llvm.SRemOp) for op in ops)


def test_arith_andi_lowering(llvm_ctx):
    """``arith.andi`` (bitwise AND) → ``llvm.and``.

    Bitwise AND is identical for signed and unsigned types — both dialects
    treat the operands as bitvectors and compute ``lhs & rhs``.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.AndIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.AndIOp) for op in ops)
    assert any(isinstance(op, llvm.AndOp) for op in ops)


def test_arith_ori_lowering(llvm_ctx):
    """``arith.ori`` (bitwise OR) → ``llvm.or``."""
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.OrIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.OrIOp) for op in ops)
    assert any(isinstance(op, llvm.OrOp) for op in ops)


def test_arith_xori_lowering(llvm_ctx):
    """``arith.xori`` (bitwise XOR) → ``llvm.xor``."""
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.XOrIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.XOrIOp) for op in ops)
    assert any(isinstance(op, llvm.XOrOp) for op in ops)


def test_arith_shli_lowering(llvm_ctx):
    """``arith.shli`` (shift left, zero-fill) → ``llvm.shl``.

    ``rhs >= bitwidth`` is poison in both dialects.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.ShLIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.ShLIOp) for op in ops)
    assert any(isinstance(op, llvm.ShlOp) for op in ops)


def test_arith_shrsi_lowering(llvm_ctx):
    """``arith.shrsi`` (arithmetic / signed shift right) → ``llvm.ashr``.

    Fills high bits with copies of the sign bit (sign-extension).
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.ShRSIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.ShRSIOp) for op in ops)
    assert any(isinstance(op, llvm.AShrOp) for op in ops)


def test_arith_shrui_lowering(llvm_ctx):
    """``arith.shrui`` (logical / unsigned shift right) → ``llvm.lshr``.

    Fills high bits with zeros.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.ShRUIOp(a, b))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.ShRUIOp) for op in ops)
    assert any(isinstance(op, llvm.LShrOp) for op in ops)


# -- arith.cmpi ----------------------------------------------------------


def test_arith_cmpi_lowering(llvm_ctx):
    """``arith.cmpi`` (integer comparison) → ``llvm.icmp``.

    Predicate encoding (0 = eq, 1 = ne, 2 = slt, …, 9 = uge) is identical
    in both dialects. Result is ``i1``.
    """
    module, block = _make_binary_func(builtin.i32, lambda a, b: arith.CmpiOp(a, b, "eq"))
    _run_classical_lowering(module, llvm_ctx)
    ops = list(block.ops)
    assert not any(isinstance(op, arith.CmpiOp) for op in ops)
    assert any(isinstance(op, llvm.ICmpOp) for op in ops)
