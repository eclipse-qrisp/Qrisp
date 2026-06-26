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

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    Float16Type,
    Float32Type,
    Float64Type,
    BFloat16Type,
    TensorType,
    f64,
)
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    SSAValue,
)

from qrisp.jasp.mlir.quake_lowering.jasp_to_quake.quake_dialect import (
    QuakeRefType,
    QuakeVeqType,
    VeqSizeOp,
)
from qrisp.jasp.mlir.xdsl_dialect import (
    QuantumStateType,
    QubitArrayType,
    QubitType,
)

# ---------------------------------------------------------------------------
# Helpers to identify Jasp types
# ---------------------------------------------------------------------------


def _is_qst(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.QuantumState``."""
    return isinstance(t, QuantumStateType)


def _is_qubit_array(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.QubitArray``."""
    return isinstance(t, QubitArrayType)


def _is_qubit(t: Attribute) -> bool:
    """Return True if *t* is ``!jasp.Qubit``."""
    return isinstance(t, QubitType)


def _quake_type_for(jasp_type: Attribute) -> Attribute | None:
    """Map a Jasp qubit type to its Quake equivalent, or None for qst."""
    if _is_qubit_array(jasp_type):
        return QuakeVeqType()
    if _is_qubit(jasp_type):
        return QuakeRefType()
    return None  # QuantumState → dropped


# ---------------------------------------------------------------------------
# Numeric-type helpers
# ---------------------------------------------------------------------------


def _is_qubit_type(t: Attribute) -> bool:
    """Return True if *t* is any qubit/veq type (Jasp or Quake)."""
    return isinstance(t, (QuakeRefType, QuakeVeqType)) or _is_qubit(t) or _is_qubit_array(t)


def _is_numeric_type(t: Attribute) -> bool:
    """Return True if *t* is a scalar or rank-0 tensor of int/float."""

    scalar = t
    if isinstance(t, TensorType):
        if t.get_shape():  # only rank-0
            return False
        scalar = t.element_type
    return isinstance(scalar, IntegerType) or _is_float_type(scalar)


def _is_float_type(t: Attribute) -> bool:
    """Return True for any xDSL float type."""

    return isinstance(t, (Float16Type, Float32Type, Float64Type, BFloat16Type))


def _scalar_type_of(t: Attribute) -> Attribute:
    """Return the scalar element type (unwrap rank-0 tensor if needed)."""

    if isinstance(t, TensorType) and not t.get_shape():
        return t.element_type
    return t


def _coerce_to_f64(val: SSAValue, block: Block, insert_before: Operation) -> SSAValue:
    """Extract from tensor if needed, then cast int→f64 if needed.

    Returns an SSAValue of type f64.
    """

    # Step 1: unwrap rank-0 tensor → scalar
    scalar_type = _scalar_type_of(val.type)
    scalar = _extract_scalar(val, scalar_type, block, insert_before)

    # Step 2: already f64 → done
    if scalar.type == f64:
        return scalar

    # Step 3: other float → fpext/fptrunc to f64
    if _is_float_type(scalar.type):
        cast = arith.ExtFOp(scalar, f64)  # or SIToFPOp etc.
        block.insert_ops_before([cast], insert_before)
        return cast.result

    # Step 4: integer → sitofp to f64
    if isinstance(scalar.type, IntegerType):
        cast = arith.SIToFPOp(scalar, f64)
        block.insert_ops_before([cast], insert_before)
        return cast.result

    raise ValueError(f"Cannot coerce {scalar.type} to f64")


def _normalize_index_for_veq(veq: SSAValue, idx: SSAValue, block: Block, insert_before_op: Operation) -> SSAValue:
    """Implement Python-style negative indexing on a veq index.

    If idx < 0, return idx + size(veq), else idx.

    All ops are inserted before `insert_before_op`.
    Assumes `idx` is an i64 scalar.
    """
    # len = quake.veq_size %veq
    size = VeqSizeOp(veq)

    zero = arith.ConstantOp(IntegerAttr(0, 64))
    is_neg = arith.CmpiOp(idx, zero.result, "slt")  # idx < 0 ?
    idx_plus_size = arith.AddiOp(idx, size.result)  # idx + len
    norm = arith.SelectOp(is_neg.result, idx_plus_size.result, idx)

    block.insert_ops_before(
        [size, zero, is_neg, idx_plus_size, norm],
        insert_before_op,
    )
    return norm.result


# ---------------------------------------------------------------------------
# Tensor-extract helper
# ---------------------------------------------------------------------------


def _extract_scalar(val: SSAValue, scalar_type: Attribute, block: Block, insert_before_op: Operation) -> SSAValue:
    """Insert a ``tensor.extract %val[] : tensor<T>`` op and return the result.

    If *val* is already of *scalar_type* (not a tensor), return it unchanged.
    """
    if val.type == scalar_type:
        return val
    extract = tensor.ExtractOp(val, [], scalar_type)
    block.insert_ops_before([extract], insert_before_op)
    return extract.result


def _wrap_scalar(val: SSAValue, tensor_type: Attribute, block: Block, insert_before_op: Operation) -> SSAValue:
    """Insert ``tensor.from_elements %val : tensor<T>`` and return the result.

    If *val* is already of *tensor_type*, return it unchanged.
    """
    if val.type == tensor_type:
        return val
    from_elem = tensor.FromElementsOp(operands=[[val]], result_types=[tensor_type])
    block.insert_ops_before([from_elem], insert_before_op)
    return from_elem.result
