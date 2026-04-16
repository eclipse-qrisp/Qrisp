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
PASS 3 – Tensor unwrapping and function signature rewrite.

After Passes 1 and 2, the IR may still contain:

* ``arith.constant dense<N> : tensor<T>`` ops whose values are consumed
  exclusively by quantum-adjacent ops that now expect scalars (``i64``, etc.).
* Lingering ``tensor<T>`` function argument/return types (rare after Pass 1 but
  possible for auxiliary helper functions).

This pass folds simple dense-constant tensors into scalar ``arith.constant``
ops where they feed ``tensor.extract`` ops, effectively unwrapping the tensor
round-trip introduced by the JAX/StableHLO→arith lowering.

The following transformations are performed:

1. ``arith.constant dense<V> : tensor<T>`` where *all* uses are
   ``tensor.extract []`` (rank-0 extraction) → replace with
   ``arith.constant V : T`` and remove the now-dead tensor constant +
   extract ops.

2. ``tensor.extract %t [] : tensor<T>`` where ``%t`` was already a scalar
   (type == ``T``) → replace uses of the extract result with ``%t`` and
   erase the extract.

3. ``func.func`` argument types that are ``tensor<T>`` for trivially
   scalar-able types are *not* unwrapped here – that would require ABI
   changes which are out of scope.  The pass only touches *internal* IR.
"""


from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    FloatAttr,
    TensorType,
    i1,
    i32,
    i64,
    f64,
)
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def unwrap_tensors(module) -> None:
    """In-place PASS 3: fold trivial tensor constants and extract ops.

    Parameters
    ----------
    module:
        An xDSL ``builtin.ModuleOp`` previously processed by Passes 1 and 2.
    """
    changed = True
    # Iterate until no more changes (fold chains can span multiple passes)
    for _ in range(10):
        changed = _process_module(module)
        if not changed:
            break


def _process_module(module) -> bool:
    changed = False
    for op in list(module.body.blocks[0].ops):
        changed |= _process_region_changed(op.regions[0] if op.regions else None)
    return changed


def _process_region_changed(region) -> bool:
    if region is None:
        return False
    changed = False
    for block in region.blocks:
        changed |= _process_block_changed(block)
    return changed


def _process_block_changed(block: Block) -> bool:
    changed = False
    for op in list(block.ops):
        changed |= _fold_op(op, block)
        # Recurse into nested regions
        for region in op.regions:
            changed |= _process_region_changed(region)
    return changed


def _fold_op(op, block: Block) -> bool:
    """Try to fold *op*.  Return True if the IR was modified."""

    # Case 1: arith.constant dense<V> : tensor<T> where T is a scalar type
    # and all users are tensor.extract [] ops.
    if op.name == "arith.constant":
        return _fold_dense_constant(op, block)

    # Case 2: tensor.extract %t [] : tensor<T> where %t is a scalar.
    if op.name == "tensor.extract":
        return _fold_extract(op, block)

    return False


def _fold_dense_constant(const_op, block: Block) -> bool:
    """Fold ``arith.constant dense<V> : tensor<T>`` → ``arith.constant V : T``."""
    result = const_op.result
    tensor_type = result.type

    if not isinstance(tensor_type, TensorType):
        return False
    scalar_type = tensor_type.element_type

    # Only fold rank-0 tensors
    if hasattr(tensor_type, "shape") and tensor_type.shape.data:
        return False  # Non-scalar tensor – leave alone.

    # Check that all uses are tensor.extract [] ops
    uses = list(result.uses)
    if not uses:
        return False

    extract_ops = []
    for use in uses:
        user = use.operation
        if user.name != "tensor.extract":
            return False
        # Must have no indices (rank-0 extraction)
        if len(list(user.operands)) != 1:
            return False
        extract_ops.append(user)

    # Build the scalar constant value attribute
    val_attr = const_op.properties.get("value") or const_op.attributes.get("value")
    if val_attr is None:
        return False

    scalar_attr = _dense_to_scalar_attr(val_attr, scalar_type)
    if scalar_attr is None:
        return False

    scalar_const = arith.ConstantOp(scalar_attr)
    block.insert_ops_before([scalar_const], const_op)

    # Replace all extract uses with the scalar constant
    for extr in extract_ops:
        extr.results[0].replace_all_uses_with(scalar_const.result)
        Rewriter.erase_op(extr, safe_erase=False)

    Rewriter.erase_op(const_op, safe_erase=False)
    return True


def _fold_extract(extr_op, block: Block) -> bool:
    """Fold ``tensor.extract %scalar [] : tensor<T>`` when %scalar is already T."""
    if len(list(extr_op.operands)) != 1:
        return False
    src = extr_op.operands[0]
    if isinstance(src.type, TensorType):
        return False  # Source is still a tensor – can't fold yet.
    # Source is already a scalar
    extr_op.results[0].replace_all_uses_with(src)
    Rewriter.erase_op(extr_op, safe_erase=False)
    return True


def _dense_to_scalar_attr(dense_attr, scalar_type):
    """Extract a scalar attribute from a DenseIntOrFPElementsAttr."""
    if not isinstance(dense_attr, DenseIntOrFPElementsAttr):
        return None
    vals = list(dense_attr.iter_values())
    if not vals:
        return None
    first = vals[0]

    if scalar_type in (i1, i32, i64):
        return IntegerAttr(int(first), scalar_type)
    if scalar_type == f64:
        return FloatAttr(float(first), scalar_type)
    return None
