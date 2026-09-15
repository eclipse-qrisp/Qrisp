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

"""Provide IR helpers shared by the Quake lowering passes."""

# Shared IR helpers
# =================
#
# Type predicates and attribute constructors that several lowering passes need.
# They live here rather than in one of the passes so that the passes agree on
# what a scalar tensor is, on how a literal becomes an attribute, and on the
# sentinel MLIR uses for a dynamic extent.

from xdsl.dialects.builtin import (
    BFloat16Type,
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    TensorType,
)
from xdsl.ir import Attribute

from qrisp.jasp.cudaq_interface.quake_lowering.dialects.cc_dialect import CcArrayType, CcPtrType

# MLIR's sentinel for "dynamic dimension/offset"
_MLIR_DYNAMIC = -9223372036854775808


# ===================================================================
# Type predicates
# ===================================================================


def _is_ranked_tensor(t: Attribute) -> bool:
    """Return True if *t* is a tensor type with at least one dimension."""
    return isinstance(t, TensorType) and len(t.get_shape()) > 0


def _is_scalar_tensor(t: Attribute) -> bool:
    """Return True if *t* is a rank-0 tensor type, the shape a scalar round-trips through."""
    return isinstance(t, TensorType) and not t.get_shape()


def _is_rank_1_tensor(t: Attribute) -> bool:
    """Return True if *t* is a tensor type with exactly one dimension."""
    return isinstance(t, TensorType) and len(t.get_shape()) == 1


def _is_array_pointer(t: Attribute) -> bool:
    """Return True if *t* is ``!cc.ptr<!cc.array<...>>``."""
    return isinstance(t, CcPtrType) and isinstance(t.element_type, CcArrayType)


def _is_float_type(t: Attribute) -> bool:
    """Return True for any xDSL float type."""
    return isinstance(t, (Float16Type, Float32Type, Float64Type, BFloat16Type))


# ===================================================================
# Attributes
# ===================================================================


def _scalar_attr(value, scalar_type: Attribute) -> Attribute | None:
    """Return *value* as an attribute of *scalar_type*, or None if that type holds no literal."""
    if _is_float_type(scalar_type):
        return FloatAttr(float(value), scalar_type)
    if isinstance(scalar_type, (IntegerType, IndexType)):
        return IntegerAttr(int(value), scalar_type)
    return None


def _dense_values(attr: Attribute) -> list | None:
    """Return the literal elements of a dense attribute, or None if it holds none."""
    if not isinstance(attr, DenseIntOrFPElementsAttr):
        return None
    try:
        return list(attr.iter_values())
    except (AttributeError, TypeError):
        return None
