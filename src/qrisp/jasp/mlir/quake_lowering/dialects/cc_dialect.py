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
xDSL dialect definitions for the CUDA-Q Classical Control (CC) dialect.

Covers:
- Types (!cc.ptr<T>, !cc.array<T x N>, !cc.stdvec<T>, !cc.struct<"name" {T1, T2, ...}>)
- Local memory ops (cc.alloca, cc.store, cc.load)
- Structured control-flow ops (cc.if, cc.loop, cc.condition, cc.continue, cc.break)
- Pointer arithmetic ops (cc.compute_ptr, cc.cast)
- StdVec ops (cc.stdvec_data)
- Struct ops (cc.undef, cc.insert_value, cc.log_output) — used by CUDA-Q
  preparation for multi-return packing and .run variant synthesis.

Control-flow semantics (matching native CUDA-Q output):

- ``cc.if`` supports **returning values** via ``cc.continue %val : type``
  terminators in each branch, analogous to ``scf.if`` with ``scf.yield``.
- ``cc.loop`` carries loop-state via **block arguments** in its while-region,
  forwarded by ``cc.condition`` and updated by ``cc.continue`` in body/step.
- ``cc.continue`` is the universal branch terminator: it carries operands
  both inside ``cc.if`` branches (as return values) and inside ``cc.loop``
  body/step regions (as loop-carried values).

References
----------
https://github.com/NVIDIA/cuda-quantum/tree/main/lib/Optimizer/Dialect/CC
"""

from typing import Sequence

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i1, i64, StringAttr
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Region,
    SSAValue,
    TypeAttribute,
    ParametrizedAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    attr_def,
    operand_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.printer import Printer

# ---------------------------------------------------------------------------
# CC types
# ---------------------------------------------------------------------------


@irdl_attr_definition
class CcArrayType(ParametrizedAttribute, TypeAttribute):
    """Fixed-size or dynamic-size array type: ``!cc.array<T x N>`` or ``!cc.array<T x ?>``.

    Use size=-1 to represent dynamic size (prints as ``?``).
    """

    name = "cc.array"

    element_type: Attribute
    size: IntegerAttr

    def __init__(self, element_type: Attribute, size: int) -> None:
        super().__init__(element_type, IntegerAttr(size, i64))

    def print_parameters(self, printer: Printer) -> None:
        size_val = self.size.value.data
        printer.print_string("<")
        printer.print_attribute(self.element_type)
        if size_val < 0:
            printer.print_string(" x ?>")
        else:
            printer.print_string(f" x {size_val}>")


@irdl_attr_definition
class CcPtrType(ParametrizedAttribute, TypeAttribute):
    """Pointer type: ``!cc.ptr<T>``."""

    name = "cc.ptr"

    element_type: Attribute

    def __init__(self, element_type: Attribute) -> None:
        super().__init__(element_type)


@irdl_attr_definition
class CcStdVecType(ParametrizedAttribute, TypeAttribute):
    """CUDA-Q CC ``!cc.stdvec<T>`` type.

    When used without an explicit element_type parameter, prints as
    ``!cc.stdvec<!quake.measure>`` (the return type of quake.mz on veq).

    When constructed with an element_type parameter, prints as
    ``!cc.stdvec<T>`` for the given element type (used for array params).
    """

    name = "cc.stdvec"

    element_type: Attribute

    def __init__(self, element_type: Attribute | None = None) -> None:
        if element_type is None:
            # Sentinel: use a StringAttr to mark "quake.measure" without importing it
            element_type = StringAttr("!quake.measure")
        super().__init__(element_type)

    def print_parameters(self, printer: Printer) -> None:

        if isinstance(self.element_type, StringAttr):
            # Legacy: print as <!quake.measure>
            printer.print_string("<")
            printer.print_string(self.element_type.data)
            printer.print_string(">")
        else:
            printer.print_string("<")
            printer.print_attribute(self.element_type)
            printer.print_string(">")


@irdl_attr_definition
class CcStructType(ParametrizedAttribute, TypeAttribute):
    """Struct type: ``!cc.struct<"name" {T1, T2, ...}>``.

    Used by CUDA-Q to pack multiple return values into a single SSA value
    for the .run execution mode.

    Examples
    --------
    ::

        !cc.struct<"tuple" {i64, f64}>
        !cc.struct<"tuple" {i64, i64, i64}>
    """

    name = "cc.struct"

    struct_name: Attribute  # StringAttr
    field_types: Attribute  # ArrayAttr

    def __init__(self, struct_name: str, field_types: Sequence[Attribute]) -> None:

        super().__init__(StringAttr(struct_name), ArrayAttr(list(field_types)))

    def print_parameters(self, printer: Printer) -> None:

        printer.print_string('<"')
        printer.print_string(self.struct_name.data)
        printer.print_string('" {')
        fields = list(self.field_types.data)
        for i, f in enumerate(fields):
            if i > 0:
                printer.print_string(", ")
            printer.print_attribute(f)
        printer.print_string("}>")


# ---------------------------------------------------------------------------
# Memory ops (used by pass1 measurement lowering)
# ---------------------------------------------------------------------------


@irdl_op_definition
class CcAllocaOp(IRDLOperation):
    """``%ptr = cc.alloca T``"""

    name = "cc.alloca"
    elem_type = attr_def(Attribute)
    result = result_def(AnyAttr())

    def __init__(self, elem_type: Attribute) -> None:
        super().__init__(
            attributes={"elem_type": elem_type},
            result_types=[CcPtrType(elem_type)],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.elem_type)


@irdl_op_definition
class CcStoreOp(IRDLOperation):
    """``cc.store %val, %ptr : !cc.ptr<T>``"""

    name = "cc.store"
    value = operand_def(AnyAttr())
    ptr = operand_def(AnyAttr())

    def __init__(self, value: SSAValue, ptr: SSAValue) -> None:
        super().__init__(operands=[value, ptr])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(", ")
        printer.print_ssa_value(self.ptr)
        printer.print_string(" : ")
        printer.print_attribute(self.ptr.type)


@irdl_op_definition
class CcLoadOp(IRDLOperation):
    """``%val = cc.load %ptr : !cc.ptr<T>``"""

    name = "cc.load"
    ptr = operand_def(AnyAttr())
    result = result_def(AnyAttr())

    def __init__(self, ptr: SSAValue) -> None:
        if not hasattr(ptr.type, "element_type"):
            raise TypeError(f"CcLoadOp requires a CcPtrType operand, got {ptr.type}")
        super().__init__(operands=[ptr], result_types=[ptr.type.element_type])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.ptr)
        printer.print_string(" : ")
        printer.print_attribute(self.ptr.type)


# ---------------------------------------------------------------------------
# Control-flow ops (pure SSA, matching native CUDA-Q output)
# ---------------------------------------------------------------------------


@irdl_op_definition
class CcIfOp(IRDLOperation):
    """Classical conditional with optional results.

    Result-free form::

        cc.if(%cond) {
          ...
          cc.continue
        } else {
          ...
          cc.continue
        }

    Result-carrying form (proven by native CUDA-Q output)::

        %res = cc.if(%cond) -> i64 {
          ...
          cc.continue %val : i64
        } else {
          ...
          cc.continue %val : i64
        }
    """

    name = "cc.if"
    cond = operand_def(i1)
    then_region = region_def()
    else_region = region_def()
    res = var_result_def(AnyAttr())

    def __init__(
        self,
        cond: SSAValue,
        result_types: Sequence[Attribute],
        then_region: Region,
        else_region: Region | None = None,
    ) -> None:
        if else_region is None:
            else_region = Region([Block()])
        super().__init__(
            operands=[cond],
            result_types=[list(result_types)],
            regions=[then_region, else_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string("(")
        printer.print_ssa_value(self.cond)
        printer.print_string(")")

        res_list = list(self.res)
        if res_list:
            printer.print_string(" -> ")
            if len(res_list) == 1:
                printer.print_attribute(res_list[0].type)
            else:
                printer.print_string("(")
                printer.print_list(res_list, lambda v: printer.print_attribute(v.type))
                printer.print_string(")")

        printer.print_string(" ")
        printer.print_region(self.then_region)
        printer.print_string(" else ")
        printer.print_region(self.else_region)


@irdl_op_definition
class CcLoopOp(IRDLOperation):
    """Classical loop with SSA loop-carried values.

    Native CUDA-Q format::

        %res = cc.loop while ((%arg0 = %init) -> (i64)) {
          ...
          cc.condition %cond(%arg0 : i64)
        } do {
        ^bb0(%arg0: i64):
          ...
          cc.continue %arg0 : i64
        } step {
        ^bb0(%arg0: i64):
          ...
          cc.continue %next : i64
        }
    """

    name = "cc.loop"
    while_region = region_def()
    body_region = region_def()
    step_region = region_def()
    arguments = var_operand_def(AnyAttr())
    res = var_result_def(AnyAttr())

    def __init__(
        self,
        arguments: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        while_region: Region,
        body_region: Region,
        step_region: Region | None = None,
    ) -> None:
        if step_region is None:
            step_region = Region([Block()])
        super().__init__(
            operands=[list(arguments)],
            result_types=[list(result_types)],
            regions=[while_region, body_region, step_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" while ")

        if self.arguments:
            # Format: ((%arg = %init) -> (types))
            printer.print_string("((")
            block_args = list(self.while_region.blocks[0].args)
            for i, (b_arg, init_val) in enumerate(zip(block_args, self.arguments)):
                if i > 0:
                    printer.print_string(", ")
                printer.print_ssa_value(b_arg)
                printer.print_string(" = ")
                printer.print_ssa_value(init_val)
            printer.print_string(") -> (")
            res_list = list(self.res)
            for i, r in enumerate(res_list):
                if i > 0:
                    printer.print_string(", ")
                printer.print_attribute(r.type)
            printer.print_string(")) ")
            try:
                printer.print_region(self.while_region, print_entry_block_args=False)
            except TypeError:
                printer.print_region(self.while_region)
        else:
            printer.print_region(self.while_region)

        printer.print_string(" do ")
        printer.print_region(self.body_region)
        printer.print_string(" step ")
        printer.print_region(self.step_region)


@irdl_op_definition
class CcBreakOp(IRDLOperation):
    """Break out of a ``cc.loop``: ``cc.break``."""

    name = "cc.break"

    def __init__(self) -> None:
        super().__init__()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class CcContinueOp(IRDLOperation):
    """Universal branch terminator: ``cc.continue %args... : types``.

    Used in:
    - ``cc.if`` branches to yield return values.
    - ``cc.loop`` body/step to carry loop-state to the next iteration.
    - Empty (no operands) for side-effect-only branches.
    """

    name = "cc.continue"
    operands_ = var_operand_def(AnyAttr())

    def __init__(self, *operands: SSAValue) -> None:
        super().__init__(operands=[list(operands)])

    def print(self, printer: Printer) -> None:
        ops = list(self.operands_)
        if ops:
            printer.print_string(" ")
            printer.print_list(ops, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(ops, lambda v: printer.print_attribute(v.type))


@irdl_op_definition
class CcConditionOp(IRDLOperation):
    """Loop back-edge condition: ``cc.condition %cond(%args... : types)``.

    The forwarded arguments are passed to the next iteration's block args.
    """

    name = "cc.condition"
    cond = operand_def(i1)
    arguments = var_operand_def(AnyAttr())

    def __init__(self, cond: SSAValue, *args: SSAValue) -> None:
        super().__init__(operands=(cond, list(args)))

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.cond)
        args = list(self.arguments)
        if args:
            printer.print_string("(")
            printer.print_list(args, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(args, lambda v: printer.print_attribute(v.type))
            printer.print_string(")")


# ---------------------------------------------------------------------------
# Pointer arithmetic ops
# ---------------------------------------------------------------------------


@irdl_op_definition
class CcComputePtrOp(IRDLOperation):
    """Compute a pointer to an array element.

    Static index::

        %ptr = cc.compute_ptr %arr[2] : (!cc.ptr<!cc.array<f64 x 5>>) -> !cc.ptr<f64>

    Dynamic index::

        %ptr = cc.compute_ptr %arr[%i] : (!cc.ptr<!cc.array<f64 x 5>>, i64) -> !cc.ptr<f64>
    """

    name = "cc.compute_ptr"
    base = operand_def(AnyAttr())
    dynamic_index = var_operand_def(AnyAttr())  # empty for static, [i64] for dynamic
    static_index = attr_def(Attribute, default=None)  # IntegerAttr or None
    result = result_def(AnyAttr())

    def __init__(self, base: SSAValue, index, element_type: Attribute) -> None:
        """
        Parameters
        ----------
        base : SSAValue
            Pointer to the array (!cc.ptr<!cc.array<T x N>>).
        index : int | SSAValue
            Static integer index or dynamic SSAValue (i64).
        element_type : Attribute
            The element type (T) for the result pointer type.
        """

        if isinstance(index, int):
            # Static index
            super().__init__(
                operands=[base, []],
                attributes={"static_index": IntegerAttr(index, i64)},
                result_types=[CcPtrType(element_type)],
            )
        else:
            # Dynamic index
            super().__init__(
                operands=[base, [index]],
                result_types=[CcPtrType(element_type)],
            )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.base)
        printer.print_string("[")
        dyn = list(self.dynamic_index)
        if dyn:
            printer.print_ssa_value(dyn[0])
        elif self.static_index is not None:
            idx = getattr(self.static_index, "value", self.static_index)
            idx_val = getattr(idx, "data", idx)
            printer.print_string(str(idx_val))
        printer.print_string("] : (")
        printer.print_attribute(self.base.type)
        if dyn:
            printer.print_string(", ")
            printer.print_attribute(dyn[0].type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class CcCastOp(IRDLOperation):
    """Cast between CC pointer types.

    ``%ptr_elem = cc.cast %ptr_arr : (!cc.ptr<!cc.array<f64 x 5>>) -> !cc.ptr<f64>``
    """

    name = "cc.cast"
    value = operand_def(AnyAttr())
    result = result_def(AnyAttr())

    def __init__(self, value: SSAValue, result_type: Attribute) -> None:
        super().__init__(operands=[value], result_types=[result_type])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(" : (")
        printer.print_attribute(self.value.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)


# ---------------------------------------------------------------------------
# StdVec ops (used by pass4 array-to-stdvec lowering)
# ---------------------------------------------------------------------------


@irdl_op_definition
class CcStdVecDataOp(IRDLOperation):
    """Extract raw data pointer from a stdvec.

    ``%ptr = cc.stdvec_data %vec : (!cc.stdvec<T>) -> !cc.ptr<!cc.array<T x ?>>``
    """

    name = "cc.stdvec_data"
    vec = operand_def(AnyAttr())
    result = result_def(AnyAttr())

    def __init__(self, vec: SSAValue, result_type: Attribute) -> None:
        super().__init__(operands=[vec], result_types=[result_type])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.vec)
        printer.print_string(" : (")
        printer.print_attribute(self.vec.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)


# ===========================================================================
# Struct / value ops (used by pass6 CUDA-Q preparation)
# ===========================================================================


@irdl_op_definition
class CcUndefOp(IRDLOperation):
    """Create an undefined value of a given type.

    Typically used to create an uninitialized struct that is then populated
    with ``cc.insert_value``::

        %s = cc.undef !cc.struct<"tuple" {i64, f64}>
    """

    name = "cc.undef"
    result = result_def(AnyAttr())

    def __init__(self, result_type: Attribute) -> None:
        super().__init__(result_types=[result_type])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class CcInsertValueOp(IRDLOperation):
    """Insert a value into a struct at a given index.

    ::

        %new = cc.insert_value %struct[0], %val : (!cc.struct<"tuple" {i64, f64}>, i64) -> !cc.struct<"tuple" {i64, f64}>

    The result type is always the same as the struct type.
    """

    name = "cc.insert_value"
    struct = operand_def(AnyAttr())
    value = operand_def(AnyAttr())
    index = attr_def(IntegerAttr)
    result = result_def(AnyAttr())

    def __init__(self, struct: SSAValue, index: int, value: SSAValue) -> None:
        super().__init__(
            operands=[struct, value],
            attributes={"index": IntegerAttr(index, i64)},
            result_types=[struct.type],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.struct)
        idx_val = self.index.value.data
        printer.print_string(f"[{idx_val}], ")
        printer.print_ssa_value(self.value)
        printer.print_string(" : (")
        printer.print_attribute(self.struct.type)
        printer.print_string(", ")
        printer.print_attribute(self.value.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class CcLogOutputOp(IRDLOperation):
    """Log a classical value for retrieval by ``cudaq.run``.

    ::

        cc.log_output %val : i64

    This op is a side-effecting terminator-like op that records the value
    for the CUDA-Q runtime to collect across shots. It has no results.
    """

    name = "cc.log_output"
    value = operand_def(AnyAttr())

    def __init__(self, value: SSAValue) -> None:
        super().__init__(operands=[value])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


class CcDialect(Dialect):
    """xDSL dialect for CUDA-Q's CC (classical-control) dialect."""

    name = "cc"
    operations = [
        # Control flow
        CcIfOp,
        CcLoopOp,
        CcBreakOp,
        CcContinueOp,
        CcConditionOp,
        # Memory
        CcAllocaOp,
        CcStoreOp,
        CcLoadOp,
        # Pointer arithmetic
        CcComputePtrOp,
        CcCastOp,
        # StdVec
        CcStdVecDataOp,
        # Struct / value
        CcUndefOp,
        CcInsertValueOp,
        CcLogOutputOp,
    ]
    attributes = [CcArrayType, CcPtrType, CcStdVecType, CcStructType]
