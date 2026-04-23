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
xDSL dialect definitions for the CUDA-Q Classical Control (CC) dialect.

Only the structured control-flow ops needed for SCF→CC lowering are defined.

References
----------
https://github.com/NVIDIA/cuda-quantum (cuda_quantum/mlir/include/cudaq/Optimizer/Dialect/CC)
"""


from typing import Sequence

from xdsl.dialects.builtin import i1
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
    operand_def,
    opt_result_def,
    region_def,
    var_operand_def,
    var_result_def,
    attr_def,
    result_def,
)
from xdsl.printer import Printer


# ---------------------------------------------------------------------------
# CC types
# ---------------------------------------------------------------------------


@irdl_attr_definition
class CcStdVecType(ParametrizedAttribute, TypeAttribute):
    """CUDA-Q CC ``!cc.stdvec<!quake.measure>`` type — return type of ``quake.mz`` on a veq.

    Each qubit in a ``!quake.veq<?>`` produces a ``!quake.measure`` value; measuring
    the whole register returns a span of those values.

    Example::

        %ms = quake.mz %veq : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
    """

    name = "cc.stdvec"

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<!quake.measure>")

@irdl_attr_definition
class CcPtrType(ParametrizedAttribute, TypeAttribute):
    """Pointer type for the CC dialect: !cc.ptr<T>"""
    name = "cc.ptr"
    
    # In xDSL 0.55.4, this class-level annotation defines the attribute's parameters
    element_type: Attribute

    def __init__(self, element_type: Attribute) -> None:
        # Pass the parameter directly (without the list brackets!)
        super().__init__(element_type)


# ---------------------------------------------------------------------------
# CC ops for local memory management
# ---------------------------------------------------------------------------


@irdl_op_definition
class CcAllocaOp(IRDLOperation):
    """Allocates memory for a given type: cc.alloca i64 -> !cc.ptr<i64>"""
    name = "cc.alloca"
    elem_type = attr_def(Attribute)
    result = result_def(AnyAttr())

    def __init__(self, elem_type: Attribute) -> None:
        super().__init__(
            attributes={"elem_type": elem_type},
            result_types=[CcPtrType(elem_type)]
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.elem_type)


@irdl_op_definition
class CcStoreOp(IRDLOperation):
    """Stores a value into a pointer: cc.store %val, %ptr : !cc.ptr<i64>"""
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
    """Loads a value from a pointer: %res = cc.load %ptr : !cc.ptr<i64>"""
    name = "cc.load"
    ptr = operand_def(AnyAttr())
    result = result_def(AnyAttr())

    def __init__(self, ptr: SSAValue) -> None:
        # Infer the result type from the pointer's inner element type
        res_type = ptr.type.element_type if hasattr(ptr.type, "element_type") else ptr.type
        super().__init__(operands=[ptr], result_types=[res_type])

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.ptr)
        printer.print_string(" : ")
        printer.print_attribute(self.ptr.type)


# ---------------------------------------------------------------------------
# CC ops for structured classical control flow
# ---------------------------------------------------------------------------


@irdl_op_definition
class CcIfOp(IRDLOperation):
    """Classical conditional: ``cc.if (%cond : i1) { ... } else { ... }``

    Unlike ``scf.if``, ``cc.if`` does **not** return values.  Value passing
    across branches should be done via ``cc.alloca`` / ``cc.store`` / ``cc.load``
    (handled by the calling compiler).  For this lowering we only emit the
    structural shell; downstream passes are responsible for value threading.

    The then-region and else-region each contain exactly one block.
    """

    name = "cc.if"
    cond = operand_def(i1)
    then_region = region_def()
    else_region = region_def()

    def __init__(
        self,
        cond: SSAValue,
        then_region: Region,
        else_region: Region | None = None,
    ) -> None:
        if else_region is None:
            else_region = Region(Block())
        super().__init__(
            operands=[cond],
            regions=[then_region, else_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" (")
        printer.print_ssa_value(self.cond)
        printer.print_string(")")
        printer.print_string(" ")
        printer.print_region(self.then_region)
        # Always emit else, even if empty – required by CC dialect parser.
        printer.print_string(" else ")
        printer.print_region(self.else_region)


@irdl_op_definition
class CcLoopOp(IRDLOperation):
    """Classical loop: ``cc.loop while { cond } do { body } step { }``

    Maps to the CC dialect's general ``cc.loop`` construct which can represent
    both *while* and *for* loops depending on the step region.

    Block arguments carry the loop-carried values (same semantics as
    ``scf.while``).
    """

    name = "cc.loop"
    while_region = region_def()   # condition evaluation (like scf.while before)
    body_region = region_def()    # loop body       (like scf.while after)
    step_region = region_def()    # step / update   (empty for plain while)
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
            step_region = Region(Block())
        super().__init__(
            operands=[arguments],
            result_types=[result_types],
            regions=[while_region, body_region, step_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" while ")
        printer.print_region(self.while_region)
        printer.print_string(" do ")
        printer.print_region(self.body_region)
        printer.print_string(" step ")
        printer.print_region(self.step_region)
        if self.arguments:
            printer.print_string(" (")
            printer.print_list(self.arguments, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(
                self.arguments, lambda v: printer.print_attribute(v.type)
            )
            printer.print_string(")")


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
    """Continue to next ``cc.loop`` iteration: ``cc.continue``."""

    name = "cc.continue"

    def __init__(self) -> None:
        super().__init__()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class CcConditionOp(IRDLOperation):
    """Loop back-edge condition for ``cc.loop`` while-region: ``cc.condition %cond``."""

    name = "cc.condition"
    cond = operand_def(i1)
    arguments = var_operand_def(AnyAttr())

    def __init__(
        self,
        cond: SSAValue,
        *args: SSAValue,
    ) -> None:
        super().__init__(operands=(cond, args))

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_ssa_value(self.cond)
        if list(self.arguments):
            printer.print_string(", ")
            printer.print_list(self.arguments, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(
                self.arguments, lambda v: printer.print_attribute(v.type)
            )


# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


class CcDialect(Dialect):
    """Minimal xDSL dialect for CUDA-Q's CC (classical-control) dialect."""

    name = "cc"
    operations = [
        CcIfOp,
        CcLoopOp,
        CcBreakOp,
        CcContinueOp,
        CcConditionOp,
        CcAllocaOp,
        CcStoreOp,
        CcLoadOp,
    ]
    attributes = [CcStdVecType, CcPtrType]