
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
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.printer import Printer


@irdl_attr_definition
class CcStdVecType(ParametrizedAttribute, TypeAttribute):
    """CUDA-Q CC ``!cc.stdvec<!quake.measure>`` type."""
    name = "cc.stdvec"

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<!quake.measure>")


# ---------------------------------------------------------------------------
# CC ops for structured classical control flow
# ---------------------------------------------------------------------------

@irdl_op_definition
class CcIfOp(IRDLOperation):
    """Classical conditional: ``%res... = cc.if (%cond : i1) -> (types) { ... } else { ... }``"""

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
            result_types=[result_types],
            regions=[then_region, else_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" (")
        printer.print_ssa_value(self.cond)
        printer.print_string(")")
        
        # Output result types if the IF operation returns values
        if self.results:
            printer.print_string(" -> (")
            printer.print_list(self.results, lambda v: printer.print_attribute(v.type))
            printer.print_string(")")
            
        printer.print_string(" ")
        printer.print_region(self.then_region)
        printer.print_string(" else ")
        printer.print_region(self.else_region)


@irdl_op_definition
class CcLoopOp(IRDLOperation):
    """Classical loop: ``cc.loop while { cond } do { body } step { }``"""

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
            step_region = Region(Block())
        super().__init__(
            operands=[arguments],
            result_types=[result_types],
            regions=[while_region, body_region, step_region],
        )

    def print(self, printer: Printer) -> None:
        printer.print_string(" while ")
        
        if self.arguments:
            # Format: ((%arg = %init) -> (types))
            printer.print_string("((")
            block_args = self.while_region.blocks[0].args
            for i, (b_arg, init_val) in enumerate(zip(block_args, self.arguments)):
                if i > 0:
                    printer.print_string(", ")
                printer.print_ssa_value(b_arg)
                printer.print_string(" = ")
                printer.print_ssa_value(init_val)
            printer.print_string(") -> (")
            for i, res in enumerate(self.results):
                if i > 0:
                    printer.print_string(", ")
                printer.print_attribute(res.type)
            printer.print_string(")) ")
            
            # CUDA-Q parser expects the while block to omit the `^bb0(%arg):` header 
            # if args are declared in the loop signature.
            try:
                printer.print_region(self.while_region, print_entry_block_args=False)
            except TypeError:
                # Fallback for older xDSL versions
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
    """Continue to next ``cc.loop`` iteration: ``cc.continue %args...``."""
    name = "cc.continue"
    operands_ = var_operand_def(AnyAttr())

    def __init__(self, *operands: SSAValue) -> None:
        super().__init__(operands=[operands])

    def print(self, printer: Printer) -> None:
        if self.operands_:
            printer.print_string(" ")
            printer.print_list(self.operands_, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(self.operands_, lambda v: printer.print_attribute(v.type))


@irdl_op_definition
class CcConditionOp(IRDLOperation):
    """Loop back-edge condition for ``cc.loop`` while-region: ``cc.condition %cond (%args...)``."""

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
        
        if self.arguments:
            # Requires parentheses around forwarded arguments, not commas
            printer.print_string("(")
            printer.print_list(self.arguments, printer.print_ssa_value)
            printer.print_string(" : ")
            printer.print_list(
                self.arguments, lambda v: printer.print_attribute(v.type)
            )
            printer.print_string(")")


class CcDialect(Dialect):
    """Minimal xDSL dialect for CUDA-Q's CC (classical-control) dialect."""
    name = "cc"
    operations = [
        CcIfOp,
        CcLoopOp,
        CcBreakOp,
        CcContinueOp,
        CcConditionOp,
    ]
    attributes = [CcStdVecType]