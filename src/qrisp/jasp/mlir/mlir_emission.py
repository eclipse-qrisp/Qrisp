"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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
Utilities to lower a JAXPR produced by Qrisp to MLIR using the JASP dialect
and convert it into an xDSL module for downstream transformations.

Two key steps happen here:
1) Lower JAX primitives to our custom JASP dialect (via ``lower_jaspr_to_MLIR_raw``).
2) Print the MLIR module in generic form and parse it with xDSL so we can run
   xDSL-based rewrites (e.g., control-flow fixes for quantum types).
"""

import sys
from io import StringIO
from typing import Any

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects import builtin, func

from qrisp.jasp.mlir.jaspr_lowering import lower_jaspr_to_MLIR_raw
from qrisp.jasp.mlir.quantum_control_flow import fix_quantum_control_flow


def generic_mlir_to_xdsl(mlir_string: str) -> builtin.ModuleOp:
    """Parse a generic-format MLIR string into an xDSL ``builtin.module``.

    Parameters
    ----------
    mlir_string:
        MLIR in generic op form. This is what MLIR prints when
        ``print_generic_op_form=True``.

    Returns
    -------
    builtin.ModuleOp
        The parsed xDSL module operation. Unregistered ops are allowed so that
        custom JASP ops survive the round-trip.
    """
    # Create context with unregistered operations allowed so our custom ops
    # remain intact when parsed by xDSL.
    ctx = Context()
    ctx.allow_unregistered = True

    # Register essential dialects used by the wrapper module and functions.
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)

    # Parse the MLIR string and return the module op.
    parser = Parser(ctx, mlir_string)
    return parser.parse_operation()

def jaspr_to_mlir(jaspr: Any) -> builtin.ModuleOp:
    """Lower a JAXPR to MLIR (JASP dialect), then to an xDSL module.

    This function performs three steps:
    1) Use Qrisp's JAXPR lowering to create an MLIR module with JASP ops.
    2) Print the module in MLIR's generic form and re-parse it with xDSL to
       obtain an xDSL ``builtin.module`` object.
    3) Apply structural fixes for quantum control flow (e.g., replace
       StableHLO control-flow with SCF to support quantum types).

    Parameters
    ----------
    jaspr:
        A ClosedJaxpr (or JAXPR-like) object produced by Qrisp/JAX tracing.

    Returns
    -------
    builtin.ModuleOp
        The xDSL module containing the lowered program. Some control-flow ops
        may be rewritten to SCF so that JASP quantum types are supported.
    """
    # 1) Lower to MLIR (jasp dialect) using the custom lowering pipeline.
    mlir_module = lower_jaspr_to_MLIR_raw(jaspr)

    # 2) Capture generic MLIR printing. Use try/finally to avoid stdout leaks
    #    if an exception is thrown during printing.
    old_stdout = sys.stdout
    captured_output = StringIO()
    try:
        sys.stdout = captured_output
        # Print in generic format - key for xDSL compatibility
        mlir_module.operation.print(print_generic_op_form=True)
    finally:
        sys.stdout = old_stdout

    generic_mlir_string = captured_output.getvalue()

    # Parse to xDSL.
    xdsl_module = generic_mlir_to_xdsl(generic_mlir_string)

    # 3) Fix control-flow for quantum types in-place.
    fix_quantum_control_flow(xdsl_module)

    return xdsl_module