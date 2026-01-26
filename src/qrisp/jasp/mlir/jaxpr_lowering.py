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

import sys
from io import StringIO

from jax.interpreters.mlir import LoweringParameters, ModuleContext, lower_jaxpr_to_fun
from jaxlib.mlir import ir
from jax._src import core

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects import builtin, func

def lower_jaxpr_to_MLIR(jaxpr, lowering_rules = tuple([])):
    """
    Lowers a Jaxpr object into an MLIR string uses Jax's MLIR infrastructure.

    Parameters
    ----------
    jaxpr : ClosedJaxpr
        The Jaxpr to lower.
    lowering_rules : tuple, optional
        The lowering rules to apply to custom primitives. 
        Check the jasp_lowering_rules file for an example. 
        The default is tuple([]).

    Returns
    -------
    MLIR Module
        An MLIR Module object in Jax's MLIR infrastructure.

    """
    # Create the necessary components for ModuleContext
    keepalives = []
    host_callbacks = []
    channel_iter = 1
    
    lowering_params = LoweringParameters(override_lowering_rules=lowering_rules)
    
    ctx = ModuleContext(
        backend=None,
        platforms=["cpu"],
        axis_context=None,
        keepalives=keepalives,
        channel_iterator=channel_iter,
        host_callbacks=host_callbacks,
        lowering_parameters=lowering_params,
    )
    
    # Enable unregistered dialects
    ctx.context.allow_unregistered_dialects = True
    
    # Lower JAXPR to MLIR using Catalyst's method
    with ctx.context, ir.Location.unknown(ctx.context):
        
        ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get("jasp_module")
        
        try:
            lower_jaxpr_to_fun(
                ctx,
                "main",
                jaxpr,  # Pass the full ClosedJaxpr object
                jaxpr.effects,
                num_const_args=len(core.jaxpr_const_args(jaxpr.jaxpr)),
                in_avals=[var.aval for var in core.jaxpr_const_args(jaxpr.jaxpr) + jaxpr.jaxpr.invars]
            )
        except Exception as e:
            print(f"Error in lower_jaxpr_to_fun: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    return ctx.module

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


def jaxpr_to_xdsl(jaxpr, lowering_rules = tuple([])):
    """Lower a JAXPR to an xDSL module.

    This function performs two steps:
    1) Use Qrisp's JAXPR lowering to create an MLIR module with custom ops.
    2) Print the module in MLIR's generic form and re-parse it with xDSL to
       obtain an xDSL ``builtin.module`` object.


    Parameters
    ----------
    jaspr:
        A ClosedJaxpr (or JAXPR-like) object produced by Qrisp/JAX tracing.

    Returns
    -------
    builtin.ModuleOp
        The xDSL module containing the lowered program.
    """
    # 1) Lower to MLIR (jasp dialect) using the custom lowering pipeline.
    mlir_module = lower_jaxpr_to_MLIR(jaxpr, lowering_rules = lowering_rules)

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
    return generic_mlir_to_xdsl(generic_mlir_string)
    
    