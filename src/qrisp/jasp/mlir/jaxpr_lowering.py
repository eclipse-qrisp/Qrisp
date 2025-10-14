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

from jax.interpreters.mlir import LoweringParameters, ModuleContext, lower_jaxpr_to_fun
from jaxlib.mlir import ir

def lower_jaxpr_to_MLIR(jaspr, lowering_rules = tuple([])):
    
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
        
        from jax._src.source_info_util import NameStack
        
        try:
            lower_jaxpr_to_fun(
                ctx,
                "main",
                jaspr,  # Pass the full ClosedJaxpr object
                jaspr.effects,
                public=True,
                name_stack=NameStack(),
            )
        except Exception as e:
            print(f"Error in lower_jaxpr_to_fun: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    return ctx.module