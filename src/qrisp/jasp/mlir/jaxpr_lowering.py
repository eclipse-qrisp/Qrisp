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

import sys
from io import StringIO

from jax._src import core
from jax.interpreters.mlir import LoweringParameters, ModuleContext, lower_jaxpr_to_fun
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir._mlir_libs import _stablehlo, _mlirHlo, _chlo, _jax_mlir_ext

from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.parser import Parser

from qrisp.jasp.mlir.xdsl_dialect import JaspDialect
from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules


def lower_jaxpr_to_stablehlo_MLIR(jaxpr, lowering_rules=tuple([])):
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
                in_avals=[
                    var.aval
                    for var in core.jaxpr_const_args(jaxpr.jaxpr) + jaxpr.jaxpr.invars
                ],
                main_function=True,
            )

            # Remove unused functions (like shadow definitions for primitives)
            # symbol-dce removes private functions that are not referenced.
            # We set main_function=True above to ensure @main is public and preserved.
            # pm = passmanager.PassManager.parse("builtin.module(symbol-dce)")
            # pm.run(ctx.module.operation)

        except Exception as e:
            print(f"Error in lower_jaxpr_to_fun: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            raise

    return ctx.module


def MLIR_str_to_xdsl(mlir_string: str) -> builtin.ModuleOp:
    """Parse a generic-format MLIR string into an xDSL ``builtin.module``.

    Parameters
    ----------
    mlir_string:
        MLIR in generic op form. This is what MLIR prints when
        ``print_generic_op_form=True``.

    Returns
    -------
    Context
        The xDSL context in which the module resides.
    builtin.ModuleOp
        The parsed xDSL module operation. Unregistered ops are allowed so that
        custom JASP ops survive the round-trip.
    """
    # Create context with unregistered operations allowed so our custom ops
    # remain intact when parsed by xDSL.
    from xdsl.dialects import builtin, func, linalg, arith, tensor, scf, math
    from xdsl.parser import Parser

    ctx = Context()
    ctx.allow_unregistered = True

    # Register essential dialects used by the wrapper module and functions.
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(tensor.Tensor)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(math.Math)
    ctx.load_dialect(JaspDialect)

    # Parse the MLIR string and return the module op.
    parser = Parser(ctx, mlir_string)
    return ctx, parser.parse_module()


def jaxpr_to_xdsl(jaxpr, lower_stableHLO = False, lowering_rules=tuple([])):
    """Lower a JAXPR to an xDSL module.

    This function performs three steps:
    
    1. Use Qrisp's JAXPR lowering to create an MLIR module with custom ops.
    2. (Optional) Lower StableHLO to linalg and other lower-level dialects.
    3. Print the module in MLIR's generic form and re-parse it with xDSL to
       obtain an xDSL ``builtin.module`` object.

    Parameters
    ----------
    jaxpr : ClosedJaxpr
        A ClosedJaxpr (or JAXPR-like) object produced by Qrisp/JAX tracing.
    lower_stableHLO : bool, optional
        If True, runs MLIR passes to lower StableHLO arithmetic and data ops
        to linalg, arith, and tensor dialects. StableHLO control flow ops
        (case, while) are preserved for xDSL rewriting. The default is False.
    lowering_rules : tuple, optional
        Additional lowering rules to apply to custom primitives.
        The default is tuple([]).

    Returns
    -------
    Context
        The xDSL context in which the module resides.
    builtin.ModuleOp
        The xDSL module containing the lowered program.
    """

    # 1) Lower to MLIR (jasp dialect) using the custom lowering pipeline.
    if lower_stableHLO:
        mlir_module = lower_jaxpr_to_linalg_MLIR(jaxpr, lowering_rules=lowering_rules)
    else:
        mlir_module = lower_jaxpr_to_stablehlo_MLIR(jaxpr, lowering_rules=lowering_rules)

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
    xdsl_ctx, xdsl_module = MLIR_str_to_xdsl(generic_mlir_string)
    
    return xdsl_ctx, xdsl_module

def lower_jaxpr_to_linalg_MLIR(jaxpr, lowering_rules):
    """Lower a Jaxpr to an MLIR module with StableHLO ops lowered to linalg.

    This function applies MLIR passes to convert StableHLO arithmetic and
    data operations to lower-level dialects (linalg, arith, tensor).
    StableHLO control flow operations (case, while) that carry quantum types
    are left intact because they fail StableHLO type constraints - these are
    subsequently rewritten to SCF by xDSL.

    .. note::
    
        Not all StableHLO operations are lowered. Operations like 
        ``stablehlo.scatter`` (used for indexed array updates) remain as-is
        because they require specialized lowering not covered by the standard
        ``stablehlo-legalize-to-linalg`` pass.

    The pipeline applied:
    
    1. symbol-dce: removes unused private shadow functions that JAX emits
    2. stablehlo-convert-to-signless: converts to signless integers
    3. stablehlo-legalize-to-linalg: converts arithmetic/data ops to linalg

    Parameters
    ----------
    jaxpr : ClosedJaxpr
        The Jaxpr to lower.
    lowering_rules : tuple
        The lowering rules to apply to custom primitives.

    Returns
    -------
    MLIRModule
        An MLIR module object in JAX's MLIR infrastructure with StableHLO
        arithmetic lowered to linalg.
    """
    # --- Step 1: Lower Jaxpr to a JAX MLIR module ---------------------------
    mlir_module = lower_jaxpr_to_stablehlo_MLIR(jaxpr, lowering_rules=lowering_rules)

    # --- Step 2: Register passes and run inside the module's context ---------
    # lower_jaxpr_to_MLIR exits its `with ctx.context:` block, so we must
    # re-enter the MLIR context to use the PassManager.
    ctx = mlir_module.context
    with ctx:
        _stablehlo.register_dialect(ctx)
        _stablehlo.register_stablehlo_passes()
        _mlirHlo.register_mhlo_dialect(ctx)
        _mlirHlo.register_mhlo_passes()
        _chlo.register_dialect(ctx)

        # symbol-dce removes unused private shadow functions that JAX emits.
        # stablehlo-legalize-to-linalg converts arithmetic/data ops to linalg.
        # stablehlo control-flow ops (case, while) are left untouched here.
        pipeline = "builtin.module(" \
                   "symbol-dce," \
                   "stablehlo-convert-to-signless," \
                   "stablehlo-legalize-to-linalg" \
                   ")"

        pm = passmanager.PassManager.parse(pipeline)
        # Disable verifier: stablehlo.case carries !jasp.QuantumState which
        # fails StableHLO type constraints.  The legalizer itself only touches
        # arithmetic ops and leaves case/while alone, so skipping verification
        # is safe here.  The control-flow rewrite to SCF happens next via xDSL.
        pm.enable_verifier(False)
        pm.run(mlir_module.operation)

        # --- Step 3: Print to generic MLIR text ------------------------------
        generic_mlir = mlir_module.operation.get_asm(
            print_generic_op_form=True
        )

    
    return mlir_module
