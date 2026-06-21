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
Utilities to lower a JAXPR produced by Qrisp to MLIR using the JASP dialect
and convert it into an xDSL module for downstream transformations.

Three key steps happen here:
1) Decompose composite gates in the Jaspr into their constituent gates (via ``decompose_composite_gates``).
2) Lower JAX primitives to our custom JASP dialect (via ``jaxpr_to_xdsl`` and ``jasp_lowering_rules``).
3) Run xDSL-based rewrites (e.g., control-flow fixes for quantum types).
"""

from xdsl.dialects import builtin

from qrisp.jasp.interpreter_tools.interpreters.composite_gate_interpreter import decompose_composite_gates
from qrisp.jasp.mlir.jaxpr_lowering import jaxpr_to_xdsl
from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules
from qrisp.jasp.mlir.quantum_control_flow import fix_quantum_control_flow
from qrisp.jasp.mlir.mlir_rewrites.scalar_tensor_folding import scalar_tensor_folding
from qrisp.jasp.mlir.mlir_rewrites.scalar_linalg_folding import scalar_linalg_folding
from qrisp.jasp.mlir.mlir_rewrites.cmpi_extui_folding import cmpi_extui_folding

from qrisp.jasp.jasp_expression import Jaspr


def jaspr_to_mlir(jaspr: Jaspr, lower_stableHLO=False) -> builtin.ModuleOp:
    """Convert a Jaspr to an xDSL MLIR module using the JASP dialect.

    This function lowers a Jaspr (JAX-traced quantum program) to MLIR with
    the JASP dialect for quantum operations, then applies xDSL-based rewrites
    to fix control flow for quantum types.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr object produced by Qrisp/JAX tracing.
    lower_stableHLO : bool, optional
        If True, runs additional MLIR passes to lower StableHLO operations
        (arithmetic, data ops) to lower-level dialects such as linalg, arith,
        and tensor. StableHLO control flow involving quantum types is preserved
        and rewritten to SCF by xDSL. The default is False.

    Returns
    -------
    builtin.ModuleOp
        The xDSL module representing the quantum computation with the JASP
        dialect and optionally lowered StableHLO operations.
    """
    jaspr_no_composite_gates = decompose_composite_gates(jaspr)
    xdsl_ctx, xdsl_module = jaxpr_to_xdsl(jaspr_no_composite_gates, lower_stableHLO, lowering_rules=jasp_lowering_rules)
    fix_quantum_control_flow(xdsl_module)

    # Run xDSL optimization passes to clean up lowering artifacts
    # (e.g., verbose scalar ``linalg.generic`` chains produced by
    # ``stablehlo-legalize-to-linalg``).
    if lower_stableHLO:
        scalar_linalg_folding(xdsl_ctx, xdsl_module)
        scalar_tensor_folding(xdsl_ctx, xdsl_module)
        cmpi_extui_folding(xdsl_ctx, xdsl_module)

    return xdsl_module
