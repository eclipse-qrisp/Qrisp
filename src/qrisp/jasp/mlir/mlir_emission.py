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
1) Lower JAX primitives to our custom JASP dialect (via ``jaxpr_to_xdsl`` and ``jasp_lowering_rules``).
2) Run xDSL-based rewrites (e.g., control-flow fixes for quantum types).
"""

from xdsl.dialects import builtin

from qrisp.jasp.mlir.jaxpr_lowering import jaxpr_to_xdsl
from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules
from qrisp.jasp.mlir.quantum_control_flow import fix_quantum_control_flow

from qrisp.jasp.jasp_expression import Jaspr

def jaspr_to_mlir(jaspr: Jaspr) -> builtin.ModuleOp:
    
    xdsl_module = jaxpr_to_xdsl(jaspr, lowering_rules = jasp_lowering_rules)
    fix_quantum_control_flow(xdsl_module)

    return xdsl_module