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

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects import builtin, func

from qrisp.jasp.mlir.jaspr_lowering import lower_jaspr_to_MLIR_raw
from qrisp.jasp.mlir.quantum_control_flow import fix_quantum_control_flow



def generic_mlir_to_xdsl(mlir_string):
    """
    Convert generic MLIR string to xDSL module.
    
    Args:
        mlir_string: MLIR string in generic format
        
    Returns:
        xDSL Operation (module)
    """
    # Create context with unregistered operations allowed
    ctx = Context()
    ctx.allow_unregistered = True
    
    # Register essential dialects
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(func.Func)
    
    # Parse MLIR
    parser = Parser(ctx, mlir_string)
    return parser.parse_operation()

def jaspr_to_mlir(jaspr):
    
    mlir_module = lower_jaspr_to_MLIR_raw(jaspr)
    
    # Capture generic format output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    # Print in generic format - key for xDSL compatibility
    mlir_module.operation.print(print_generic_op_form=True)
    
    sys.stdout = old_stdout
    
    generic_mlir_string = captured_output.getvalue()
    
    xdsl_module = generic_mlir_to_xdsl(generic_mlir_string)
    
    fix_quantum_control_flow(xdsl_module)
    
    return xdsl_module