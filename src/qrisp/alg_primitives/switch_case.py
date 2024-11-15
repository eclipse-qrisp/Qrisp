"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

from qrisp.core.gate_application_functions import h
from qrisp.alg_primitives.qft import QFT
from qrisp.core import QuantumArray
from qrisp.qtypes import QuantumBool
from qrisp.environments import conjugate, control
from qrisp.alg_primitives import demux

def qswitch(operand, case, case_function_list, method = "sequential"):
    """
    Executes a switch - case statement distinguishing between a list of
    given in-place functions.
    
    Parameters
    ----------
    operand : QuantumVariable
        The QuantumVariable to operate on.
    case : QuantumFloat
        The QuantumFloat indicating which functions to execute.
    case_function_list : list[callable]
        The list of functions which are executed depending on the case.
        

    """
    
    if method == "sequential":
    
        for i in range(len(case_function_list)):
            with i == case:
                case_function_list[i](operand)
        
    elif method == "parallel":
        
        # Idea: Use demux function to move operand and enabling bool into QuantumArray
        # to execute cases in parallel.
        case_amount = len(case_function_list)
    
        # This QuantumArray acts as an addressable QRAM via the demux function
        enable = QuantumArray(qtype = QuantumBool(), shape = (case_amount,))
        enable[0].flip()
        
        qa = QuantumArray(qtype = operand, shape = ((case_amount,)))
    
        with conjugate(demux)(operand, case, qa, parallelize_qc = True):
            with conjugate(demux)(enable[0], case, enable, parallelize_qc = True):
                for i in range(case_amount):
                    with control(enable[i]):
                        case_function_list[i](qa[i])
                        
        qa.delete()
        
        enable[0].flip()
        enable.delete()
    
    else:
        raise Exception("Don't know compile method {method} for switch-case structure.")