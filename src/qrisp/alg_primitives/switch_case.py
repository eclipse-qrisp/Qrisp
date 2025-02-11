"""
\********************************************************************************
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
********************************************************************************/
"""

from qrisp.core import QuantumArray
from qrisp.qtypes import QuantumBool
from qrisp.environments import conjugate, control
from qrisp.alg_primitives import demux
from qrisp.core.gate_application_functions import mcx
from qrisp.jasp import check_for_tracing_mode

def qswitch(operand, case, case_function_list, method = "sequential"):
    """
    Executes a switch - case statement distinguishing between a list of
    given in-place functions.
        

    Parameters
    ----------
    operand : QuantumVariable
        The quantum argument on which to execute the case function.
    case : QuantumFloat
        A QuantumFloat specifying which case function should be executed.
    case_function_list : list[callable]
        A list of functions, performing some in-place operation on ``operand``.
    method : str, optional
        The compilation method. Available are ``parallel`` and ``sequential``. 
        ``parallel`` is exponentially fast but requires more temporary qubits.
        The default is "sequential".

    Examples
    --------
    
    We create some sample functions:

    >>> from qrisp import *
    >>> def f0(x): x += 1
    >>> def f1(x): inpl_mult(x, 3, treat_overflow = False)
    >>> def f2(x): pass
    >>> def f3(x): h(x[1])
    >>> case_function_list = [f0, f1, f2, f3]

    Create operand and case variable
    
    >>> operand = QuantumFloat(4)
    >>> operand[:] = 1
    >>> case = QuantumFloat(2)
    >>> h(case)

    Execute switch_case function
    
    >>> qswitch(operand, case, case_function_list)
    
    Simulate
    
    >>> print(multi_measurement([case, operand]))
    {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}

    """
    
    
    if method == "sequential":

        if check_for_tracing_mode():

            control_qbl = QuantumBool()

            def conjugator(case, control_qbl):
                mcx(case,
                    control_qbl,
                    ctrl_state=i)

            for i in range(len(case_function_list)):
                with conjugate(conjugator)(case, control_qbl):
                    with control(control_qbl):
                        case_function_list[i](operand)

            control_qbl.delete()

        else:
    
            for i in range(len(case_function_list)):
                with i == case:
                    case_function_list[i](operand)
        
    elif method == "parallel":

        if check_for_tracing_mode():
            raise Exception("Compile method {method} for switch-case structure not available in Jasp mode.")
        
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