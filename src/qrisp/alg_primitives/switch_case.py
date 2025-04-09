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

from qrisp.core.gate_application_functions import h, x
from qrisp.alg_primitives.qft import QFT
from qrisp.core import QuantumArray
from qrisp.qtypes import QuantumBool, QuantumVariable
from qrisp.environments import conjugate, control
from qrisp.alg_primitives import demux

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
        The compilation method. Available are ``parallel``, ``sequential``, and ``tree``. 
        ``parallel`` is exponentially fast but requires more temporary qubits.
        ``tree`` is based on the balanced binary tree in
        <https://arxiv.org/pdf/2407.17966v1>. The default is "sequential".

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

    elif method == "tree":
        n = case.size
        cfl = case_function_list.copy()
        anc = QuantumVariable(n+1)
        x(anc[0])

        def bounce(d: int):
            with control(anc[d-1]):
                x(anc[d])
            with control(anc[d]):
                x(anc[d+1])
            with control(anc[d-1]):
                with control(case[n-1-d]):
                    x(anc[d+1])

        def down(d: int):
            with control(anc[d]):
                x(case[n-1-d])
                with control(case[n-1-d]):
                    x(anc[d+1])
                x(case[n-1-d])

        def up(d: int):
            with control(anc[d]):
                with control(case[n-1-d]):
                    x(anc[d+1])

        def leaf(d: int, A, B):
            with control(anc[d+1]):
                A(operand)
            with control(anc[d]):
                x(anc[d+1])
            with control(anc[d+1]):
                B(operand)

        down(0)
        stack = [(down, 1), (bounce, 1), (up, 1)]
        while stack:
            op, depth = stack.pop(0)
            op(depth)
            if op == down or op == bounce:
                bounce_add = 0
                if depth + 1 + bounce_add >= n:
                    A = cfl.pop(0)
                    B = cfl.pop(0)
                    stack.insert(0, (lambda x: leaf(x, A, B), depth + bounce_add))
                else:
                    stack.insert(0, (up, depth + 1))
                    stack.insert(0, (bounce, depth + 1))
                    stack.insert(0, (down, depth + 1))
        up(0)
        anc.delete()
    
    else:
        raise Exception("Don't know compile method {method} for switch-case structure.")