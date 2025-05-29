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

from qrisp.core import QuantumArray, QuantumVariable, x
from qrisp.qtypes import QuantumBool
from qrisp.environments import conjugate, control, custom_inversion, invert
from qrisp.alg_primitives import demux
from qrisp.core.gate_application_functions import mcx
from qrisp.jasp import check_for_tracing_mode, jrange, q_fori_loop, q_cond
import numpy as np
import jax.numpy as jnp


def qswitch(operand, case, case_function, method="auto", inv = False):
    """
    Executes a switch - case statement distinguishing between a list of
    given in-place functions.


    Parameters
    ----------
    operand : :ref:`QuantumVariable`
        The argument on which the case function operates.
    case : :ref:`QuantumFloat`
        The index specifying which case should be executed.
    case_function : list[callable] or callable
        A list of functions, performing some in-place operation on ``operand``, or
        a function ``case_function(i, operand)`` performing some in-place operation on ``operand`` depending on a nonnegative integer index ``i`` specifying the case.
    method : str, optional
        The compilation method. Available are ``sequential``, ``parallel``, ``tree`` and ``auto``.
        ``parallel`` is exponentially fast but requires more temporary qubits. ``tree`` uses `balanced binaray trees <https://arxiv.org/pdf/2407.17966v1>`_.
        The default is ``auto``.

    Examples
    --------

    First, we consider the case where ``case_function`` is a **list of functions**:

    We create some sample functions:

    ::

        from qrisp import *

        def f0(x): x += 1
        def f1(x): inpl_mult(x, 3, treat_overflow = False)
        def f2(x): pass
        def f3(x): h(x[1])
        case_function_list = [f0, f1, f2, f3]

    Create operand and case variable:

    ::

        operand = QuantumFloat(4)
        operand[:] = 1
        case = QuantumFloat(2)
        h(case)

    Execute switch - case function:

    >>> qswitch(operand, case, case_function_list)

    Simulate:

    >>> print(multi_measurement([case, operand]))
    {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}


    Second, we consider the case where ``case_function`` is a **function**:

    ::

        def case_function(i, qv):
            x(qv[i])

        operand = QuantumFloat(4)
        case = QuantumFloat(2)
        h(case)

        qswitch(operand, case, case_function)

    Simulate:

    >>> print(multi_measurement([case, operand]))
    {(0, 1): 0.25, (1, 2): 0.25, (2, 4): 0.25, (3, 8): 0.25}

    """
    
    def invert_inpl_function(func):
        def return_func(*args, **kwargs):
            with invert():
                res = func(*args, **kwargs)
            return res
        return return_func
    

    if callable(case_function):
        case_amount = 2**case.size
        xrange = jrange
        if method == "auto":
            method = "tree"
            
        if inv:
            case_function = invert_inpl_function(case_function)

    else:
        case_amount = len(case_function)

        # Extend case_function list by identity such that its size is 2*n (necessary for tree qswitch)
        def identity(operand):
            pass

        case_function.extend(
            [identity] * ((1 << ((case_amount - 1).bit_length())) - case_amount)
        )
        
        if inv:
            case_function = [invert_inpl_function(func) for func in case_function]
            

        xrange = range
        if method == "auto":
            if case_amount <= 4:
                method = "sequential"
            else:
                method = "tree"

    if method == "sequential":

        control_qbl = QuantumBool()

        for i in xrange(case_amount):
            with conjugate(mcx)(case, control_qbl, ctrl_state=i):
                with control(control_qbl):
                    if callable(case_function):
                        case_function(i, operand)
                    else:
                        case_function[i](operand)

        control_qbl.delete()

    elif method == "parallel":

        if check_for_tracing_mode():
            raise Exception(
                f"Compile method {method} for switch-case structure not available in tracing mode."
            )

        # Idea: Use demux function to move operand and enabling bool into QuantumArray
        # to execute cases in parallel.

        # This QuantumArray acts as an addressable QRAM via the demux function
        enable = QuantumArray(qtype=QuantumBool(), shape=(case_amount,))
        enable[0].flip()

        qa = QuantumArray(qtype=operand, shape=((case_amount,)))

        with conjugate(demux)(operand, case, qa, parallelize_qc=True):
            with conjugate(demux)(enable[0], case, enable, parallelize_qc=True):
                for i in range(case_amount):
                    with control(enable[i]):
                        if callable(case_function):
                            case_function(i, qa[i])
                        else:
                            case_function[i](qa[i])

        qa.delete()

        enable[0].flip()
        enable.delete()

    # Uses balanced binaray trees https://arxiv.org/pdf/2407.17966v1
    elif method == "tree":
        n = case.size

        def bounce(d: int, anc, ca, oper):
            with control(anc[d - 1]):
                x(anc[d])
            with control(anc[d]):
                x(anc[d + 1])
            with control(anc[d - 1]):
                with control(ca[n - 1 - d]):
                    x(anc[d + 1])

        def down(d: int, anc, ca, oper):
            with control(anc[d]):
                x(ca[n - 1 - d])
                with control(ca[n - 1 - d]):
                    x(anc[d + 1])
                x(ca[n - 1 - d])

        def up(d: int, anc, ca, oper):
            with control(anc[d]):
                with control(ca[n - 1 - d]):
                    x(anc[d + 1])

        # Jasp mode
        if check_for_tracing_mode():
            xrange = jrange
            x_fori_loop = q_fori_loop

            def bitwise_count_diff(a, b):
                return jnp.bitwise_count(jnp.bitwise_xor(a, b))

        # Normal mode
        else:
            xrange = range

            def x_fori_loop(lower, upper, body_fun, init_val):
                val = init_val
                for i in range(lower, upper):
                    val = body_fun(i, val)
                return val

            def bitwise_count_diff(a, b):
                return np.bitwise_count(np.bitwise_xor(a, b))

        # Function mode
        if callable(case_function):

            def leaf(d: int, anc, ca, oper, i):
                with control(anc[d + 1]):
                    case_function(i, oper)
                with control(anc[d]):
                    x(anc[d + 1])
                with control(anc[d + 1]):
                    case_function(i + 1, oper)

        # List mode
        elif isinstance(case_function, list):
            if check_for_tracing_mode():

                def leaf(d: int, anc, ca, oper, i):
                    def apply_leaf(A, B):
                        with control(anc[d + 1]):
                            A(oper)
                        with control(anc[d]):
                            x(anc[d + 1])
                        with control(anc[d + 1]):
                            B(oper)

                    for j in range(0, len(case_function), 2):
                        q_cond(
                            j == i,
                            apply_leaf,
                            lambda a, b: None,
                            case_function[j],
                            case_function[j + 1],
                        )

            else:

                def leaf(d: int, anc, ca, oper, i):
                    with control(anc[d + 1]):
                        case_function[i](oper)
                    with control(anc[d]):
                        x(anc[d + 1])
                    with control(anc[d + 1]):
                        case_function[i + 1](oper)

        else:
            raise TypeError(
                "Argument 'case_function' must be a list or a callable(i, x)"
            )

        def body_fun(pos, val):
            anc, ca, oper = val

            # Apply leaf
            leaf(n - 1, anc, ca, oper, 2 * pos)

            # Jump to next leaf
            q = bitwise_count_diff(pos, pos + 1)
            for j in xrange(0, q - 1, 1):
                up(n - j - 1, anc, ca, oper)
            bounce(n - q, anc, ca, oper)
            for j in xrange(0, q - 1, 1):
                down(n - (q - 1) + j, anc, ca, oper)

            return anc, ca, oper

        anc = QuantumVariable(n + 1)
        x(anc[0])

        # Go to first node
        for j in xrange(0, n, 1):
            down(j, anc, case, operand)

        # Perform leafs and jumps
        anc_, case, operand = x_fori_loop(
            0, 2 ** (n - 1) - 1, body_fun, (anc, case, operand)
        )

        # Perfrom last leaf
        leaf(n - 1, anc, case, operand, 2**n - 2)

        # Go back from last node
        for j in xrange(0, n, 1):
            up(n - j - 1, anc, case, operand)

        x(anc[0])
        anc.delete()

    else:
        raise Exception(
            f"Don't know compile method {method} for switch-case structure."
        )

temp = qswitch.__doc__
qswitch = custom_inversion(qswitch)
qswitch.__doc__ = temp
