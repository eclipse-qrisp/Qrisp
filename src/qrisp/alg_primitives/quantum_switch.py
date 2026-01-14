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

import warnings

import jax.numpy as jnp
import numpy as np

from qrisp.alg_primitives import demux
from qrisp.core import QuantumArray, QuantumVariable, cx, mcx, x
from qrisp.environments import (
    conjugate,
    control,
    custom_control,
    custom_inversion,
    invert,
)
from qrisp.jasp import check_for_tracing_mode, jrange, q_cond, q_fori_loop
from qrisp.qtypes import QuantumBool


def _invert_inpl_function(func):
    """Helper function to invert in-place functions."""

    def inverted_func(*args, **kwargs):
        with invert():
            return func(*args, **kwargs)

    return inverted_func


def quantum_switch(
    case, branches, *operands, method="auto", branch_amount=None, case_amount=None, inv=False, ctrl=None
):
    r"""
    Executes a switch - case statement distinguishing between a list of
    given in-place functions.

    More precisely, the qswitch applies the unitary $U_i$ to the operand in state $\ket{\psi}$ given that the case variable is in state $\ket{i}$, i.e.,

    .. math::

        \text{qswitch}\ket{i}_{\text{case}}\ket{\psi}_{\text{operand}} = \ket{i}_{\text{case}}U_i\ket{\psi}_{\text{operand}}

    Parameters
    ----------
    case : :ref:`QuantumFloat` or list[:ref:`Qubit`]
        The case variable specifying which case should be executed.
    branches : list[callable] or callable
        A list of functions, performing some in-place operation on ``*operands``, or
        a function ``branches(i, *operands)`` performing some in-place operation on ``*operands`` depending on a nonnegative integer index ``i`` specifying the case.
    operands : :ref:`QuantumVariable`
        The arguments on which the branches are applied.
    method : str, optional
        The compilation method. Available are ``sequential``, ``parallel``, ``tree`` and ``auto``.
        ``parallel`` is exponentially fast but requires more temporary qubits. ``tree`` uses `balanced binaray trees <https://arxiv.org/pdf/2407.17966v1>`_.
        The default is ``auto``.
    branch_amount : int, optional
        The number of cases. By default the number is inferred automatically:
        - When ``branches`` is a single function, the size of the ``case`` variable is used.
        - When ``branches`` is a list of functions, the length of that list is used instead.

    Examples
    --------

    First, we consider the case where ``branches`` is a **list of functions**:

    We create some sample functions:

    ::

        from qrisp import *

        def f0(x): x += 1
        def f1(x): inpl_mult(x, 3, treat_overflow = False)
        def f2(x): pass
        def f3(x): h(x[1])
        branches_list = [f0, f1, f2, f3]

    Create operand and case variable:

    ::

        operand = QuantumFloat(4)
        operand[:] = 1
        case = QuantumFloat(2)
        h(case)

    Execute switch - case function:

    >>> qswitch(operand, case, branches_list)

    Simulate:

    >>> print(multi_measurement([case, operand]))
    {(0, 2): 0.25, (1, 3): 0.25, (2, 1): 0.25, (3, 1): 0.125, (3, 3): 0.125}


    Second, we consider the case where ``branches`` is a **function**:

    ::

        def branches(i, qv):
            x(qv[i])

        operand = QuantumFloat(4)
        case = QuantumFloat(2)
        h(case)

        qswitch(operand, case, branches)

    Simulate:

    >>> print(multi_measurement([case, operand]))
    {(0, 1): 0.25, (1, 2): 0.25, (2, 4): 0.25, (3, 8): 0.25}

    """

    if is_function_mode := callable(branches):
        if branch_amount is None:
            case_size = len(case) if isinstance(case, list) else case.size
            branch_amount = 2**case_size
        xrange = jrange if check_for_tracing_mode() else range
        if inv:
            branches = _invert_inpl_function(branches)

    elif isinstance(branches, list):
        if branch_amount is None:
            branch_amount = len(branches)
        elif method == "sequential":
            raise TypeError(
                "Argument 'branch_amount' must be None when using the 'sequential' method and a list as a 'branches'"
            )
        if inv:
            branches = [_invert_inpl_function(func) for func in branches]

        xrange = range

    else:
        raise TypeError("Argument 'branches' must be a list or a callable(i, x)")

    method = "tree" if method == "auto" else method

    if method == "sequential":

        control_qbl = QuantumBool()

        for i in xrange(branch_amount):
            with conjugate(mcx)(case, control_qbl, ctrl_state=i):
                with control(control_qbl):
                    if ctrl is None:
                        if is_function_mode:
                            branches(i, *operands)
                        else:
                            branches[i](*operands)
                    else:
                        if is_function_mode:
                            with control(ctrl):
                                branches(i, *operands)
                        else:
                            with control(ctrl):
                                branches[i](*operands)

        control_qbl.delete()

    elif method == "parallel":

        if check_for_tracing_mode():
            raise NotImplementedError(
                f"Compile method {method} for switch-case structure not available in tracing mode."
            )

        if isinstance(case, list):
            raise NotImplementedError(
                "Compile method 'parallel' for switch-case structure not available when 'case' is a list of qubits."
            )
        
        if len(operands)>1:
            raise NotImplementedError(
                "Compile method 'parallel' for switch-case structure not available when more then one 'operands' are provided."
            )

        # Idea: Use demux function to move operand and enabling bool into QuantumArray
        # to execute cases in parallel.

        # This QuantumArray acts as an addressable QRAM via the demux function

        if branch_amount != 2**case.size:

            warnings.warn(
                "Warning: Additional qubit overhead because case amount is smaller than case QuantumVariable!"
            )

        enable = QuantumArray(qtype=QuantumBool(), shape=(2**case.size,))
        enable[0].flip()

        qa = QuantumArray(qtype=operands[0], shape=((2**case.size,)))

        with conjugate(demux)(operands[0], case, qa, parallelize_qc=True):
            with conjugate(demux)(enable[0], case, enable, parallelize_qc=True):
                for i in range(branch_amount):
                    with control(enable[i]):
                        if ctrl is None:
                            if is_function_mode:
                                branches(i, qa[i])
                            else:
                                branches[i](qa[i])
                        else:
                            if is_function_mode:
                                with control(ctrl):
                                    branches(i, qa[i])
                            else:
                                with control(ctrl):
                                    branches[i](qa[i])

        qa.delete()

        enable[0].flip()
        enable.delete()

    # Uses balanced binaray trees https://arxiv.org/pdf/2407.17966v1
    elif method == "tree":

        # Jasp mode
        if check_for_tracing_mode():
            xrange = jrange
            x_fori_loop = q_fori_loop
            x_cond = q_cond

            def bitwise_count_diff(a, b):
                return jnp.int32(jnp.bitwise_count(jnp.bitwise_xor(a, b)))

        # Normal mode
        else:
            xrange = range

            def x_fori_loop(lower, upper, body_fun, init_val):
                val = init_val
                for i in range(lower, upper):
                    val = body_fun(i, val)
                return val

            def x_cond(pred, true_fun, false_fun, *operands):
                if pred:
                    return true_fun(*operands)
                return false_fun(*operands)

            def bitwise_count_diff(a, b):
                return np.int32(np.bitwise_count(np.bitwise_xor(a, b)))

        n = len(case) if isinstance(case, list) else case.size

        def nor_x(t):
            x(t)

        def nor_cx(c, t):
            cx(c, t)

        def nor_mcx(c, t):
            mcx(c, t)

        def bounce(d: int, anc, ca, oper):
            # with control(anc[d - 2]):
            #    x(anc[d-1])
            if ctrl is None:
                x_cond(
                    d - 2 == -1,
                    lambda: nor_x(anc[d - 1]),
                    lambda: nor_cx(anc[d - 2], anc[d - 1]),
                )
            else:
                x_cond(
                    d - 2 == -1,
                    lambda: nor_cx(ctrl, anc[d - 1]),
                    lambda: nor_cx(anc[d - 2], anc[d - 1]),
                )

            with control(anc[d - 1]):
                x(anc[d])

            # with control(anc[d - 2]):
            #    with control(ca[n - 1 - d]):
            #        x(anc[d])
            if ctrl is None:
                x_cond(
                    d - 2 == -1,
                    lambda: nor_cx(ca[n - 1 - d], anc[d]),
                    lambda: nor_mcx([ca[n - 1 - d], anc[d - 2]], anc[d]),
                )
            else:
                x_cond(
                    d - 2 == -1,
                    lambda: nor_mcx([ca[n - 1 - d], ctrl], anc[d]),
                    lambda: nor_mcx([ca[n - 1 - d], anc[d - 2]], anc[d]),
                )

        def down(d: int, anc, ca, oper):
            x(ca[n - 1 - d])
            # with control(anc[d-1]):
            #    with control(ca[n - 1 - d]):
            #        x(anc[d])
            if ctrl is None:
                x_cond(
                    d - 1 == -1,
                    lambda: nor_cx(ca[n - 1 - d], anc[d]),
                    lambda: nor_mcx([ca[n - 1 - d], anc[d - 1]], anc[d]),
                )
            else:
                x_cond(
                    d - 1 == -1,
                    lambda: nor_mcx([ca[n - 1 - d], ctrl], anc[d]),
                    lambda: nor_mcx([ca[n - 1 - d], anc[d - 1]], anc[d]),
                )
            x(ca[n - 1 - d])

        def up(d: int, anc, ca, oper):
            # with control(anc[d-1]):
            #    with control(ca[n - 1 - d]):
            #        x(anc[d])
            if ctrl is None:
                x_cond(
                    d - 1 == -1,
                    lambda: nor_cx(ca[n - 1 - d], anc[d]),
                    lambda: nor_mcx([ca[n - 1 - d], anc[d - 1]], anc[d]),
                )
            else:
                x_cond(
                    d - 1 == -1,
                    lambda: nor_mcx([ca[n - 1 - d], ctrl], anc[d]),
                    lambda: nor_mcx([ca[n - 1 - d], anc[d - 1]], anc[d]),
                )

        # Function mode
        if is_function_mode:

            def leaf(d: int, anc, ca, i, *oper):
                with control(anc[d]):
                    branches(i, *oper)

                # with control(anc[d-1]):
                #    x(anc[d])
                if ctrl is None:
                    x_cond(
                        d - 1 == -1,
                        lambda: nor_x(anc[d]),
                        lambda: nor_cx(anc[d - 1], anc[d]),
                    )
                else:
                    x_cond(
                        d - 1 == -1,
                        lambda: nor_cx(ctrl, anc[d]),
                        lambda: nor_cx(anc[d - 1], anc[d]),
                    )

                with control(anc[d]):
                    branches(i + 1, *oper)

            def last_leaf(d: int, anc, ca, i, *oper):
                with control(anc[d]):
                    branches(i, *oper)

        # List mode
        elif isinstance(branches, list):

            if len(branches) % 2 != 0:

                def identity(_):
                    pass

                branches.append(identity)

            if check_for_tracing_mode():

                def leaf(d: int, anc, ca, i, *oper):
                    def apply_leaf(A, B):
                        with control(anc[d]):
                            A(*oper)

                        # with control(anc[d-1]):
                        #    x(anc[d])
                        if ctrl is None:
                            x_cond(
                                d - 1 == -1,
                                lambda: nor_x(anc[d]),
                                lambda: nor_cx(anc[d - 1], anc[d]),
                            )
                        else:
                            x_cond(
                                d - 1 == -1,
                                lambda: nor_cx(ctrl, anc[d]),
                                lambda: nor_cx(anc[d - 1], anc[d]),
                            )

                        with control(anc[d]):
                            B(*oper)

                    for j in range(0, len(branches), 2):
                        x_cond(
                            j == i,
                            apply_leaf,
                            lambda a, b: None,
                            branches[j],
                            branches[j + 1],
                        )

            else:

                def leaf(d: int, anc, ca, i, *oper):
                    with control(anc[d]):
                        branches[i](*oper)

                    # with control(anc[d-1]):
                    #    x(anc[d])
                    if ctrl is None:
                        x_cond(
                            d - 1 == -1,
                            lambda: nor_x(anc[d]),
                            lambda: nor_cx(anc[d - 1], anc[d]),
                        )
                    else:
                        x_cond(
                            d - 1 == -1,
                            lambda: nor_cx(ctrl, anc[d]),
                            lambda: nor_cx(anc[d - 1], anc[d]),
                        )

                    with control(anc[d]):
                        branches[i + 1](*oper)

            def last_leaf(d: int, anc, ca, i, *oper):
                def apply(f):
                    with control(anc[d]):
                        f(*oper)

                for j in range(0, len(branches)):
                    x_cond(j == i, apply, lambda x: None, branches[j])

        else:
            raise TypeError(
                "Argument 'branches' must be a list or a callable(i, x)"
            )

        def body_fun(pos, val):
            anc, ca, *oper = val

            # Apply leaf
            leaf(n - 1, anc, ca, 2 * pos, *oper)

            # Jump to next leaf
            q = bitwise_count_diff(pos, pos + 1)
            for j in xrange(0, q - 1, 1):
                up(n - j - 1, anc, ca, *oper)
            bounce(n - q, anc, ca, *oper)
            for j in xrange(0, q - 1, 1):
                down(n - (q - 1) + j, anc, ca, *oper)

            return anc, ca, *oper

        anc = QuantumVariable(n)
        # x(anc[0])

        # Go to first node
        for j in xrange(0, n, 1):
            down(j, anc, case, *operands)

        # Perform leafs and jumps

        _, _, _ = x_fori_loop(
            0, -(-branch_amount // 2) - 1, body_fun, (anc, case, *operands)
        )

        # Perfrom last leaf
        x_cond(
            branch_amount % 2 == 0,
            lambda: leaf(n - 1, anc, case, branch_amount - 2, *operands),
            lambda: last_leaf(n - 1, anc, case, branch_amount - 1, *operands),
        )

        # Go back from last node
        diff = 2**n - branch_amount
        for j in xrange(0, n, 1):
            up(n - j - 1, anc, case, *operands)

            def bf():
                # with control(anc[n-j-2]):
                #    x(anc[n - j-1])
                # return None
                if ctrl is None:
                    x_cond(
                        n - j - 2 == -1,
                        lambda: nor_x(anc[n - j - 1]),
                        lambda: nor_cx(anc[n - j - 2], anc[n - j - 1]),
                    )
                else:
                    x_cond(
                        n - j - 2 == -1,
                        lambda: nor_cx(ctrl, anc[n - j - 1]),
                        lambda: nor_cx(anc[n - j - 2], anc[n - j - 1]),
                    )

            # The x_cond applies:
            # if (diff >> j) & 1:
            #    with control(anc[n-j-1]):
            #        x(anc[n - j])
            x_cond((diff >> j) & 1, lambda: bf(), lambda: None)

        # x(anc[0])

        anc.delete()

    else:
        raise Exception(
            f"Don't know compile method {method} for switch-case structure."
        )


temp = quantum_switch.__doc__
qswitch = custom_control(custom_inversion(quantum_switch))
qswitch.__doc__ = temp
