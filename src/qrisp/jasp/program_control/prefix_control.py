# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Jasp-compatible quantum control flow primitives: q_while_loop, q_fori_loop, q_cond, and q_switch."""

from jax.lax import cond, switch, while_loop

from qrisp.core import recursive_qv_search
from qrisp.jasp.primitives import AbstractQuantumState
from qrisp.jasp.tracing_logic import (
    TracingQuantumSession,
    check_for_tracing_mode,
    get_last_equation,
)


def q_while_loop(cond_fun, body_fun, init_val):
    """Jasp compatible version of
    `jax.lax.while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html#jax.lax.while_loop>`_.
    The parameters and semantics are the same as for the Jax version.

    In particular the following loop is performed

    ::

        def q_while_loop(cond_fun, body_fun, init_val):
            val = init_val
            while cond_fun(val):
                val = body_fun(val)
            return val

    Parameters
    ----------
    cond_fun : callable
        A function that evaluates the condition of the while loop. Must not
        contain any quantum operations.
    body_fun : callable
        A function describing the body of the loop.
    init_val : object
        An object to initialize the loop.

    Raises
    ------
    Exception
        Tried to modify quantum state during while condition evaluation.

    Returns
    -------
    val
        The result of ``body_fun`` after the last iteration.

    Examples
    --------
    We write a dynamic loop that collects measurement values of a quantum
    qubits into an accumulator. Note that the accumulator variable is a carry
    value implying the loop could not be implemented using :ref:`jrange`.


    ::

        from qrisp import *
        from qrisp.jasp import *

        @jaspify
        def main(k):

            qf = QuantumFloat(6)

            def body_fun(val):
                i, acc, qf = val
                x(qf[i])
                acc += measure(qf[i])
                i += 1
                return i, acc, qf

            def cond_fun(val):
                return val[0] < 5

            i, acc, qf = q_while_loop(cond_fun, body_fun, (0, 0, qf))

            return acc, measure(qf)

        print(main(6))
        # Yields
        # (Array(5, dtype=int64), Array(31., dtype=float64))

    """
    if not check_for_tracing_mode():
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    def new_cond_fun(val):
        temp_qc = qs.abs_qst
        res = cond_fun(val[0])
        if qs.abs_qst is not temp_qc:
            raise Exception("Tried to modify quantum state during while condition evaluation")
        return res

    def new_body_fun(val):
        qs.start_tracing(val[1])
        # The QuantumVariables from the arguments went through a flatten/unflattening cycle.
        # The unflattening creates a new QuantumVariable object, that is however not yet
        # registered in any QuantumSession. We register these in the current QuantumSession.
        for qv in recursive_qv_search(val[0]):
            qs.register_qv(qv, None)
        res = body_fun(val[0])
        abs_qst = qs.conclude_tracing()
        return (res, abs_qst)

    qs = TracingQuantumSession.get_instance()
    abs_qst = qs.abs_qst
    new_init_val = (init_val, abs_qst)
    while_res = while_loop(new_cond_fun, new_body_fun, new_init_val)

    eqn = get_last_equation()

    body_jaxpr = eqn.params["body_jaxpr"]

    # If the AbstractQuantumState is part of the constants of the body,
    # the body did not execute any quantum operations.
    # We remove the AbstractQuantumState from the body signature
    # to make the loop purely classical.
    for i in range(eqn.params["body_nconsts"]):
        if isinstance(body_jaxpr.jaxpr.invars[i].aval, AbstractQuantumState):
            eqn.invars.pop(i + eqn.params["cond_nconsts"])
            body_jaxpr.jaxpr.invars.pop(i)
            eqn.params["body_nconsts"] -= 1
            return while_res[0]

    from qrisp import Jaspr

    eqn.params["body_jaxpr"] = Jaspr.from_cache(body_jaxpr)

    qs.abs_qst = while_res[1]
    return while_res[0]


def q_fori_loop(lower, upper, body_fun, init_val):
    """Jasp compatible version of `jax.lax.fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop>`_.

    The parameters and semantics are the same as for the Jax version.

    In particular the following loop is performed

    ::

        def q_fori_loop(lower, upper, body_fun, init_val):
            val = init_val
            for i in range(lower, upper):
                val = body_fun(i, val)
            return val

    Parameters
    ----------
    lower : int or jax.core.Tracer
        An integer representing the loop index lower bound (inclusive).
    upper : int or jax.core.Tracer
        An integer representing the loop index upper bound (exclusive).
    body_fun : callable
        The function describing the loop body.
    init_val : object
        Some object to initialize the loop with.

    Returns
    -------
    val : object
        The return value of body_fun after the final iteration.

    Examples
    --------
    We write a dynamic loop that collects measurement values of a quantum
    qubits into an accumulator:

    ::

        from qrisp import *
        from qrisp.jasp import *

        @jaspify
        def main(k):

            qf = QuantumFloat(6)

            def body_fun(i, val):
                acc, qf = val
                x(qf[i])
                acc += measure(qf[i])
                return acc, qf

            acc, qf = q_fori_loop(0, k, body_fun, (0, qf))

            return acc, measure(qf)

        print(main(5))
        # Yields:
        # (Array(5, dtype=int64), Array(31., dtype=float64))

    """

    def new_body_fun(val):
        body_val = val[0]
        i = val[1]
        return (body_fun(i, body_val), i + 1, val[2])

    def new_cond_fun(val):
        i = val[1]
        upper = val[2]
        return i < upper

    return q_while_loop(new_cond_fun, new_body_fun, (init_val, lower, upper))[0]


def _wrap_branch_for_tracing(fn, qs):
    """Wrap a q_cond/q_switch branch function for tracing under jax.lax.cond/switch.

    Starts a nested tracing scope on the incoming AbstractQuantumState, registers
    every QuantumVariable found in the operands so it can be resolved during
    tracing, calls the branch, then concludes tracing and returns (result,
    updated_abs_qst) -- the two-tuple shape jax.lax.cond/switch's own branch
    functions must return.
    """

    def wrapped(*operands):
        qs.start_tracing(operands[1])
        for qv in recursive_qv_search(operands[0]):
            qs.register_qv(qv, None)
        res = fn(*operands[0])
        abs_qst = qs.conclude_tracing()
        return (res, abs_qst)

    return wrapped


def _get_last_cond_eqn():
    """Find the most recently traced 'cond' equation.

    JAX sometimes performs an automatic type conversion after a cond/switch
    call, so the cond equation isn't always the very last one traced -- this
    walks backwards until it finds it.
    """
    i = 1
    while True:
        eqn = get_last_equation(-i)
        if eqn.primitive.name == "cond":
            return eqn
        i += 1


def _finalize_branch_eqn(eqn, branch_jaxprs, res, qs, fun_name):
    """Validate and finalize a traced cond/switch equation's branch jaxprs.

    Raises if the AbstractQuantumState was implicitly imported (i.e. not part
    of the traced body's own signature). If none of the branches actually
    touch the quantum state, strips the now-unused trailing QuantumState
    invar/outvar from the equation and every branch and returns the plain
    result; otherwise writes back the (Jaspr-cached) branches, updates
    qs.abs_qst from the traced result, and returns the result.
    """
    from qrisp.jasp import Jaspr

    if not isinstance(branch_jaxprs[0].jaxpr.invars[-1].aval, AbstractQuantumState):
        raise Exception(
            f"Found implicit variable import in {fun_name}. "
            "Please make sure all used variables are part of the body signature."
        )

    if all(not isinstance(branch_jaxpr.jaxpr.outvars[-1].aval, AbstractQuantumState) for branch_jaxpr in branch_jaxprs):
        eqn.invars.pop(-1)
        for branch_jaxpr in branch_jaxprs:
            branch_jaxpr.jaxpr.invars.pop(-1)
        return res[0]

    eqn.params["branches"] = tuple(Jaspr.from_cache(branch_jaxpr) for branch_jaxpr in branch_jaxprs)

    qs.abs_qst = res[-1]

    return res[0]


def q_cond(pred, true_fun, false_fun, *operands):
    r"""Jasp compatible version of
    `jax.lax.cond <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html#jax.lax.cond>`_.
    The parameters and semantics are the same as for the Jax version.

    Performs the following semantics:

    ::

        def q_cond(pred, true_fun, false_fun, *operands):
            if pred:
                return true_fun(*operands)
            else:
                return false_fun(*operands)


    Parameters
    ----------
    pred : bool or jax.core.Tracer
        A boolean value, deciding which function gets executed.
    true_fun : callable
        The function that is executed when ``pred`` is True.
    false_fun : callable
        The function that is executed when ``pred`` is False.
    *operands : tuple
        The input values for both functions.

    Returns
    -------
    object
        The return value of the respective function.

    Examples
    --------
    We write a script that brings a :ref:`QuantumBool` into superpostion and
    subsequently measures it. If the measurement result is ``False`` we flip
    it such that in the end, the bool will always be in the $\ket{\text{True}}$
    state.

    ::

        from qrisp import *
        from qrisp.jasp import *

        @jaspify
        def main():

            def false_fun(qbl):
                qbl.flip()
                return qbl

            def true_fun(qbl):
                return qbl

            qbl = QuantumBool()
            h(qbl)
            pred = measure(qbl)

            qbl = q_cond(pred,
                         true_fun,
                         false_fun,
                         qbl)

            return measure(qbl)

        print(main())
        # Yields:
        # True

    """
    if not check_for_tracing_mode():
        if pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)

    qs = TracingQuantumSession.get_instance()
    abs_qst = qs.abs_qst

    new_operands = (operands, abs_qst)

    cond_res = cond(
        pred,
        _wrap_branch_for_tracing(true_fun, qs),
        _wrap_branch_for_tracing(false_fun, qs),
        *new_operands,
    )

    eqn = _get_last_cond_eqn()

    return _finalize_branch_eqn(eqn, eqn.params["branches"], cond_res, qs, "q_cond")


# Switch implementation for classical index
def _q_switch_c(index, branches, *operands):
    r"""Jasp compatible version of
    `jax.lax.switch <https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html>`_.
    The parameters and semantics are the same as for the Jax version.

    Performs the following semantics:

    ::

        def q_switch(index, branches, *operands):
            return branches[index](*operands)


    Parameters
    ----------
    index : int or jax.core.Tracer
        An integer value, deciding which function gets executed.
    branches : list[callable]
        List of functions to be executed based on ``index``.
    *operands : tuple
        The input values for whichever function is applied.

    Returns
    -------
    object
        The return value of the respective function.

    Examples
    --------
    We write a script that brings a :ref:`QuantumFloat` into superpostion and
    subsequently measures it. If the measurement result is ``k`` we add ``3-k``
    such that in the end, the float will always be in the $\ket{\text{3}}$
    state.

    ::

        from qrisp import *
        from qrisp.jasp import *
        import jax.numpy as jnp

        @jaspify
        def main():

            def f0(x): x += 3
            def f1(x): x += 2
            def f2(x): x += 1
            def f3(x): pass
            branches = [f0, f1, f2, f3]

            operand = QuantumFloat(2)
            h(operand)
            index = jnp.int32(measure(operand))

            q_switch(index, branches, operand)
            return measure(operand)

        print(main())
        # 3.0

    """
    if not check_for_tracing_mode():
        return branches[index](*operands)

    qs = TracingQuantumSession.get_instance()
    abs_qst = qs.abs_qst

    new_branches = [_wrap_branch_for_tracing(branch, qs) for branch in branches]

    new_operands = (operands, abs_qst)

    switch_res = switch(index, new_branches, *new_operands)

    eqn = _get_last_cond_eqn()

    return _finalize_branch_eqn(eqn, eqn.params["branches"], switch_res, qs, "q_switch")


def q_switch(index, branches, *operands, branch_amount=None, method="auto"):
    r"""**Classical index**

    Jasp compatible version of
    `jax.lax.switch <https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html>`_.
    The parameters and semantics are the same as for the Jax version.

    Performs the following semantics:

    ::

        def q_switch(index, branches, *operands):
            return branches[index](*operands)

    **Quantum index**

    Executes a quantum switch - case statement distinguishing between given
    in-place functions.

    Implements the operation

    .. math::

        \text{SELECT} = \sum_i \ket{i}\bra{i} \otimes U_i

    for unitaries (branches) $U_i$,
    applying the $i$-th unitary conditioned on the index variable being in state $\ket{i}$.

    Parameters
    ----------
    index : int or jax.core.Tracer or QuantumVariable or list[Qubit]
        An integer value, deciding which function gets executed.
    branches : list[callable] or callable
        List of functions to be executed based on ``index`` or a single function
        that takes the index as first argument.
    *operands : tuple
        The input values for whichever function is applied.
    branch_amount : int, optional
        The amount of branches.
        Only needed if ``index`` is a :ref:`QuantumVariable` and ``branches`` is a function.
        Is automatically inferred from the length of ``branches`` if it is a list.
    method : str, optional
        Only needed if ``index`` is a :ref:`QuantumVariable`.
        The method used to implement the quantum switch. Can be ``"auto"``, ``"sequential"``, ``"parallel"``,
        or ``"tree"``. Default is ``"auto"``.
        Method ``"tree"`` uses `balanced binary trees <https://arxiv.org/pdf/2407.17966v1>`_.
        Method ``"parallel"`` is exponentially faster but requires more qubits.

    Returns
    -------
    object
        The return value of the respective function.

    Examples
    --------
    **Classical index**

    We write a script that brings a :ref:`QuantumFloat` into superpostion and
    subsequently measures it. If the measurement result is ``k`` we add ``3-k``
    such that in the end, the float will always be in the $\ket{\text{3}}$
    state.

    ::

        from qrisp import *
        from qrisp.jasp import *
        import jax.numpy as jnp

        @jaspify
        def main():

            def f0(x): x += 3
            def f1(x): x += 2
            def f2(x): x += 1
            def f3(x): pass
            branches = [f0, f1, f2, f3]

            operand = QuantumFloat(2)
            h(operand)
            index = jnp.int32(measure(operand))

            q_switch(index, branches, operand)
            return measure(operand)

        print(main())
        # 3.0

    **Quantum index**

    We write a script that uses a :ref:`QuantumFloat` as index to select
    different operations on another operand :ref:`QuantumFloat`. The index variable is
    put into superposition such that all branches are executed in superposition.

    ::

        from qrisp import *
        from qrisp.jasp import *

        @terminal_sampling
        def main():

            def f0(x): x += 1
            def f1(x): x += 2
            def f2(x): pass
            def f3(x): h(x[1])
            branches = [f0, f1, f2, f3]

            operand = QuantumFloat(4)
            operand[:] = 1
            index = QuantumFloat(2)
            h(index)

            q_switch(index, branches, operand)
            return index, operand

        print(main())
        # {(0.0, 2.0): 0.25000000372529035, (1.0, 3.0): 0.25000000372529035,
        # (2.0, 1.0): 0.25000000372529035, (3.0, 1.0): 0.12499999441206447,
        # (3.0, 3.0): 0.12499999441206447}

    """
    from qrisp.alg_primitives.program_control.quantum_switch import _q_switch_q
    from qrisp.circuit import Qubit
    from qrisp.core import QuantumVariable
    from qrisp.jasp.tracing_logic import DynamicQubitArray

    if (
        isinstance(index, QuantumVariable)
        or (isinstance(index, list) and all(isinstance(q, Qubit) for q in index))
        or isinstance(index, DynamicQubitArray)
    ):
        return _q_switch_q(index, branches, *operands, branch_amount=branch_amount, method=method)

    if callable(branches):
        return branches(index, *operands)

    return _q_switch_c(index, branches, *operands)
