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

from jax.lax import fori_loop, while_loop, cond
from jax.extend.core import ClosedJaxpr
import jax

from qrisp.core import recursive_qv_search, recursive_qa_search
from qrisp.jasp.tracing_logic import TracingQuantumSession, check_for_tracing_mode, get_last_equation
from qrisp.jasp.primitives import AbstractQuantumCircuit


def q_while_loop(cond_fun, body_fun, init_val):
    """
    Jasp compatible version of
    `jax.lax.while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html#jax.lax.while_loop>`_
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
        temp_qc = qs.abs_qc
        res = cond_fun(val[0])
        if not qs.abs_qc is temp_qc:
            raise Exception(
                "Tried to modify quantum state during while condition evaluation"
            )
        return res

    def new_body_fun(val):
        qs.start_tracing(val[1])
        # The QuantumVariables from the arguments went through a flatten/unflattening cycle.
        # The unflattening creates a new QuantumVariable object, that is however not yet
        # registered in any QuantumSession. We register these in the current QuantumSession.
        for qv in recursive_qv_search(val[0]):
            qs.register_qv(qv, None)
        res = body_fun(val[0])
        abs_qc = qs.conclude_tracing()
        return (res, abs_qc)

    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    new_init_val = (init_val, abs_qc)
    while_res = while_loop(new_cond_fun, new_body_fun, new_init_val)

    eqn = get_last_equation()
    
    body_jaxpr = eqn.params["body_jaxpr"]
    
    # If the AbstractQuantumCircuit is part of the constants of the body,
    # the body did not execute any quantum operations.
    # We remove the AbstractQuantumCircuit from the body signature
    # to make the loop purely classical.
    for i in range(eqn.params["body_nconsts"]):
        if isinstance(body_jaxpr.jaxpr.invars[i].aval, AbstractQuantumCircuit):
            eqn.invars.pop(i + eqn.params["cond_nconsts"])
            body_jaxpr.jaxpr.invars.pop(i)
            eqn.params["body_nconsts"] -= 1
            return while_res[0]
        
    from qrisp import Jaspr

    eqn.params["body_jaxpr"] = Jaspr.from_cache(body_jaxpr)

    qs.abs_qc = while_res[1]
    return while_res[0]


def q_fori_loop(lower, upper, body_fun, init_val):
    """
    Jasp compatible version of
    `jax.lax.fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop>`_
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

        print(main(k))
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


def q_cond(pred, true_fun, false_fun, *operands):
    r"""
    Jasp compatible version of
    `jax.lax.cond <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html#jax.lax.cond>`_
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

    def new_true_fun(*operands):
        qs.start_tracing(operands[1])
        for qv in recursive_qv_search(operands[0]):
            qs.register_qv(qv, None)
        res = true_fun(*operands[0])
        abs_qc = qs.conclude_tracing()
        return (res, abs_qc)

    def new_false_fun(*operands):
        qs.start_tracing(operands[1])
        for qv in recursive_qv_search(operands[0]):
            qs.register_qv(qv, None)
        res = false_fun(*operands[0])
        abs_qc = qs.conclude_tracing()
        return (res, abs_qc)

    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc

    new_operands = (operands, abs_qc)

    cond_res = cond(pred, new_true_fun, new_false_fun, *new_operands)

    # There seem to be situations, where Jax performs some automatic type
    # conversion after the cond call. This results in the cond equation
    # not being the most recent equation.
    # We therefore search for the last cond primitive.
    i = 1
    while True:
        eqn = get_last_equation(-i)
        if eqn.primitive.name == "cond":
            break
        i += 1
        
    false_jaxpr = eqn.params["branches"][0]
    true_jaxpr = eqn.params["branches"][1]

    if not isinstance(false_jaxpr.jaxpr.invars[-1].aval, AbstractQuantumCircuit):
        raise Exception(
            "Found implicit variable import in q_cond. Please make sure all used variables are part of the body signature."
        )

    from qrisp.jasp import Jaspr

    if (not isinstance(false_jaxpr.jaxpr.outvars[-1].aval, AbstractQuantumCircuit)) and (not isinstance(true_jaxpr.jaxpr.outvars[-1].aval, AbstractQuantumCircuit)):
        eqn.invars.pop(-1)
        false_jaxpr.jaxpr.invars.pop(-1)
        true_jaxpr.jaxpr.invars.pop(-1)
        return cond_res[0]
    
    eqn.params["branches"] = (
            Jaspr.from_cache(false_jaxpr),
            Jaspr.from_cache(true_jaxpr)
    )

    qs.abs_qc = cond_res[-1]

    return cond_res[0]
