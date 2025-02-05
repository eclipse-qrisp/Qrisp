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

from jax.lax import fori_loop, while_loop, cond

from qrisp.jasp.tracing_logic import TracingQuantumSession

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
    
    def new_body_fun(i, val):
        qs.start_tracing(val[0])
        res = body_fun(i, val[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    
    new_init_val = (abs_qc, init_val)
    fori_res = fori_loop(lower, upper, new_body_fun, new_init_val)
    
    qs.abs_qc = fori_res[0]
    return fori_res[1]
    
    
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
    
    def new_cond_fun(val):
        temp_qc = qs.abs_qc
        res = cond_fun(val[1])
        if not qs.abs_qc is temp_qc:
            raise Exception("Tried to modify quantum state during while condition evaluation")
        return res
    
    def new_body_fun(val):
        qs.start_tracing(val[0])
        res = body_fun(val[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    
    new_init_val = (abs_qc, init_val)
    while_res = while_loop(new_cond_fun, new_body_fun, new_init_val)
    
    qs.abs_qc = while_res[0]
    return while_res[1]

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

    def new_true_fun(*operands):
        qs.start_tracing(operands[0])
        res = true_fun(*operands[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    def new_false_fun(*operands):
        qs.start_tracing(operands[0])
        res = false_fun(*operands[1])
        abs_qc = qs.conclude_tracing()
        return (abs_qc, res)
    
    qs = TracingQuantumSession.get_instance()
    abs_qc = qs.abs_qc
    
    new_operands = (abs_qc, operands)
    
    cond_res = cond(pred, new_true_fun, new_false_fun, *new_operands)
        
    qs.abs_qc = cond_res[0]
    return cond_res[1]
