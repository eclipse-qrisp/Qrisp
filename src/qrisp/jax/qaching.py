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

import jax
from qrisp.jax import get_tracing_qs, check_for_tracing_mode

def qache(func):
    """
    This decorator allows you to mark a function as "reusable". Reusable here means
    that the Jisp expression of this function will be cached and reused in the next
    calls (if the function is called with the same signature).
    
    A qached function therefore has to be traced by the Python interpreter only once
    and after that the function can be called without any Python-interpreter induced
    delay. This can significantly speed up the compilation process.
    
    Using the ``qache`` decorator not only improves the compilation speed but also
    enables the compiler to speed up transformation processes. 

    Parameters
    ----------
    func : callable
        The function to be qached.

    Returns
    -------
    qached_function : callable
        A function that will be traced on it's first execution and retrieved from
        the cache in any other call.
        
    Examples
    --------
    
    We create a simple function that is qached. To simulate an expensive compilation
    task we insert a ``time.sleep`` command.
    
    ::
        
        import time
        from qrisp import *
        from qrisp.jax import qache
        
        @qache
        def inner_function(qv):
            h(qv[0])
            cx(qv[0], qv[1])
            res_bl = measure(qv[0])
            
            # Simulate demanding compilation procedure by calling
            time.sleep(1)
            
            return res_bl
        
        def outer_function():
            a = QuantumVariable(2)
            b = QuantumFloat(2)
            
            bl_0 = inner_function(a)
            bl_1 = inner_function(b)
            bl_2 = inner_function(a)
            bl_3 = inner_function(b)
            
            return bl_0 & bl_1 & bl_2 & bl_3
        
        # Measure the time required for tracing
        t0 = time.time()
        jaxpr = make_jaxpr(outer_function)().jaxpr
        print(time.time() - t0) # 2.0225703716278076
        
    Even though ``inner_function`` has been called 4 times, we only see a delay of 2 seconds.
    This is because the function has been called with two different quantum types, implying it
    has been traced twice and recalled from the cache twice. We take a look at the jaxpr.
    
    >>> print(jaxpr)
    let inner_function = { lambda ; a:QuantumCircuit b:QubitArray. let
        c:Qubit = get_qubit b 0
        d:QuantumCircuit = h a c
        e:Qubit = get_qubit b 0
        f:Qubit = get_qubit b 1
        g:QuantumCircuit = cx d e f
        h:Qubit = get_qubit b 0
        i:QuantumCircuit j:bool[] = measure g h
      in (i, j) } in
    let inner_function1 = { lambda ; k:QuantumCircuit l:QubitArray m:i32[] n:bool[]. let
        o:Qubit = get_qubit l 0
        p:QuantumCircuit = h k o
        q:Qubit = get_qubit l 0
        r:Qubit = get_qubit l 1
        s:QuantumCircuit = cx p q r
        t:Qubit = get_qubit l 0
        u:QuantumCircuit v:bool[] = measure s t
      in (u, v) } in
    { lambda ; . let
        w:QuantumCircuit = qdef 
        x:QuantumCircuit y:QubitArray = create_qubits w 2
        z:QuantumCircuit ba:QubitArray = create_qubits x 2
        bb:QuantumCircuit bc:bool[] = pjit[name=inner_function jaxpr=inner_function] z
          y
        bd:QuantumCircuit be:bool[] = pjit[
          name=inner_function
          jaxpr=inner_function1
        ] bb ba 0 False
        bf:QuantumCircuit bg:bool[] = pjit[name=inner_function jaxpr=inner_function] bd
          y
        _:QuantumCircuit bh:bool[] = pjit[name=inner_function jaxpr=inner_function1] bf
          ba 0 False
        bi:bool[] = and bc be
        bj:bool[] = and bi bg
        bk:bool[] = and bj bh
      in (bk,) }

    """
    
    # 
    # To achieve the desired behavior we leverage the Jax inbuild caching mechanism.
    # This feature can be used by calling a jitted function in a tracing context.
    # To cache the function we therefore simply need to wrap it with jit and
    # it will be properly cached.
    
    # There are however some more things to consider.
    
    # The Qrisp function doesn't have the AbstractQuantumCircuit object (which is carried by 
    # the tracing QuantumSession) in the signature.
    
    # To make jax properly treat this, we modify the function signature
    
    # This function performs the input function but also has the AbstractQuantumCircuit
    # in the signature.
    def ammended_function(abs_qc, *args):
        
        # Set the given AbstractQuantumCircuit as the 
        # one carried by the tracing QuantumSession
        qs = get_tracing_qs()
        qs.abs_qc = abs_qc
        
        # Execute the function
        res = func(*args)
        
        # Return the result and the result AbstractQuantumCircuit.
        return qs.abs_qc, res
    
    # Modify the name of the ammended function to reflect the input
    ammended_function.__name__ = func.__name__
    # Wrap in jax.jit
    ammended_function = jax.jit(ammended_function)
    
    from qrisp.core.quantum_variable import QuantumVariable, flatten_qv, unflatten_qv
    from qrisp.jax import Jispr
    
    
    # We now prepare the return function
    def return_function(*args, **kwargs):
        
        # If we are not in tracing mode, simply execute the function
        if not check_for_tracing_mode():
            return func(*args, **kwargs)
        
        
        # Calling the jitted function on a QuantumVariable will call the 
        # flatten/unflatten procedure. This will set the traced attributes to the
        # tracers of the jit trace. To reverse this, we store the current tracers
        # by flattening each QuantumVariable in the signature.
        flattened_qvs = []
        for arg in args:
            if isinstance(arg, QuantumVariable):
                flattened_qvs.append(flatten_qv(arg))
        
        # Get the AbstractQuantumCircuit for tracing
        abs_qs = get_tracing_qs()
        
        # Excecute the function
        abs_qc_new, res = ammended_function(abs_qs.abs_qc, *args, **kwargs)
        
        # eqn = jax._src.core.thread_local_state.trace_state.trace_stack.dynamic.jaxpr_stack[0].eqns[-1]
        # eqn.params["jaxpr"] = "="
        
        abs_qs.abs_qc = abs_qc_new
        
        # Update the QuantumVariable objects to their former tracers (happens in-place)
        for tup in flattened_qvs:
            unflatten_qv(*tup[::-1])
        
        # Return the result.
        return res
    
    return return_function
