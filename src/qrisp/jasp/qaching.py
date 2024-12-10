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
from qrisp.jasp import TracingQuantumSession, check_for_tracing_mode

def qache(*func, **kwargs):
    """
    This decorator allows you to mark a function as "reusable". Reusable here means
    that the jasp expression of this function will be cached and reused in the next
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
        from qrisp.jasp import qache
        
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
        jaspr = make_jaspr(outer_function)()
        print(time.time() - t0) # 2.0225703716278076
        
    Even though ``inner_function`` has been called 4 times, we only see a delay of 2 seconds.
    This is because the function has been called with two different quantum types, implying it
    has been traced twice and recalled from the cache twice. We take a look at the :ref:`jaspr`.
    
    >>> print(jaspr)
    let inner_function = { lambda ; a:QuantumCircuit b:QubitArray. let
        c:Qubit = get_qubit b 0
        d:QuantumCircuit = h a c
        e:Qubit = get_qubit b 1
        f:QuantumCircuit = cx d c e
        g:QuantumCircuit h:bool[] = measure f c
      in (g, h) } in
    let inner_function1 = { lambda ; i:QuantumCircuit j:QubitArray k:i32[] l:bool[]. let
        m:Qubit = get_qubit j 0
        n:QuantumCircuit = h i m
        o:Qubit = get_qubit j 1
        p:QuantumCircuit = cx n m o
        q:QuantumCircuit r:bool[] = measure p m
      in (q, r) } in
    { lambda ; s:QuantumCircuit. let
        t:QuantumCircuit u:QubitArray = create_qubits s 2
        v:QuantumCircuit w:QubitArray = create_qubits t 2
        x:QuantumCircuit y:bool[] = pjit[name=inner_function jaxpr=inner_function] v
          u
        z:QuantumCircuit ba:bool[] = pjit[name=inner_function jaxpr=inner_function1] x
          w 0 False
        bb:QuantumCircuit bc:bool[] = pjit[name=inner_function jaxpr=inner_function] z
          u
        bd:QuantumCircuit be:bool[] = pjit[
          name=inner_function
          jaxpr=inner_function1
        ] bb w 0 False
        bf:bool[] = and y ba
        bg:bool[] = and bf bc
        bh:bool[] = and bg be
      in (bd, bh) }

    As expected, we see three different function definitions:
    
    * The first one describes ``inner_function`` called with a :ref:`QuantumVariable`. For this kind of signature only the ``QubitArray`` is required.
    * The second one describes ``inner_function`` called with :ref:`QuantumFloat`. Additionally to the ``QubitArray``, the ``.exponent`` and ``.signed`` attribute are also passed to the function.
    * The third function definition is ``outer_function``, which calls the previously defined functions.

    """
    
    if len(kwargs):
        return lambda x : qache_helper(x, kwargs)
    else:
        return qache_helper(func[0], {})
    
    
    
def qache_helper(func, jax_kwargs):
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
    def ammended_function(abs_qc, *args, **kwargs):
        
        # Set the given AbstractQuantumCircuit as the 
        # one carried by the tracing QuantumSession
        qs = TracingQuantumSession.get_instance()
        qs.abs_qc = abs_qc
        
        # Execute the function
        res = func(*args, **kwargs)
        
        # Return the result and the result AbstractQuantumCircuit.
        return qs.abs_qc, res
    
    # Modify the name of the ammended function to reflect the input
    ammended_function.__name__ = func.__name__
    # Wrap in jax.jit
    ammended_function = jax.jit(ammended_function, **jax_kwargs)
    
    from qrisp.core.quantum_variable import QuantumVariable, flatten_qv, unflatten_qv
    
    
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
        abs_qs = TracingQuantumSession.get_instance()
        
        temp_qubit_cache = abs_qs.qubit_cache
        abs_qs.qubit_cache = {}
        
        # Excecute the function
        abs_qc_new, res = ammended_function(abs_qs.abs_qc, *args, **kwargs)
        
        abs_qs.qubit_cache = temp_qubit_cache
        eqn = jax._src.core.thread_local_state.trace_state.trace_stack.dynamic.jaxpr_stack[0].eqns[-1]
        # eqn.params["jaxpr"] = "="
        from qrisp.jasp import Jaspr
        
        
        eqn.params["jaxpr"] = jax.core.ClosedJaxpr(Jaspr.from_cache(eqn.params["jaxpr"].jaxpr), eqn.params["jaxpr"].consts)
        
        # print(type(eqn.params["jaxpr"].jaxpr))
        
        abs_qs.abs_qc = abs_qc_new
        
        # Update the QuantumVariable objects to their former tracers (happens in-place)
        for tup in flattened_qvs:
            unflatten_qv(*tup[::-1])
        
        # Return the result.
        return res
    
    return return_function
