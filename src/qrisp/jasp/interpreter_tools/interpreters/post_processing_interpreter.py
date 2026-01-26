"""
Post-processing extraction for Jaspr objects.

This module provides functionality to extract the post-processing logic from a Jaspr object
and convert it into a function that can be applied to measurement results from a 
quantum backend.

The key function is `extract_post_processing(jaspr, *args)` which:
1. Takes the static argument values that were used for circuit extraction
2. Analyzes the Jaspr to identify all measurement operations
3. Uses a custom equation evaluator to skip quantum operations
4. Replaces measurement results with indexing into a JAX array
5. Binds static arguments in the evaluation context
6. Returns a callable that performs the classical post-processing

The returned function accepts a single JAX array of boolean measurement results.

Implementation uses the equation evaluator pattern (similar to qc_extraction_interpreter.py)
to avoid manually building slice/squeeze equations.
"""

def extract_post_processing(jaspr, *args, array_input=False):
    """
    Extracts the post-processing logic from a Jaspr object and returns a function
    that performs the post-processing.
    
    Uses a custom equation evaluator that intercepts measurement operations and
    replaces them with array indexing, while skipping quantum operations entirely.
    This avoids the complexity of manually building slice/squeeze equations.
    
    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr object to extract post-processing from. Must be completely flat
        (no jit calls, no conditionals, no loops).
    *args : 
        The static argument values that were used for circuit extraction (excluding 
        the QuantumCircuit argument). These will be bound into the post-processing
        function.
    array_input : bool, optional
        If True, the returned function will accept measurement results as a JAX array
        of booleans instead of a bitstring. Default is False (bitstring input).
    
    Returns
    -------
    callable
        A function that takes measurement results and returns the post-processed results.
        If array_input=False (default), accepts a string of '0' and '1' characters.
        If array_input=True, accepts a JAX array of booleans with shape (n,).
    
    Examples
    --------
    
    ::
    
        from qrisp import *
        from qrisp.jasp import make_jaspr
        import jax.numpy as jnp
        
        @make_jaspr
        def main(i, j):
            qv = QuantumFloat(5)
            return measure(qv[i]) + 1, measure(qv[1])
        
        jaspr = main(1, 2)
        
        # Bitstring input (default)
        post_proc = extract_post_processing(jaspr, 1, 2)
        result = post_proc("01")
        
        # Array input (for JAX jitting)
        post_proc_array = extract_post_processing(jaspr, 1, 2, array_input=True)
        result = post_proc_array(jnp.array([False, True]))
        
    """
    from jax.extend.core import ClosedJaxpr
    from qrisp.jasp.primitives import AbstractQuantumCircuit, QuantumPrimitive, AbstractQubitArray
    from qrisp.jasp.interpreter_tools.abstract_interpreter import eval_jaxpr_with_context_dic, ContextDict
    import jax.numpy as jnp
    
    # Get the inner jaxpr
    if isinstance(jaspr.jaxpr, ClosedJaxpr):
        inner_jaxpr = jaspr.jaxpr.jaxpr
        consts = jaspr.jaxpr.consts
    else:
        inner_jaxpr = jaspr.jaxpr
        consts = []
    
    # Create static value map (excluding QuantumCircuit)
    static_value_map = {}
    arg_index = 0
    for var in inner_jaxpr.invars:
        if not isinstance(var.aval, AbstractQuantumCircuit):
            if arg_index < len(args):
                static_value_map[var] = args[arg_index]
                arg_index += 1
    
    # Custom equation evaluator
    def create_post_processing_evaluator():
        """Create an equation evaluator that consumes bits from a per-QuantumCircuit
        tuple (measurement_array, last_popped_index) stored in the context.

        This avoids a global counter and mirrors the multi-value representation
        used in the catalyst interpreter: a single logical QuantumCircuit value is
        represented by two runtime values.
        """
        from jax.lax import fori_loop

        def eval_eqn(eqn, context_dic):
            # Handle measurement: consume bits from the QuantumCircuit tuple
            if eqn.primitive.name == "jasp.measure":
                # invars: [qubit_array, quantum_circuit]
                qc_var = eqn.invars[1]
                qc_tuple = context_dic[qc_var]
                meas_arr, last_popped = qc_tuple

                # Determine if we're measuring a QubitArray (multi-bit) or single qubit
                if isinstance(eqn.invars[0].aval, AbstractQubitArray):
                    # Size of the qubit array (may be dynamic!)
                    size = context_dic[eqn.invars[0]]
                    
                    # Use JAX loop to build integer from bits (handles dynamic size)
                    # Note: Bitstrings from quantum backends are typically big-endian
                    # (most significant bit first), but we build little-endian internally
                    # (bit 0 is LSB), so we need to reverse the bit order.
                    # Loop body: accumulator = accumulator | ((bit ? 1 : 0) << i)
                    def loop_body(i, acc):
                        # Reverse index: read from end of slice backwards
                        bit = meas_arr[last_popped + size - 1 - i]
                        # Convert bool to int: bit -> 1 if True else 0
                        bit_int = jnp.where(bit, jnp.array(1, dtype=jnp.int64), jnp.array(0, dtype=jnp.int64))
                        # Shift and OR into accumulator
                        acc = acc | (bit_int << i)
                        return acc
                    
                    result = fori_loop(0, size, loop_body, jnp.array(0, dtype=jnp.int64))
                    new_last = last_popped + size
                else:
                    # Single qubit measurement -> boolean (keep as bool for compatibility)
                    bit = meas_arr[last_popped]
                    result = bit  # Keep as boolean
                    new_last = last_popped + 1

                # Insert measurement result
                if eqn.outvars:
                    context_dic[eqn.outvars[0]] = result

                # Update the QuantumCircuit tuple: output QC gets the updated tuple
                # (measurement consumes bits, so we advance the pointer)
                qc_outvar = eqn.outvars[1] if len(eqn.outvars) > 1 else None
                if qc_outvar:
                    context_dic[qc_outvar] = (meas_arr, new_last)
                return False

            # Handle create_qubits: store size in context as QubitArray representation
            if eqn.primitive.name == "jasp.create_qubits":
                # invars[0] is the size, invars[1] is the QuantumCircuit
                size = context_dic[eqn.invars[0]]
                # Store size as the QubitArray representation (we don't need actual qubits)
                context_dic[eqn.outvars[0]] = size
                # Propagate the QuantumCircuit tuple from input to output (unchanged)
                qc_var_in = eqn.invars[1]
                qc_var_out = eqn.outvars[1]
                context_dic[qc_var_out] = context_dic[qc_var_in]
                return False

            # Handle get_size: return the stored size
            if eqn.primitive.name == "jasp.get_size":
                # invars[0] is the QubitArray (represented by its size)
                size = context_dic[eqn.invars[0]]
                context_dic[eqn.outvars[0]] = size
                return False

            # Handle slice: compute new size as (stop - start)
            if eqn.primitive.name == "jasp.slice":
                # invars[0] is the QubitArray (size), invars[1] is start, invars[2] is stop
                start = context_dic[eqn.invars[1]]
                stop = context_dic[eqn.invars[2]]
                # New size is the difference
                new_size = stop - start
                context_dic[eqn.outvars[0]] = new_size
                return False

            # Handle fuse: add the two sizes
            if eqn.primitive.name == "jasp.fuse":
                # invars[0] and invars[1] are the two arrays (represented by sizes)
                size1 = context_dic[eqn.invars[0]]
                size2 = context_dic[eqn.invars[1]]
                # Combined size is the sum
                combined_size = size1 + size2
                context_dic[eqn.outvars[0]] = combined_size
                return False

            # Skip other quantum primitives entirely
            # But we must propagate QuantumCircuit tuples from input to output!
            if isinstance(eqn.primitive, QuantumPrimitive):
                # Find any QuantumCircuit invars and propagate them to outvars
                if isinstance(eqn.invars[-1].aval, AbstractQuantumCircuit) and isinstance(eqn.outvars[-1].aval, AbstractQuantumCircuit):
                    context_dic[eqn.outvars[-1]] = context_dic[eqn.invars[-1]]
                return False

            # Intercept JAX control-flow / compilation primitives that would otherwise
            # trigger compilation (pjit/jit). Evaluate their contained jaxprs eagerly
            # using our same evaluator to avoid XLA compilation during post-processing.
            if eqn.primitive.name in ("jit", "pjit"):
                # Evaluate the nested jaxpr eagerly
                from qrisp.jasp.interpreter_tools.abstract_interpreter import (
                    eval_jaxpr,
                    extract_invalues,
                    insert_outvalues,
                )

                jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
                if jaxpr is None:
                    return False

                invalues = extract_invalues(eqn, context_dic)
                # eval_jaxpr returns a python-callable that uses our eval path
                outvals = eval_jaxpr(jaxpr, eqn_evaluator=eval_eqn)(*invalues)
                
                # eval_jaxpr returns a single value if there's one output, tuple otherwise
                # But insert_outvalues expects a tuple if eqn.primitive.multiple_results is True
                if not isinstance(outvals, tuple):
                    outvals = (outvals,)
                
                insert_outvalues(eqn, context_dic, outvals)
                return False

            # For other primitives, use default evaluation
            return True

        return eval_eqn
    
    # Return function that evaluates with custom evaluator
    def post_processing_func(measurement_results):
        """
        Post-processing function that takes measurement results.
        
        Parameters
        ----------
        measurement_results : str or jax.Array
            If array_input=False (default): A string of '0' and '1' characters.
            If array_input=True: A 1D array of boolean measurement results.
        
        Returns
        -------
        tuple or single value
            The post-processed results.
        """
        # Convert bitstring to JAX array if needed
        if not array_input:
            # Convert string "01001..." to array [False, True, False, False, True, ...]
            measurement_results = jnp.array([c == '1' for c in measurement_results], dtype=bool)
        
        # Create evaluator
        eqn_evaluator = create_post_processing_evaluator()

        # Create initial context with static args
        context_dic = ContextDict()
        for var, val in static_value_map.items():
            context_dic[var] = val

        # Initialize QuantumCircuit variables as tuples (measurement_array, last_popped_index)
        # so that measurements can consume from the per-circuit array.
        for var in inner_jaxpr.invars:
            if isinstance(var.aval, AbstractQuantumCircuit):
                # measurement_results is the full boolean array; start pointer at 0
                context_dic[var] = (measurement_results, 0)
        
        # Evaluate using the original jaxpr with custom evaluator
        eval_jaxpr_with_context_dic(inner_jaxpr, context_dic, eqn_evaluator)
        
        # Extract outputs (excluding QuantumCircuit)
        outputs = []
        for var in inner_jaxpr.outvars:
            if not isinstance(var.aval, AbstractQuantumCircuit):
                outputs.append(context_dic[var])
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    return post_processing_func
