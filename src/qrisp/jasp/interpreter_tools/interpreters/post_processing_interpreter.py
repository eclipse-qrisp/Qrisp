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

def extract_post_processing(jaspr, *args):
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
    
    Returns
    -------
    callable
        A function that takes a JAX array of boolean measurement results and returns 
        the post-processed results. The array should have shape (n,) where n is the 
        number of measurements in the original Jaspr.
    
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
        # Extract post-processing with the same arguments used for circuit extraction
        post_proc_func = extract_post_processing(jaspr, 1, 2)
        
        # Pass measurement results as a JAX array
        result = post_proc_func(jnp.array([False, True]))
        # Returns the same as the original function would have with those measurements
        
    """
    from jax.extend.core import ClosedJaxpr
    from qrisp.jasp.primitives import AbstractQuantumCircuit, QuantumPrimitive
    from qrisp.jasp.interpreter_tools.abstract_interpreter import eval_jaxpr_with_context_dic, ContextDict
    
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
    
    # Track measurements for indexing
    measurement_counter = [0]  # Use list so we can modify in closure
    
    # Custom equation evaluator
    def create_post_processing_evaluator(measurement_array):
        """Create an equation evaluator with closure over measurement_array."""
        measurement_counter[0] = 0  # Reset counter
        
        def eval_eqn(eqn, context_dic):
            # Replace measure with array indexing (check BEFORE QuantumPrimitive check)
            if eqn.primitive.name == "jasp.measure":
                result = measurement_array[measurement_counter[0]]
                measurement_counter[0] += 1
                
                if eqn.outvars:
                    context_dic[eqn.outvars[0]] = result
                return False
            
            # Handle create_qubits: store size in context as QubitArray representation
            if eqn.primitive.name == "jasp.create_qubits":
                # invars[0] is the size, invars[1] is the QuantumCircuit
                size = context_dic[eqn.invars[0]]
                # Store size as the QubitArray representation (we don't need actual qubits)
                context_dic[eqn.outvars[0]] = size
                # Skip QuantumCircuit outvar (eqn.outvars[1])
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
                array_size = context_dic[eqn.invars[0]]
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
            if isinstance(eqn.primitive, QuantumPrimitive):
                return False
            
            # For other primitives, use default evaluation
            return True
        
        return eval_eqn
    
    # Return function that evaluates with custom evaluator
    def post_processing_func(measurement_results):
        """
        Post-processing function that takes measurement results as a JAX array.
        
        Parameters
        ----------
        measurement_results : jax.Array
            A 1D array of boolean measurement results from the quantum circuit execution.
            Should have shape (n,) where n is the number of measurements.
        
        Returns
        -------
        tuple or single value
            The post-processed results.
        """
        # Create evaluator with closure over measurement_results
        eqn_evaluator = create_post_processing_evaluator(measurement_results)
        
        # Create initial context with static args (skip QuantumCircuit)
        context_dic = ContextDict()
        for var, val in static_value_map.items():
            context_dic[var] = val
        
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
