"""
Post-processing extraction for Jaspr objects.

This module provides functionality to extract the post-processing logic from a Jaspr object
and convert it into a function that can be applied to measurement results from a 
quantum backend.

The key function is `extract_post_processing(jaspr, *args)` which:
1. Takes the static argument values that were used for circuit extraction
2. Analyzes the Jaspr to identify all measurement operations
3. Removes quantum operations (measure, create_qubits, reset, etc.)
4. Replaces measurement results with function arguments
5. Binds static arguments as Literals in the Jaxpr
6. Returns a callable that performs the classical post-processing
"""

def extract_post_processing(jaspr, *args):
    """
    Extracts the post-processing logic from a Jaspr object and returns a function
    that performs the post-processing.
    
    This function iterates through the Jaxpr equations, removes all measure primitives,
    and replaces their outputs with new invars (boolean arguments). The static argument
    values are bound directly into the post-processing function since they must be known
    at circuit extraction time anyway.
    
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
        A function that takes n boolean arguments (measurement results) and returns 
        the post-processed results.
    
    Examples
    --------
    
        from qrisp import *
        from qrisp.jasp import make_jaspr
        
        @make_jaspr
        def main(i, j):
            qv = QuantumFloat(5)
            return measure(qv[i]) + 1, measure(qv[1])
        
        jaspr = main(1, 2)
        # Extract post-processing with the same arguments used for circuit extraction
        post_proc_func = extract_post_processing(jaspr, 1, 2)
        
        # Now post_proc_func(False, True) will return the same as
        # the original function would have returned with measurements
        # yielding False and True respectively
    """
    from jax.extend.core import Var, Jaxpr, ClosedJaxpr, JaxprEqn, Literal
    from qrisp.jasp import eval_jaxpr
    from qrisp.jasp.primitives import AbstractQuantumCircuit, QuantumPrimitive
    
    # Get the inner jaxpr
    if isinstance(jaspr, ClosedJaxpr):
        inner_jaxpr = jaspr.jaxpr
        consts = jaspr.consts
    else:
        inner_jaxpr = jaspr
        consts = []
    
    # Create a context dictionary to bind the static argument values
    # Map original invars (excluding QuantumCircuit) to their static values
    static_value_map = {}
    arg_index = 0
    for var in inner_jaxpr.invars:
        if not isinstance(var.aval, AbstractQuantumCircuit):
            if arg_index < len(args):
                static_value_map[var] = args[arg_index]
                arg_index += 1
    
    # We'll build a new jaxpr without measure primitives
    # and with new invars for the measurement results
    new_invars = []
    new_eqns = []
    
    # Track measurement outputs and their corresponding new invars
    measure_replacement_map = {}
    measure_invars = []  # List of new boolean invars for measurements
    
    # First pass: identify all measure primitives and create new invars
    for eqn in inner_jaxpr.eqns:
        if eqn.primitive.name == "jasp.measure":
            # Create new invar for this measurement result
            # The first outvar is the measurement result (bool or int64)
            meas_result_var = eqn.outvars[0]
            new_invar = Var(aval=meas_result_var.aval)
            measure_invars.append(new_invar)
            measure_replacement_map[meas_result_var] = new_invar
            
            # The second outvar is the quantum circuit, we need to track it
            if len(eqn.outvars) > 1:
                # We need to map the output quantum circuit to the input quantum circuit
                qc_out_var = eqn.outvars[1]
                qc_in_var = eqn.invars[-1]  # Last invar should be the quantum circuit
                measure_replacement_map[qc_out_var] = qc_in_var
    
    # Build new invars list: only measurement invars
    # Original arguments are bound as static values
    new_invars = list(measure_invars)
    
    # Second pass: rebuild equations, replacing measure primitives and updating var references
    def replace_var(var):
        """Replace a var with its mapped replacement if it exists."""
        # Literals are not hashable, so we can't use them as dict keys
        if isinstance(var, Literal):
            return var
        # Check if this var should be replaced by a measurement result
        if var in measure_replacement_map:
            return measure_replacement_map[var]
        # Check if this var should be replaced by a static value (convert to Literal)
        if var in static_value_map:
            return Literal(static_value_map[var], var.aval)
        return var
    
    for eqn in inner_jaxpr.eqns:
        # Skip measure primitives
        if eqn.primitive.name == "jasp.measure":
            continue
        
        # Skip quantum circuit operations (create_qubits, reset, delete_qubits, gates, etc.)
        # These are not needed for post-processing
        if isinstance(eqn.primitive, QuantumPrimitive):
            # Map any quantum circuit outputs to their inputs
            for i, outvar in enumerate(eqn.outvars):
                if isinstance(outvar.aval, AbstractQuantumCircuit):
                    # Find the corresponding input quantum circuit
                    for invar in eqn.invars:
                        if isinstance(invar.aval, AbstractQuantumCircuit):
                            measure_replacement_map[outvar] = replace_var(invar)
                            break
            continue
        
        # For non-quantum primitives, update invars and outvars with replacements
        new_eqn_invars = [replace_var(v) for v in eqn.invars]
        new_eqn_outvars = list(eqn.outvars)
        
        new_eqn = JaxprEqn(
            primitive=eqn.primitive,
            invars=new_eqn_invars,
            outvars=new_eqn_outvars,
            params=dict(eqn.params),
            source_info=eqn.source_info,
            effects=eqn.effects,
            ctx=eqn.ctx,
        )
        new_eqns.append(new_eqn)
    
    # Build new outvars list (excluding QuantumCircuit)
    new_outvars = []
    for var in inner_jaxpr.outvars:
        if not isinstance(var.aval, AbstractQuantumCircuit):
            new_outvars.append(replace_var(var))
    
    # Create the new jaxpr
    new_jaxpr = Jaxpr(
        constvars=inner_jaxpr.constvars,
        invars=new_invars,
        outvars=new_outvars,
        eqns=new_eqns,
        effects=inner_jaxpr.effects,
        debug_info=inner_jaxpr.debug_info,
    )
    
    # Create a closed jaxpr
    new_closed_jaxpr = ClosedJaxpr(new_jaxpr, consts)
    
    # Return a function that evaluates this jaxpr
    def post_processing_func(*measurement_results):
        """
        Post-processing function that takes measurement results as booleans.
        
        Parameters
        ----------
        *measurement_results : bool or int
            The measurement results from the quantum circuit execution.
        
        Returns
        -------
        tuple or single value
            The post-processed results.
        """
        # Evaluate the new jaxpr with the measurement results
        result = eval_jaxpr(new_closed_jaxpr)(*measurement_results)
        return result
    
    return post_processing_func
