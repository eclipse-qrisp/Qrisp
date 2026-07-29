"""Post-processing extraction for Jaspr objects.

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

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    ContextDict,
    eval_jaxpr,
    eval_jaxpr_with_context_dic,
    extract_invalues,
    insert_outvalues,
)
from qrisp.jasp.interpreter_tools.interpreters.control_flow_interpretation import (
    evaluate_cond_under_trace,
    evaluate_scan_under_trace,
    evaluate_while_loop_under_trace,
)


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def get_post_processing_evaluator(jaxpr):
    """Get a cached evaluator for a jaxpr.

    This function is LRU cached so that identical jaxprs return the same
    function instance. This is crucial for compilation efficiency: when the
    post-processing function is traced (e.g., for jax.jit), JAX will see the
    same function object for identical nested jaxprs and compile them only once.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The jaxpr to create an evaluator for.

    Returns
    -------
    callable
        A function that evaluates the jaxpr using the post-processing evaluator.

    """

    # We return a function that will use the post-processing evaluator
    # The evaluator itself is created inside extract_post_processing
    # This function will be called with the evaluator at runtime
    def cached_evaluator(eval_eqn):
        return eval_jaxpr(jaxpr, eqn_evaluator=eval_eqn)

    return cached_evaluator


def extract_post_processing(jaspr, *args):
    """Extracts the post-processing logic from a Jaspr object and returns a function
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
        the QuantumState argument). These will be bound into the post-processing
        function.

    Returns
    -------
    callable
        A function that takes measurement results and returns the post-processed results.
        Accepts either a string of '0' and '1' characters or a JAX array of booleans
        with shape (n,). String inputs are automatically converted to boolean arrays.

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

        # Extract post-processing function
        post_proc = extract_post_processing(jaspr, 1, 2)

        # Can use with bitstring input
        result = post_proc("01")

        # Or with array input (automatically detected, useful for JAX jitting)
        result = post_proc(jnp.array([False, True]))

    """
    import jax.numpy as jnp
    from jax.extend.core import ClosedJaxpr

    from qrisp.jasp.primitives import (
        AbstractQuantumState,
        AbstractQubitArray,
        QuantumPrimitive,
    )

    # Get the inner jaxpr
    if isinstance(jaspr.jaxpr, ClosedJaxpr):
        inner_jaxpr = jaspr.jaxpr.jaxpr
        consts = jaspr.jaxpr.consts
    else:
        inner_jaxpr = jaspr.jaxpr
        consts = []

    # Create static value map (excluding QuantumState)
    static_value_map = {}
    arg_index = 0
    for var in inner_jaxpr.invars:
        if not isinstance(var.aval, AbstractQuantumState):
            if arg_index < len(args):
                static_value_map[var] = args[arg_index]
                arg_index += 1

    # Custom equation evaluator
    def create_post_processing_evaluator():
        """Create an equation evaluator that consumes bits from a per-QuantumState
        tuple (measurement_array, last_popped_index) stored in the context.

        This avoids a global counter and mirrors the multi-value representation
        used in the catalyst interpreter: a single logical QuantumState value is
        represented by two runtime values.
        """
        from jax.lax import fori_loop

        def eval_eqn(eqn, context_dic):
            # Handle measurement: consume bits from the QuantumState tuple
            if eqn.primitive.name == "jasp.measure":
                # invars: [qubit_array, quantum_circuit]
                qc_var = eqn.invars[1]
                qc_tuple = context_dic[qc_var]
                meas_arr, last_popped = qc_tuple

                # Determine if we're measuring a QubitArray (multi-bit) or single qubit
                if isinstance(eqn.invars[0].aval, AbstractQubitArray):
                    # Size of the qubit array (may be dynamic!)
                    size = context_dic[eqn.invars[0]]

                    # Use JAX loop to build integer from bits
                    # Bits are in circuit order (LSB first for little-endian QuantumFloats)
                    # Loop body: accumulator = accumulator | ((bit ? 1 : 0) << i)
                    def loop_body(i, acc):
                        bit = meas_arr[last_popped + i]
                        # Convert bool to int: bit -> 1 if True else 0
                        bit_int = jnp.where(
                            bit,
                            jnp.array(1, dtype=jnp.int64),
                            jnp.array(0, dtype=jnp.int64),
                        )
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

                # Update the QuantumState tuple: output QC gets the updated tuple
                # (measurement consumes bits, so we advance the pointer)
                qc_outvar = eqn.outvars[1] if len(eqn.outvars) > 1 else None
                if qc_outvar:
                    context_dic[qc_outvar] = (meas_arr, new_last)
                return False

            # Handle create_qubits: store size in context as QubitArray representation
            if eqn.primitive.name == "jasp.create_qubits":
                # invars[0] is the size, invars[1] is the QuantumState
                size = context_dic[eqn.invars[0]]
                # Store size as the QubitArray representation (we don't need actual qubits)
                context_dic[eqn.outvars[0]] = size
                # Propagate the QuantumState tuple from input to output (unchanged)
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

            # Handle parity: compute XOR of measurement results
            if eqn.primitive.name == "jasp.parity":
                invalues = extract_invalues(eqn, context_dic)
                expectation = eqn.params.get("expectation", 0)

                # Compute parity (XOR) of all measurements
                result = sum(invalues) % 2

                # XOR result with expectation
                result = result ^ expectation

                # Insert result into context
                context_dic[eqn.outvars[0]] = jnp.array(result, dtype=bool)
                return False

            # Skip other quantum primitives entirely
            # We must:
            # 1. Propagate QuantumState tuples from input to output
            # 2. Store placeholder values for any non-QuantumState outputs (like Qubit)
            #    so that subsequent operations (like nested jit calls) can find them
            if isinstance(eqn.primitive, QuantumPrimitive):
                for outvar in eqn.outvars:
                    if isinstance(outvar.aval, AbstractQuantumState):
                        # Propagate QuantumState from input to output
                        context_dic[outvar] = context_dic[eqn.invars[-1]]
                    else:
                        # For non-QC outputs (Qubit, QubitArray, etc.), store None as placeholder
                        # These are only needed to satisfy nested jit calls which will also skip them
                        context_dic[outvar] = None
                return False

            # Intercept JAX control-flow / compilation primitives.
            # During tracing mode, we want to preserve jit calls (they get compiled later).
            # We use LRU caching to ensure identical jaxprs use the same function instance,
            # which allows JAX to compile them only once during tracing.
            if eqn.primitive.name in ("jit", "pjit"):
                closed_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
                if closed_jaxpr is None:
                    return False

                invalues = extract_invalues(eqn, context_dic)

                # Get the cached evaluator for this jaxpr
                # For identical jaxprs, this returns the same function instance
                cached_eval_func = get_post_processing_evaluator(closed_jaxpr.jaxpr)

                # Call it with the current evaluator, including the constants
                outvals = cached_eval_func(eval_eqn)(*(invalues + list(closed_jaxpr.consts)))

                # Handle wrapping based on the number of outputs in the inner jaxpr
                # Note: Our QuantumState representation is (meas_arr, last_popped) tuple,
                # which can confuse isinstance(outvals, tuple) checks. Use the jaxpr outvars count.
                if len(closed_jaxpr.jaxpr.outvars) == 1:
                    outvals = [outvals]

                insert_outvalues(eqn, context_dic, outvals)
                return False

            # Handle while loops
            if eqn.primitive.name == "while":
                evaluate_while_loop_under_trace(eqn, context_dic, eqn_evaluator=eval_eqn)
                return False

            # Handle conditional (cond/switch)
            if eqn.primitive.name == "cond":
                evaluate_cond_under_trace(eqn, context_dic, eqn_evaluator=eval_eqn)
                return False

            # Handle scan/map loops
            if eqn.primitive.name == "scan":
                evaluate_scan_under_trace(eqn, context_dic, eqn_evaluator=eval_eqn)
                return False

            # For other primitives, use default evaluation
            return True

        return eval_eqn

    # Return function that evaluates with custom evaluator
    def post_processing_func(measurement_results):
        """Post-processing function that takes measurement results.

        Parameters
        ----------
        measurement_results : str or jax.Array
            Either a string of '0' and '1' characters or a 1D array of boolean
            measurement results. String inputs are automatically converted to arrays.

        Returns
        -------
        tuple or single value
            The post-processed results.

        """
        # Convert bitstring to JAX array if it's a string
        if isinstance(measurement_results, str):
            # Convert string "01001..." to array [False, True, False, False, True, ...]
            measurement_results = jnp.array([c == "1" for c in measurement_results], dtype=bool)

        # Create evaluator
        eqn_evaluator = create_post_processing_evaluator()

        # Create initial context with static args
        context_dic = ContextDict()
        # jaxpr-level constants (e.g. array literals like scan's xs, hoisted out
        # of the traced function) are bound to inner_jaxpr.constvars positionally,
        # the same convention eval_jaxpr uses in abstract_interpreter.py.
        for var, val in zip(inner_jaxpr.constvars, consts):
            context_dic[var] = val
        for var, val in static_value_map.items():
            context_dic[var] = val

        # Initialize QuantumState variables as tuples (measurement_array, last_popped_index)
        # so that measurements can consume from the per-circuit array.
        for var in inner_jaxpr.invars:
            if isinstance(var.aval, AbstractQuantumState):
                # measurement_results is the full boolean array; start pointer at 0
                context_dic[var] = (measurement_results, 0)

        # Evaluate using the original jaxpr with custom evaluator
        eval_jaxpr_with_context_dic(inner_jaxpr, context_dic, eqn_evaluator)

        # Extract outputs (excluding QuantumState)
        outputs = []
        for var in inner_jaxpr.outvars:
            if not isinstance(var.aval, AbstractQuantumState):
                outputs.append(context_dic[var])

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return post_processing_func
