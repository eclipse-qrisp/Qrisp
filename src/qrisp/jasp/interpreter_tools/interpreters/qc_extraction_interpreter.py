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

QuantumCircuit Extraction Interpreter
=====================================

This module implements the interpreter for converting Jaspr (JAX-based quantum 
intermediate representation) into static QuantumCircuit objects.

Overview
--------
Jaspr represents hybrid quantum-classical algorithms using JAX's tracing 
infrastructure. While this representation is powerful for optimization and 
compilation, it needs to be "lowered" to a QuantumCircuit for execution on 
current quantum hardware or simulators that expect circuit-based input.

The Challenge
-------------
The main challenge in this conversion is handling **measurement results**:

1. **In Jaspr**: Measurement results are represented as JAX booleans.
   These can be manipulated using standard JAX/NumPy operations like array
   construction, indexing, arithmetic, etc.

2. **In QuantumCircuit**: Measurement results are represented as `Clbit` objects.
   These are non-JAX types that cannot be processed by JAX primitives.

When we encounter classical post-processing of measurement results (e.g., 
`meas_res * 2` or `parity(m1, m2)`), we cannot represent this computation
in the QuantumCircuit itself. Instead, we use placeholder objects.

Solution Architecture
---------------------
This module provides three key classes/concepts:

1. **ProcessedMeasurement**: A placeholder for measurement results that have
   undergone classical post-processing. Since QuantumCircuit cannot represent
   classical computation, this placeholder indicates "some classical processing
   happened here."

2. **MeasurementArray**: Handles the specific problem of measurement results
   being inserted into or extracted from JAX arrays. See detailed explanation
   in the MeasurementArray class documentation.

3. **qc_extraction_eqn_evaluator**: The equation evaluator that intercepts
   JAX primitives and either:
   - Executes quantum primitives directly (building the circuit)
   - Handles array operations on measurements (using MeasurementArray)
   - Creates ProcessedMeasurement placeholders for classical post-processing
   - Delegates to default JAX evaluation for pure classical operations

"""

import numpy as np


# =============================================================================
# SECTION 1: Placeholder Classes for Classical Post-Processing
# =============================================================================

class ProcessedMeasurement:
    """
    Placeholder for measurement results that have undergone classical post-processing.
    
    Problem Being Solved
    --------------------
    QuantumCircuit objects represent quantum operations and measurements, but they
    cannot represent classical computation on measurement results. When a Jaspr
    contains code like:
    
        meas_res = measure(qv)
        processed = meas_res * 2 + 1  # Classical post-processing
        
    We cannot encode the `* 2 + 1` operation in the QuantumCircuit. In many
    cases the post-processing can however be achieved through the 
    Jaspr.extract_post_processing feature, so instead of raising
    an error, we continue with the circuit extraction. Instead, we
    use this placeholder to indicate that classical processing occurred.
    An error is raised if a placeholder needs to decide over
    further construction steps of the circuit.
    
    Usage
    -----
    When the interpreter encounters an operation that:
    1. Takes measurement results (Clbit) as input
    2. Is not a quantum operation
    3. Produces output that depends on the measurement
    
    It creates a ProcessedMeasurement as the output, signaling that the actual
    value cannot be determined until runtime execution with real measurement data.
    
    Example
    -------
    >>> result, qc = jaspr.to_qc()
    >>> isinstance(result, ProcessedMeasurement)
    True  # The result involves classical post-processing
    """
    pass


class ParityHandle:
    """
    A handle representing a parity result with its expanded clbits.
    
    Problem Being Solved
    --------------------
    Parity operations compute the XOR of multiple measurement results. Unlike
    measurements which are physical operations, parity is a classical computation.
    Rather than creating a "fake" clbit for parity results (which breaks the
    1:1 mapping between clbits and actual measurements), we use this handle
    to track parity results.
    
    The clbits involved in the parity computation are accessible through the
    instruction's clbits attribute (nested parity handles are expanded and 
    duplicates are eliminated using symmetric difference for correct XOR semantics).
    
    Design Note
    -----------
    The handle stores a reference to the instruction object (from qc.data) and
    the QuantumCircuit itself. This makes the handle robust to circuit 
    modifications (like gate reordering by compilation passes) - similar to
    Clbit objects, every ParityHandle corresponds to a particular boolean value 
    that could in principle be uniquely computed by executing the underlying circuit.
    
    Attributes
    ----------
    instruction : Instruction
        The ParityOperation instruction object sitting in qc.data.
    qc : QuantumCircuit
        The QuantumCircuit containing this parity operation.
    
    Properties
    ----------
    clbits : list[Clbit]
        The list of Clbit objects involved in this parity. Retrieved from
        instruction.clbits.
    expectation : int
        The expected parity value (0, 1, or 2 for unknown). Retrieved from
        instruction.op.expectation.
    """
    def __init__(self, instruction, qc):
        self.instruction = instruction
        self.qc = qc
    
    @property
    def clbits(self):
        """Get the clbits involved in this parity from the underlying instruction."""
        return self.instruction.clbits
    
    @property
    def expectation(self):
        """Get the expected parity value from the underlying instruction."""
        return self.instruction.op.expectation
    
    def __repr__(self):
        return f"ParityHandle{tuple(self.clbits)}"
    
    def __hash__(self):
        # Use both instruction and qc identity for hashing
        # Same instruction can appear in different circuits with different semantics
        return hash((id(self.instruction), id(self.qc)))
    
    def __eq__(self, other):
        if isinstance(other, ParityHandle):
            # Compare by both instruction and qc identity
            return self.instruction is other.instruction and self.qc is other.qc
        return False


# =============================================================================
# SECTION 2: MeasurementArray - Handling Arrays of Measurement Results
# =============================================================================

class MeasurementArray(np.ndarray):
    """
    A numpy ndarray subclass for measurement-related values during QuantumCircuit extraction.
    
    Problem Being Solved
    --------------------
    In Jaspr, measurement results are JAX boolean values that can be freely
    combined into arrays. When lowering to QuantumCircuit, measurements return
    `Clbit` objects instead of JAX booleans. JAX array operations cannot handle
    Clbit objects directly.
    
    Solution
    --------
    MeasurementArray is a numpy ndarray subclass with dtype=object that stores:
    
    - Clbit objects: Measurement results that can be used in circuit operations
    - ParityHandle objects: Parity computation results  
    - ProcessedMeasurement: Marker for classical post-processing results
    - bool: Known boolean values (True/False)
    
    Being a numpy subclass means reshape, slice, concatenate, etc. work natively.
    The subclass provides type identification via isinstance() and helper methods.
    """
    
    def __new__(cls, data):
        """Create a new MeasurementArray from data."""
        arr = np.asarray(data, dtype=object).view(cls)
        return arr
    
    def __array_finalize__(self, obj):
        """Called after array construction to finalize the object."""
        pass
    
    def mark_as_processed(self):
        """
        Return a new MeasurementArray with all entries marked as processed.
        
        Returns
        -------
        MeasurementArray
            New array with all entries set to ProcessedMeasurement().
        """
        processed_data = np.array([ProcessedMeasurement() for _ in self.flat], dtype=object)
        return MeasurementArray(processed_data.reshape(self.shape))


# =============================================================================
# SECTION 3: Helper Functions
# =============================================================================

def contains_measurement_data(val):
    """
    Check if a value contains measurement-related data.
    
    Parameters
    ----------
    val : any
        Value to check.
    
    Returns
    -------
    bool
        True if the value is or contains Clbit, MeasurementArray, 
        ParityHandle, or ProcessedMeasurement data.
    """
    from qrisp import Clbit
    
    if isinstance(val, (Clbit, MeasurementArray, ProcessedMeasurement, ParityHandle)):
        return True
    if isinstance(val, list) and len(val):
        return contains_measurement_data(val[0])
    return False


def to_object_array(val):
    """
    Convert a measurement-related value to an object numpy array.
    
    Parameters
    ----------
    val : any
        Value to convert. Can be:
        - MeasurementArray: returned as-is (already an ndarray subclass)
        - Clbit, ParityHandle, ProcessedMeasurement, bool: wrapped in 0-d array
        - numpy array: converted to object dtype
        - Other: returned unchanged
    
    Returns
    -------
    numpy.ndarray or original value
        Object array with the value(s).
    """
    from qrisp import Clbit
    
    if isinstance(val, MeasurementArray):
        # MeasurementArray is already an ndarray subclass
        return val
    elif isinstance(val, (Clbit, ParityHandle, ProcessedMeasurement)):
        return np.array(val, dtype=object)
    elif isinstance(val, (bool, np.bool_)):
        return np.array(bool(val), dtype=object)
    elif isinstance(val, np.ndarray):
        return val.astype(object) if val.dtype != object else val
    else:
        return val


def apply_array_primitive(prim_name, params, invalues):
    """
    Apply a JAX array primitive to measurement data using numpy equivalents.
    
    Parameters
    ----------
    prim_name : str
        Name of the JAX primitive (e.g., 'broadcast_in_dim', 'concatenate').
    params : dict
        Parameters of the JAX primitive.
    invalues : list
        Input values to the primitive.
    
    Returns
    -------
    MeasurementArray, scalar, or None
        - MeasurementArray for array results
        - Scalar (Clbit, bool, ProcessedMeasurement, ParityHandle) for 0-d results
        - None if this primitive is not handled
    """
    # Convert all inputs to object arrays
    encoded = [to_object_array(v) for v in invalues]
    
    # Apply the numpy equivalent based on primitive name
    if prim_name == "broadcast_in_dim":
        shape = params["shape"]
        result = np.broadcast_to(encoded[0], shape)
        
    elif prim_name == "concatenate":
        dimension = params.get("dimension", 0)
        result = np.concatenate(encoded, axis=dimension)
        
    elif prim_name == "squeeze":
        dimensions = params.get("dimensions", None)
        result = np.squeeze(encoded[0], axis=dimensions)
        
    elif prim_name == "slice":
        start_indices = params["start_indices"]
        limit_indices = params["limit_indices"]
        slices = tuple(slice(s, e) for s, e in zip(start_indices, limit_indices))
        result = encoded[0][slices]
        
    elif prim_name == "dynamic_slice":
        # Start indices come from invalues[1:]
        start_indices = [int(encoded[i]) if np.ndim(encoded[i]) == 0 else int(encoded[i].flat[0]) 
                        for i in range(1, len(encoded))]
        slice_sizes = params["slice_sizes"]
        slices = tuple(slice(s, s + sz) for s, sz in zip(start_indices, slice_sizes))
        result = encoded[0][slices]
        
    elif prim_name == "gather":
        # Simple indexing case
        indices = encoded[1]
        if hasattr(indices, 'item'):
            idx = indices.item()
            if isinstance(idx, (int, np.integer)):
                idx = int(idx)
        elif np.ndim(indices) == 0:
            idx = int(indices) if isinstance(indices, (int, np.integer, np.ndarray)) else indices
        else:
            idx = int(indices[0]) if len(indices) == 1 else indices
        
        if isinstance(idx, (int, np.integer)):
            result = encoded[0].flat[idx]
        else:
            result = encoded[0][idx]
        
    elif prim_name == "reshape":
        new_sizes = params.get("new_sizes", params.get("dimensions", None))
        if new_sizes is not None:
            result = encoded[0].reshape(new_sizes)
        else:
            result = encoded[0]
            
    elif prim_name == "transpose":
        permutation = params.get("permutation", None)
        result = np.transpose(encoded[0], permutation)
        
    else:
        return None
    
    # Handle result
    result = np.asarray(result, dtype=object)
    
    if result.ndim == 0:
        # Scalar result - return the object directly
        return result.item()
    else:
        # Array result
        return MeasurementArray(result)


def resolve_measurement_arrays(value):
    """
    Recursively resolve MeasurementArrays to numpy arrays with dtype=object.
    
    This function is called at the end of jaspr_to_qc to convert the internal
    MeasurementArray representation to a standard numpy array that users can
    work with directly.
    
    Parameters
    ----------
    value : any
        The value to resolve. Can be:
        - MeasurementArray: viewed as plain numpy array with dtype=object
        - tuple/list: each element is recursively resolved
        - Other types: returned unchanged
    
    Returns
    -------
    any
        The resolved value with MeasurementArrays converted to numpy arrays.
    """
    if isinstance(value, MeasurementArray):
        # View as plain numpy array (MeasurementArray is already an ndarray subclass)
        return value.view(np.ndarray)
    elif isinstance(value, tuple):
        return tuple(resolve_measurement_arrays(v) for v in value)
    elif isinstance(value, list):
        return [resolve_measurement_arrays(v) for v in value]
    else:
        return value


def handle_classical_processing(invalues):
    """
    Handle operations that represent classical post-processing on measurement data.
    
    Many operations (arithmetic, comparisons, reductions, bitwise ops) represent
    classical computation that cannot be performed during circuit construction.
    This function provides a unified way to handle all such operations.
    
    The strategy is:
    1. If any input is a MeasurementArray, return a MeasurementArray of the same
       size with all entries marked as "processed".
    2. If inputs contain scalar measurement data (Clbit, ProcessedMeasurement,
       ParityHandle), return a scalar ProcessedMeasurement.
    3. If no measurement data is involved, return None to indicate default
       JAX evaluation should be used.
    
    Parameters
    ----------
    invalues : list
        Input values to the operation.
    
    Returns
    -------
    MeasurementArray, ProcessedMeasurement, or None
        - MeasurementArray with processed entries if input was an array
        - ProcessedMeasurement for scalar measurement inputs
        - None if no measurement data (use default JAX evaluation)
    """
    from qrisp import Clbit
    
    # Check for MeasurementArray inputs first (preserves array structure)
    for v in invalues:
        if isinstance(v, MeasurementArray):
            return v.mark_as_processed()
    
    # Check for scalar measurement data
    for v in invalues:
        if isinstance(v, (Clbit, ProcessedMeasurement, ParityHandle)):
            return ProcessedMeasurement()
    
    # Check for lists containing measurement data
    for v in invalues:
        if isinstance(v, list) and len(v) and contains_measurement_data(v[0]):
            return ProcessedMeasurement()
    
    # No measurement data - use default evaluation
    return None


# List of primitives that represent classical processing on measurement data.
# These operations cannot be performed during circuit construction because
# measurement results are not known until runtime.
CLASSICAL_PROCESSING_PRIMITIVES = {
    # Arithmetic operations
    "add", "sub", "mul", "div", "rem", "pow", "neg",
    "integer_pow", "floor", "ceil", "round", "abs",
    
    # Comparison operations  
    "eq", "ne", "lt", "gt", "le", "ge",
    
    # Reduction operations
    "reduce_sum", "reduce_prod", "reduce_max", "reduce_min",
    "reduce_or", "reduce_and", "reduce_xor",
    
    # Bitwise operations
    "not", "and", "or", "xor",
    "shift_left", "shift_right_arithmetic", "shift_right_logical",
}


# =============================================================================
# SECTION 4: Equation Evaluator Factory
# =============================================================================

def make_qc_extraction_eqn_evaluator(qc):
    """
    Create an equation evaluator for extracting a QuantumCircuit from a Jaspr.
    
    This factory function creates a closure over the QuantumCircuit being built,
    returning an evaluator function that can be passed to eval_jaxpr.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to build. Operations will be appended to this circuit.
    
    Returns
    -------
    callable
        An equation evaluator function with signature (eqn, context_dic) -> bool|None
    """

    def qc_extraction_eqn_evaluator(eqn, context_dic):
        """
        Evaluate a single Jaxpr equation during QuantumCircuit extraction.
        
        This function is called for each equation in the Jaspr. It determines
        how to handle the equation based on its primitive type:
        
        1. **Quantum Primitives** (jasp.*): Execute directly using their
           implementation, which appends operations to the QuantumCircuit.
        
        2. **Array Operations on Measurements**: Handle specially using
           MeasurementArray to maintain Clbit references through array ops.
        
        3. **Classical Operations on Measurements**: Create ProcessedMeasurement
           placeholders since we can't represent classical computation in QC.
        
        4. **Pure Classical Operations**: Delegate to default JAX evaluation
           by returning True.
        
        Parameters
        ----------
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : dict
            Dictionary mapping Jaxpr variables to their current values.
        
        Returns
        -------
        bool or None
            - True: Use default JAX evaluation for this equation
            - None/False: Equation was fully handled, skip default evaluation
        """
        # Import here to avoid circular imports
        from qrisp import Clbit
        from qrisp.jasp import (
            Jaspr,
            extract_invalues,
            insert_outvalues,
            QuantumPrimitive,
            ParityOperation
        )
        from qrisp.jasp.interpreter_tools.interpreters import (
            cond_to_cl_control
        )
        
        invalues = extract_invalues(eqn, context_dic)
        prim_name = eqn.primitive.name
        
        # -----------------------------------------------------------------
        # SECTION 4.1: Control Flow and Structural Primitives
        # -----------------------------------------------------------------
        
        if prim_name == "jit" and (isinstance(eqn.params["jaxpr"], Jaspr) or any(contains_measurement_data(v) for v in invalues)):
            # Nested Jaspr (from @qache or similar) - evaluate with our interpreter
            from qrisp.jasp import eval_jaxpr
            
            definition_jaxpr = eqn.params["jaxpr"]
            res = eval_jaxpr(definition_jaxpr.jaxpr, eqn_evaluator=qc_extraction_eqn_evaluator)(
                *(invalues + definition_jaxpr.consts)
            )
            
            if len(definition_jaxpr.jaxpr.outvars) == 1:
                res = [res]
            
            insert_outvalues(eqn, context_dic, res)
            return
        
        elif prim_name == "jit":
            return True
        
        elif prim_name == "cond":
            # Conditional branching - may become classically controlled operation
            return cond_to_cl_control(eqn, context_dic, qc_extraction_eqn_evaluator)
        
        elif prim_name == "while":
            # While loops need special handling - delegate to default for now
            # (the loop will be unrolled during evaluation)
            return True
        
        # -----------------------------------------------------------------
        # SECTION 4.2: Quantum Primitives with Special Handling
        # -----------------------------------------------------------------
        
        elif prim_name == "jasp.parity":
            # Parity operation: XOR of multiple classical bits
            # Note: Parity on arrays is dissolved into loops during tracing,
            # so invalues are always scalars (Clbit, ParityHandle, or ProcessedMeasurement)
            
            # Check for ProcessedMeasurement - return ProcessedMeasurement
            if any(isinstance(v, ProcessedMeasurement) for v in invalues):
                insert_outvalues(eqn, context_dic, ProcessedMeasurement())
                return
            
            # Expand all inputs to clbits using symmetric difference (XOR semantics)
            # This means duplicate clbits cancel out (a XOR a = 0)
            clbit_set = set()
            for inp in invalues:
                if isinstance(inp, Clbit):
                    clbit_set.symmetric_difference_update({inp})
                elif isinstance(inp, ParityHandle):
                    # ParityHandle stores expanded clbits
                    clbit_set.symmetric_difference_update(inp.clbits)
            
            # Convert set to sorted list for deterministic ordering
            expanded_clbits = sorted(clbit_set, key=lambda cb: qc.clbits.index(cb))
            
            # Add parity operation to the circuit (for stim conversion)
            qc.append(ParityOperation(len(expanded_clbits), expectation=eqn.params["expectation"]),
                      clbits=expanded_clbits)
            
            # Get the instruction we just appended
            parity_instr = qc.data[-1]
            
            # Create a parity handle with the instruction and circuit reference
            handle = ParityHandle(parity_instr, qc)
            insert_outvalues(eqn, context_dic, handle)
            return
        
        # -----------------------------------------------------------------
        # SECTION 4.3: Type Conversion (convert_element_type)
        # -----------------------------------------------------------------
        # JAX often inserts type conversions. For measurement data:
        # - bool->bool conversions pass through unchanged
        # - bool->int conversions for Clbit pass through (used by cond primitive)
        # - Conversions to float types mark as processed since we can't
        #   actually compute with measurement values
        
        elif prim_name == "convert_element_type":
            inval = context_dic[eqn.invars[0]]
            new_dtype = eqn.params.get("new_dtype", None)
            
            if isinstance(inval, MeasurementArray):
                # Check if converting to a non-boolean/non-integer type
                if new_dtype is not None and not (np.issubdtype(new_dtype, np.bool_) or np.issubdtype(new_dtype, np.integer)):
                    # Converting to float type - mark as processed
                    context_dic[eqn.outvars[0]] = inval.mark_as_processed()
                else:
                    # Bool-to-bool, bool-to-int, or unknown conversion - pass through
                    context_dic[eqn.outvars[0]] = inval
                return
            elif isinstance(inval, Clbit):
                # Clbit should pass through for bool->int conversions (used by cond)
                # Only mark as processed for float conversions
                if new_dtype is not None and not (np.issubdtype(new_dtype, np.bool_) or np.issubdtype(new_dtype, np.integer)):
                    context_dic[eqn.outvars[0]] = ProcessedMeasurement()
                else:
                    context_dic[eqn.outvars[0]] = inval
                return
            elif isinstance(inval, ProcessedMeasurement):
                # ProcessedMeasurement stays processed
                context_dic[eqn.outvars[0]] = ProcessedMeasurement()
                return
            elif isinstance(inval, list) and len(inval) and isinstance(
                inval[0], (ProcessedMeasurement, Clbit)
            ):
                # List of measurement data
                if new_dtype is not None and not np.issubdtype(new_dtype, np.bool_):
                    context_dic[eqn.outvars[0]] = ProcessedMeasurement()
                else:
                    context_dic[eqn.outvars[0]] = inval
                return
            return True
        
        # -----------------------------------------------------------------
        # SECTION 4.4: Array Operations on Measurement Data
        # -----------------------------------------------------------------
        # These primitives handle JAX array operations when the arrays contain
        # measurement results. We use a generic handler that converts to object
        # arrays, applies the numpy equivalent, and wraps back into MeasurementArray.
        
        elif prim_name in ("broadcast_in_dim", "concatenate", "squeeze", "slice", 
                           "dynamic_slice", "gather", "reshape", "transpose"):
            # Check if any input contains measurement data
            if any(contains_measurement_data(v) for v in invalues):
                result = apply_array_primitive(prim_name, eqn.params, invalues)
                if result is not None:
                    insert_outvalues(eqn, context_dic, result)
                    return
            return True
        
        elif prim_name == "scatter":
            # scatter: Update array elements at specified indices
            # Used in array assignment operations (e.g., building arrays in loops)
            #
            # Scatter has the form: scatter(operand, indices, updates)
            # - operand: The array being updated
            # - indices: Where to place the updates
            # - updates: The values to insert
            
            operand = invalues[0]
            indices = invalues[1]
            updates = invalues[2] if len(invalues) > 2 else None
            
            # Unwrap updates if it's a single-element list
            if isinstance(updates, list) and len(updates) == 1:
                updates = updates[0]
            
            # Helper to get index from various index formats
            def get_scatter_index(indices):
                if hasattr(indices, 'item'):
                    return int(indices.item())
                elif hasattr(indices, '__len__') and len(indices) == 1:
                    return int(indices[0])
                else:
                    return int(indices)
            
            # Check if we're working with measurement data
            if isinstance(operand, MeasurementArray):
                idx = get_scatter_index(indices)
                
                # Create a copy and update directly (object array)
                result = operand.copy()
                result[idx] = updates
                insert_outvalues(eqn, context_dic, result)
                return
            
            elif contains_measurement_data(updates):
                # Operand is not a MeasurementArray but updates contains measurement data
                # This happens when building an array from measurements in a loop
                
                # Get the shape of the operand
                if hasattr(operand, 'shape'):
                    size = int(np.prod(operand.shape))
                elif hasattr(operand, '__len__'):
                    size = len(operand)
                else:
                    size = 1
                
                # Initialize MeasurementArray with False values
                new_data = np.array([False] * size, dtype=object)
                
                # Set the update at the given index
                idx = get_scatter_index(indices)
                new_data[idx] = updates
                
                result = MeasurementArray(new_data)
                insert_outvalues(eqn, context_dic, result)
                return
            
            elif isinstance(operand, ProcessedMeasurement):
                insert_outvalues(eqn, context_dic, ProcessedMeasurement())
                return
            
            else:
                # No measurement data involved
                return True
        
        # -----------------------------------------------------------------
        # SECTION 4.5: Classical Processing Operations
        # -----------------------------------------------------------------
        # This section handles all operations that represent classical
        # computation on measurement data. These operations cannot be
        # performed during circuit construction because measurement results
        # are not known until runtime.
        #
        # We use a unified approach:
        # 1. Check if the primitive is in CLASSICAL_PROCESSING_PRIMITIVES
        # 2. Use handle_classical_processing() to create appropriate output
        # 3. This preserves array structure (MeasurementArray with processed
        #    entries) while marking the data as "processed"
        
        elif prim_name in CLASSICAL_PROCESSING_PRIMITIVES:
            result = handle_classical_processing(invalues)
            if result is not None:
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # -----------------------------------------------------------------
        # SECTION 4.6: Default Handling
        # -----------------------------------------------------------------
        
        else:
            # Check if any input contains measurement data
            for val in invalues:
                if contains_measurement_data(val):
                    break
            else:
                # No measurement data - check if it's a quantum primitive
                if isinstance(eqn.primitive, QuantumPrimitive):
                    # Execute quantum primitive directly
                    outvalues = eqn.primitive.impl(*invalues, **eqn.params)
                    insert_outvalues(eqn, context_dic, outvalues)
                    return
                else:
                    # Pure classical operation - use default JAX evaluation
                    return True
        
        # -----------------------------------------------------------------
        # SECTION 4.7: Fallback for Unhandled Measurement Operations
        # -----------------------------------------------------------------
        # If we reach here, the operation involves measurement data but isn't
        # one of the specifically handled cases above. We create 
        # ProcessedMeasurement placeholders for the outputs.
        
        if len(eqn.outvars) == 0:
            return
        elif len(eqn.outvars) == 1 and not eqn.primitive.multiple_results:
            outvalues = ProcessedMeasurement()
        elif len(eqn.outvars) >= 1:
            outvalues = [ProcessedMeasurement() for _ in range(len(eqn.outvars))]
        
        insert_outvalues(eqn, context_dic, outvalues)
        
    return qc_extraction_eqn_evaluator


# =============================================================================
# SECTION 5: Public API - jaspr_to_qc Function
# =============================================================================

def jaspr_to_qc(jaspr, *args):
    """
    Convert a Jaspr into a QuantumCircuit.
    
    This is the main entry point for converting Jaspr intermediate representation
    into a static QuantumCircuit that can be executed on quantum hardware or
    simulators.
    
    Limitations
    -----------
    - **Real-time feedback**: Algorithms that use measurement results to control
      subsequent quantum operations (true real-time feedback) cannot be fully
      represented. The control flow will be evaluated with placeholder values.
    
    - **Classical post-processing**: Any classical computation on measurement
      results (arithmetic, comparisons, etc.) cannot be represented in the
      QuantumCircuit. These are replaced with ProcessedMeasurement placeholders.
    
    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr object to convert.
    *args : tuple
        Arguments to call the Jaspr with. These should NOT include the
        QuantumCircuit argument (it's added automatically). Exclude any
        static arguments like callables.

    Returns
    -------
    tuple
        A tuple containing:
        - Return values from the Jaspr (QuantumVariable returns become qubit lists,
          measurement results become Clbit or ProcessedMeasurement objects)
        - The constructed QuantumCircuit (always the last element)

    Examples
    --------
    Basic circuit extraction:

    ::

        from qrisp import *
        from qrisp.jasp import make_jaspr

        def example_function(i):
            qv = QuantumVariable(i)
            cx(qv[0], qv[1])
            t(qv[1])
            return qv

        jaspr = make_jaspr(example_function)(2)
        qb_list, qc = jaspr_to_qc(jaspr, 2)
        
        print(qc)
        # qb_0: ──■───────
        #       ┌─┴─┐┌───┐
        # qb_1: ┤ X ├┤ T ├
        #       └───┘└───┘
        
    With measurement post-processing (returns ProcessedMeasurement):
        
    ::
        
        from qrisp.jasp.interpreter_tools.interpreters import ProcessedMeasurement
        
        def example_function(i):
            qf = QuantumFloat(i)
            cx(qf[0], qf[1])
            t(qf[1])
            
            meas_res = measure(qf)
            meas_res *= 2  # Classical post-processing
            return meas_res
        
        jaspr = make_jaspr(example_function)(2)
        meas_res, qc = jaspr_to_qc(jaspr, 2)
        
        print(isinstance(meas_res, ProcessedMeasurement))
        # True
        
    With array operations on measurements:
    
    ::
    
        import jax.numpy as jnp
        
        def array_example():
            qv = QuantumVariable(3)
            m0 = measure(qv[0])
            m1 = measure(qv[1])
            m2 = measure(qv[2])
            
            # Create array of measurements
            arr = jnp.array([m0, m1, m2])
            
            # Extract first measurement
            return arr[0]
        
        jaspr = make_jaspr(array_example)()
        result, qc = jaspr_to_qc(jaspr)
        
        # result is the Clbit corresponding to m0's measurement
        print(type(result))  # <class 'qrisp.circuit.clbit.Clbit'>
    """
    from qrisp import QuantumCircuit
    from qrisp.jasp import eval_jaxpr

    qc = QuantumCircuit()
    
    ammended_args = list(args) + [qc]
    
    if len(ammended_args) != len(jaspr.invars):
        raise Exception(
            "Supplied invalid number of arguments to Jaspr.to_qc "
            "(please exclude any static arguments, in particular callables)"
        )

    res = eval_jaxpr(
        jaspr, 
        eqn_evaluator=make_qc_extraction_eqn_evaluator(qc)
    )(*ammended_args)

    # Resolve MeasurementArrays to numpy arrays with dtype=object
    res = resolve_measurement_arrays(res)

    return res
