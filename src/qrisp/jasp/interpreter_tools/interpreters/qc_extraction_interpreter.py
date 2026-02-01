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
    
    The handle stores the fully expanded list of clbits involved in the parity
    computation (nested parity handles are expanded and duplicates are eliminated
    using symmetric difference for correct XOR semantics).
    
    Attributes
    ----------
    index : int
        The sequential index of this parity operation (for MeasurementArray encoding
        and stim converter mapping).
    clbits : list[Clbit]
        The list of Clbit objects involved in this parity (already expanded and
        deduplicated via symmetric difference).
    expectation : int
        The expected parity value (0, 1, or 2 for unknown). Used for detector mode.
    """
    def __init__(self, index, clbits, expectation=2):
        self.index = index
        self.clbits = clbits
        self.expectation = expectation
    
    def __repr__(self):
        return f"ParityHandle(index={self.index}, clbits={self.clbits}, expectation={self.expectation})"
    
    def __hash__(self):
        # Include clbits in hash so handles from different circuits don't match
        return hash((self.index, tuple(self.clbits)))
    
    def __eq__(self, other):
        if isinstance(other, ParityHandle):
            # Compare both index and clbits to catch accidental circuit mixing
            return self.index == other.index and self.clbits == other.clbits
        return False


# =============================================================================
# SECTION 2: MeasurementArray - Handling Arrays of Measurement Results
# =============================================================================

class MeasurementArray:
    """
    Represents an array of boolean values (possibly measurement results) during
    QuantumCircuit extraction.
    
    Problem Being Solved
    --------------------
    In Jaspr, measurement results are JAX boolean values that can be freely
    combined into arrays using operations like:
    
        m0 = measure(qv[0])  # Returns JAX boolean tracer
        m1 = measure(qv[1])  # Returns JAX boolean tracer
        m2 = measure(qv[2])  # Returns JAX boolean tracer
        arr = jnp.array([m0, m1, m2])  # Creates JAX boolean array
        first = arr[0]  # Extracts first element
    
    When lowering to QuantumCircuit, measurements return `Clbit` objects instead
    of JAX booleans. The problem is that JAX array operations (broadcast_in_dim,
    concatenate, slice, etc.) cannot handle Clbit objects - they expect JAX-
    compatible types.
    
    Solution
    --------
    MeasurementArray provides a bridge between the JAX array world and Clbit world:
    
    1. **Integer Encoding**: We encode array contents as integers:
       - 0: Known boolean value False
       - 1: Known boolean value True
       - 2: ProcessedMeasurement (result of classical post-processing)
       - Negative values: Measurement results (indices into qc.clbits)
         * -1 corresponds to the last clbit in the circuit
         * -2 corresponds to the second-to-last clbit
         * etc.
    
    2. **Transparent Extraction**: When extracting a single element that
       corresponds to a measurement, we return the actual Clbit object.
       This allows the extracted value to be used in subsequent operations
       (like controlled gates or parity primitives) that expect Clbit inputs.
       
    3. **ProcessedMeasurement Tracking**: When classical operations (like ~, &, |)
       are applied to measurement arrays, we mark those entries as "processed"
       (value 2). Extracting such an entry returns a ProcessedMeasurement object.
       If a circuit-influencing operation (like parity) receives a processed
       entry, it raises an error since we can't build the circuit correctly.
    
    4. **ParityHandle Tracking**: Values >= PARITY_HANDLE_OFFSET encode references
       to parity results stored in the parity_handles list. These are extracted
       as ParityHandle objects that can be used in nested parity operations.
    
    5. **Array Operations**: We intercept JAX array primitives and implement
       them using NumPy operations on our integer-encoded data.
    
    Why Negative Indexing?
    ----------------------
    We use negative indices because clbits are typically added during circuit
    construction. When a measurement happens, a new clbit is created. Using
    negative indices (from the end) means the encoding remains valid even as
    more clbits are added later.
    
    For example, if we have clbits [cb_0, cb_1, cb_2]:
    - cb_0 is at index 0, encoded as -3 (0 - 3 = -3)
    - cb_1 is at index 1, encoded as -2 (1 - 3 = -2)  
    - cb_2 is at index 2, encoded as -1 (2 - 3 = -1)
    
    When extracting, -1 gives us qc.clbits[-1] = cb_2, which is correct.
    
    Encoding Scheme
    ---------------
    - 0: Known boolean False
    - 1: Known boolean True
    - 2: ProcessedMeasurement (classical post-processing result)
    - >= 3: ParityHandle reference (index = value - 3)
    - < 0: Clbit reference (negative index into qc.clbits)
    
    Attributes
    ----------
    qc : QuantumCircuit
        Reference to the quantum circuit being built. Needed to resolve
        negative indices back to actual Clbit objects.
    parity_handles : list
        List of ParityHandle objects created during extraction. Needed to
        resolve parity handle indices back to actual ParityHandle objects.
    data : numpy.ndarray
        Array of integers encoding the boolean/measurement values.
        
    Class Constants
    ---------------
    PROCESSED_VALUE : int
        The integer value (2) used to encode ProcessedMeasurement entries.
    PARITY_HANDLE_OFFSET : int
        The offset (3) added to parity handle indices to encode them.
    """
    
    # Class constant for the "processed" marker value
    PROCESSED_VALUE = 2
    
    # Offset for encoding ParityHandle indices (value = handle.index + PARITY_HANDLE_OFFSET)
    PARITY_HANDLE_OFFSET = 3
    
    def __init__(self, qc, data, parity_handles=None, shape=None):
        """
        Initialize a MeasurementArray.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit being built.
        data : array-like
            Array of integers encoding boolean/measurement values.
        parity_handles : list, optional
            List of ParityHandle objects. If None, uses empty list.
        shape : tuple, optional
            Logical shape of the array. If None, defaults to 1D shape of data.
        """
        self.qc = qc
        self.data = np.array(data, dtype=np.int64).flatten()
        self.parity_handles = parity_handles if parity_handles is not None else []
        self._shape = shape if shape is not None else (len(self.data),)
    
    @property
    def shape(self):
        """Return the logical shape of the array."""
        return self._shape
    
    def reshape(self, new_shape):
        """Return a new MeasurementArray with the specified shape."""
        # Validate that the total size matches
        if np.prod(new_shape) != len(self.data):
            raise ValueError(f"Cannot reshape array of size {len(self.data)} into shape {new_shape}")
        return MeasurementArray(self.qc, self.data, self.parity_handles, new_shape)
    
    def __len__(self):
        return len(self.data)
    
    def _decode_value(self, val):
        """
        Decode a single integer value to its corresponding object.
        
        Parameters
        ----------
        val : int
            The encoded integer value.
        
        Returns
        -------
        Clbit, bool, ProcessedMeasurement, or ParityHandle
            The decoded object.
        """
        if val < 0:
            return self.qc.clbits[int(val)]
        elif val == self.PROCESSED_VALUE:
            return ProcessedMeasurement()
        elif val >= self.PARITY_HANDLE_OFFSET:
            return self.parity_handles[int(val) - self.PARITY_HANDLE_OFFSET]
        else:
            return bool(val)
    
    def __getitem__(self, key):
        """
        Extract element(s) from the MeasurementArray.
        
        This is the key method that bridges back to the Clbit world. When
        extracting a single element that represents a measurement, we return
        the actual Clbit object so it can be used in subsequent quantum
        operations (e.g., classically controlled gates).
        
        Parameters
        ----------
        key : int or slice
            Index or slice to extract.
        
        Returns
        -------
        Clbit, bool, ProcessedMeasurement, ParityHandle, or MeasurementArray
            - If extracting a single measurement element (negative): returns the Clbit
            - If extracting a single known boolean (0 or 1): returns that boolean
            - If extracting a processed element (2): returns ProcessedMeasurement
            - If extracting a parity handle element (>= 3): returns ParityHandle
            - If extracting a slice: returns a new MeasurementArray
        """
        if isinstance(key, (int, np.integer)):
            return self._decode_value(self.data[key])
        elif isinstance(key, slice):
            return MeasurementArray(self.qc, self.data[key], self.parity_handles)
        else:
            raise TypeError(
                f"MeasurementArray indices must be integers or slices, not {type(key)}"
            )
    
    def has_processed_entries(self):
        """Check if this array contains any ProcessedMeasurement entries."""
        return np.any(self.data == self.PROCESSED_VALUE)
    
    def to_clbit_list(self):
        """
        Convert to a list of Clbits and ParityHandles.
        
        This is used by operations like parity that need actual measurement
        or parity values.
        
        Returns
        -------
        list[Clbit or ParityHandle]
            List of Clbit objects and/or ParityHandle objects.
        
        Raises
        ------
        Exception
            If the array contains any ProcessedMeasurement entries.
        """
        if self.has_processed_entries():
            raise Exception(
                "Cannot convert MeasurementArray to Clbit list: array contains "
                "processed measurement values (from operations like ~, &, |). "
                "These values cannot be used in circuit-influencing operations."
            )
        
        result = []
        for val in self.data:
            decoded = self._decode_value(val)
            if isinstance(decoded, bool):
                raise Exception(
                    f"Cannot convert MeasurementArray entry {val} to Clbit/ParityHandle: "
                    "only measurement results (negative indices) or parity handles can be converted."
                )
            result.append(decoded)
        return result
    
    def resolve(self):
        """
        Resolve this MeasurementArray to a numpy array with dtype=object.
        
        This converts the internal integer encoding back to actual objects:
        - Negative values -> Clbit objects
        - 0 or 1 -> boolean values
        - 2 -> ProcessedMeasurement objects
        - >= 3 -> ParityHandle objects
        
        Returns
        -------
        numpy.ndarray
            Array with dtype=object containing the resolved values.
        """
        # Resolve flat data first
        resolved_flat = np.empty(len(self.data), dtype=object)
        for i, val in enumerate(self.data):
            resolved_flat[i] = self._decode_value(val)
        
        # Reshape to logical shape
        return resolved_flat.reshape(self._shape)
    
    def mark_as_processed(self):
        """
        Return a new MeasurementArray with all entries marked as processed.
        
        This is used when classical operations (like ~, &, |) are applied
        to the array. The original measurement information is lost, replaced
        with the "processed" marker.
        
        Returns
        -------
        MeasurementArray
            New array with all entries set to PROCESSED_VALUE (2).
        """
        return MeasurementArray(
            self.qc, 
            np.full_like(self.data, self.PROCESSED_VALUE),
            self.parity_handles
        )


# =============================================================================
# SECTION 3: Helper Functions
# =============================================================================

def contains_measurement_data(val):
    """
    Check if a value contains measurement-related data.
    
    This helper is used to determine whether an operation involves measurement
    results and thus requires special handling (rather than default JAX evaluation).
    
    Parameters
    ----------
    val : any
        Value to check.
    
    Returns
    -------
    bool
        True if the value is or contains Clbit, MeasurementArray, or
        ProcessedMeasurement data.
    """
    from qrisp import Clbit
    
    if isinstance(val, (Clbit, MeasurementArray, ProcessedMeasurement)):
        return True
    if isinstance(val, list) and len(val):
        return contains_measurement_data(val[0])
    return False


def encode_to_int_array(val, qc, parity_handles):
    """
    Encode a measurement-related value to an integer numpy array.
    
    This converts any measurement-related type to the internal integer encoding
    used by MeasurementArray, returning both the data and its shape.
    
    Parameters
    ----------
    val : any
        Value to encode. Can be:
        - MeasurementArray: returns its data reshaped to logical shape
        - Clbit: encoded as negative index
        - ProcessedMeasurement: encoded as PROCESSED_VALUE (2)
        - ParityHandle: encoded as index + PARITY_HANDLE_OFFSET
        - bool/int: encoded as 0 or 1
        - numpy array: passed through unchanged
    qc : QuantumCircuit
        The quantum circuit (needed for Clbit encoding).
    parity_handles : list
        List of ParityHandle objects.
    
    Returns
    -------
    numpy.ndarray
        Integer array with the encoded value(s).
    """
    from qrisp import Clbit
    
    if isinstance(val, MeasurementArray):
        return val.data.reshape(val.shape)
    elif isinstance(val, Clbit):
        clbit_idx = qc.clbits.index(val)
        neg_idx = clbit_idx - len(qc.clbits)
        return np.array(neg_idx, dtype=np.int64)
    elif isinstance(val, ProcessedMeasurement):
        return np.array(MeasurementArray.PROCESSED_VALUE, dtype=np.int64)
    elif isinstance(val, ParityHandle):
        return np.array(val.index + MeasurementArray.PARITY_HANDLE_OFFSET, dtype=np.int64)
    elif isinstance(val, (bool, np.bool_)):
        return np.array(int(val), dtype=np.int64)
    elif isinstance(val, (int, np.integer)):
        return np.array(int(val), dtype=np.int64)
    elif isinstance(val, np.ndarray):
        return val.astype(np.int64)
    else:
        return val


def apply_array_primitive(prim_name, params, invalues, qc, parity_handles):
    """
    Apply a JAX array primitive to measurement data using numpy equivalents.
    
    This generic handler converts measurement-related inputs to integer arrays,
    applies the numpy equivalent of the JAX primitive, and wraps the result
    back into a MeasurementArray.
    
    Parameters
    ----------
    prim_name : str
        Name of the JAX primitive (e.g., 'broadcast_in_dim', 'concatenate').
    params : dict
        Parameters of the JAX primitive.
    invalues : list
        Input values to the primitive.
    qc : QuantumCircuit
        The quantum circuit being built.
    parity_handles : list
        List of ParityHandle objects.
    
    Returns
    -------
    MeasurementArray, Clbit, ProcessedMeasurement, or None
        - MeasurementArray for array results
        - Scalar (Clbit, bool, ProcessedMeasurement) for scalar results
        - None if this primitive is not handled
    """
    # Encode all inputs to integer arrays
    encoded = [encode_to_int_array(v, qc, parity_handles) for v in invalues]
    
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
        # Build slice tuple for each dimension
        slices = tuple(slice(s, e) for s, e in zip(start_indices, limit_indices))
        result = encoded[0][slices]
        
    elif prim_name == "dynamic_slice":
        # Start indices come from invalues[1:]
        start_indices = [int(encoded[i]) for i in range(1, len(encoded))]
        slice_sizes = params["slice_sizes"]
        # Build slice tuple
        slices = tuple(slice(s, s + sz) for s, sz in zip(start_indices, slice_sizes))
        result = encoded[0][slices]
        
    elif prim_name == "gather":
        # Simple indexing case
        indices = encoded[1]
        if hasattr(indices, 'item'):
            idx = int(indices.item())
        elif np.ndim(indices) == 0:
            idx = int(indices)
        else:
            idx = int(indices[0]) if len(indices) == 1 else indices
        result = encoded[0].flat[idx] if isinstance(idx, (int, np.integer)) else encoded[0][idx]
        
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
    
    # Wrap result back into MeasurementArray (or scalar if 0-d)
    result = np.asarray(result, dtype=np.int64)
    
    if result.ndim == 0:
        # Scalar result - decode to actual type
        val = int(result)
        if val < 0:
            return qc.clbits[val]
        elif val == MeasurementArray.PROCESSED_VALUE:
            return ProcessedMeasurement()
        elif val >= MeasurementArray.PARITY_HANDLE_OFFSET:
            return parity_handles[val - MeasurementArray.PARITY_HANDLE_OFFSET]
        else:
            return bool(val)
    else:
        # Array result
        return MeasurementArray(qc, result.flatten(), parity_handles, result.shape)


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
        - MeasurementArray: resolved to numpy array with dtype=object
        - tuple/list: each element is recursively resolved
        - Other types: returned unchanged
    
    Returns
    -------
    any
        The resolved value with MeasurementArrays converted to numpy arrays.
    """
    if isinstance(value, MeasurementArray):
        return value.resolve()
    elif isinstance(value, tuple):
        return tuple(resolve_measurement_arrays(v) for v in value)
    elif isinstance(value, list):
        return [resolve_measurement_arrays(v) for v in value]
    else:
        return value


def handle_classical_processing(qc, invalues):
    """
    Handle operations that represent classical post-processing on measurement data.
    
    Many operations (arithmetic, comparisons, reductions, bitwise ops) represent
    classical computation that cannot be performed during circuit construction.
    This function provides a unified way to handle all such operations.
    
    The strategy is:
    1. If any input is a MeasurementArray, return a MeasurementArray of the same
       size with all entries marked as "processed" (value 2).
    2. If inputs contain scalar measurement data (Clbit, ProcessedMeasurement),
       return a scalar ProcessedMeasurement.
    3. If no measurement data is involved, return None to indicate default
       JAX evaluation should be used.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit being built (needed for MeasurementArray).
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
        if isinstance(v, (Clbit, ProcessedMeasurement)):
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
    
    # Bitwise operations (scalar versions - array versions handled separately)
    "shift_left", "shift_right_arithmetic", "shift_right_logical",
    
    # Type conversions that change semantics
    # Note: convert_element_type is handled specially since bool->bool is OK
}


# =============================================================================
# SECTION 4: Equation Evaluator Factory
# =============================================================================

def make_qc_extraction_eqn_evaluator(qc, parity_handles):
    """
    Create an equation evaluator for extracting a QuantumCircuit from a Jaspr.
    
    This factory function creates a closure over the QuantumCircuit being built
    and the parity_handles list, returning an evaluator function that can be 
    passed to eval_jaxpr.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to build. Operations will be appended to this circuit.
    parity_handles : list
        A list to accumulate ParityHandle objects during extraction.
    
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
        
        if prim_name == "jit" and isinstance(eqn.params["jaxpr"], Jaspr):
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
            # Generic jit (e.g., from jnp.vstack's atleast_2d) - evaluate if contains measurement data
            if any(contains_measurement_data(v) for v in invalues):
                from qrisp.jasp import eval_jaxpr
                
                definition_jaxpr = eqn.params["jaxpr"]
                res = eval_jaxpr(definition_jaxpr.jaxpr, eqn_evaluator=qc_extraction_eqn_evaluator)(
                    *(invalues + definition_jaxpr.consts)
                )
                
                if len(definition_jaxpr.jaxpr.outvars) == 1:
                    res = [res]
                
                insert_outvalues(eqn, context_dic, res)
                return
            else:
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
            # Parity results are stored directly in ParityHandle with expanded clbits
            
            # Check for ProcessedMeasurement in scalar inputs - return ProcessedMeasurement
            if any(isinstance(v, ProcessedMeasurement) for v in invalues):
                insert_outvalues(eqn, context_dic, ProcessedMeasurement())
                return
            
            # Check for ProcessedMeasurement in MeasurementArray inputs - return ProcessedMeasurement
            for v in invalues:
                if isinstance(v, MeasurementArray) and v.has_processed_entries():
                    insert_outvalues(eqn, context_dic, ProcessedMeasurement())
                    return
            
            # Collect all input clbits and parity handles
            parity_inputs = []
            for v in invalues:
                if isinstance(v, MeasurementArray):
                    parity_inputs.extend(v.to_clbit_list())
                elif isinstance(v, ParityHandle):
                    # Nested parity - expand and use symmetric difference
                    parity_inputs.append(v)
                else:
                    # Should be a Clbit
                    parity_inputs.append(v)
            
            # Expand all inputs to clbits using symmetric difference (XOR semantics)
            # This means duplicate clbits cancel out (a XOR a = 0)
            clbit_set = set()
            for inp in parity_inputs:
                if isinstance(inp, Clbit):
                    clbit_set.symmetric_difference_update({inp})
                elif isinstance(inp, ParityHandle):
                    # ParityHandle already stores expanded clbits
                    clbit_set.symmetric_difference_update(inp.clbits)
            
            # Convert set to sorted list for deterministic ordering
            expanded_clbits = sorted(clbit_set, key=lambda cb: qc.clbits.index(cb))
            
            # Add parity operation to the circuit (for stim conversion)
            qc.append(ParityOperation(len(expanded_clbits), expectation=eqn.params["expectation"]),
                      clbits=expanded_clbits)
            
            # Create a parity handle with the expanded clbits
            parity_index = len(parity_handles)
            handle = ParityHandle(parity_index, expanded_clbits, expectation=eqn.params["expectation"])
            parity_handles.append(handle)
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
        # measurement results. We use a generic handler that:
        # 1. Encodes all inputs to integer arrays
        # 2. Applies the numpy equivalent of the primitive
        # 3. Wraps the result back into MeasurementArray
        
        elif prim_name in ("broadcast_in_dim", "concatenate", "squeeze", "slice", 
                           "dynamic_slice", "gather", "reshape", "transpose"):
            # Check if any input contains measurement data
            if any(contains_measurement_data(v) for v in invalues):
                result = apply_array_primitive(prim_name, eqn.params, invalues, qc, parity_handles)
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
            #
            # We need to handle this carefully:
            # - If updates is a Clbit, we're building an array of measurements
            # - If updates is ProcessedMeasurement, mark that position as processed
            # - If operand is MeasurementArray, we update it properly
            
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
                
                # Create a copy of the data
                new_data = operand.data.copy()
                
                if isinstance(updates, Clbit):
                    # Insert a measurement result at the given index
                    clbit_idx = qc.clbits.index(updates)
                    neg_idx = clbit_idx - len(qc.clbits)
                    new_data[idx] = neg_idx
                elif isinstance(updates, ParityHandle):
                    # Insert a parity handle reference
                    new_data[idx] = updates.index + MeasurementArray.PARITY_HANDLE_OFFSET
                elif isinstance(updates, ProcessedMeasurement):
                    # Mark this position as processed
                    new_data[idx] = MeasurementArray.PROCESSED_VALUE
                elif isinstance(updates, (bool, np.bool_)):
                    # Insert a known boolean value
                    new_data[idx] = int(updates)
                elif isinstance(updates, MeasurementArray):
                    # Scattering another MeasurementArray
                    # For simplicity, mark as processed
                    new_data[idx] = MeasurementArray.PROCESSED_VALUE
                else:
                    # Unknown update type - mark as processed to be safe
                    new_data[idx] = MeasurementArray.PROCESSED_VALUE
                
                result = MeasurementArray(qc, new_data, parity_handles)
                insert_outvalues(eqn, context_dic, result)
                return
            
            elif isinstance(updates, Clbit):
                # Operand is not a MeasurementArray but updates is a Clbit
                # This happens when building an array from measurements in a loop
                # Convert the operand to MeasurementArray first
                
                # Get the shape of the operand
                if hasattr(operand, 'shape'):
                    size = int(np.prod(operand.shape))
                elif hasattr(operand, '__len__'):
                    size = len(operand)
                else:
                    size = 1
                
                # Initialize MeasurementArray with the operand's values
                # (assume all False/0 for boolean arrays)
                new_data = np.zeros(size, dtype=np.int64)
                
                # Set the update at the given index
                idx = get_scatter_index(indices)
                clbit_idx = qc.clbits.index(updates)
                neg_idx = clbit_idx - len(qc.clbits)
                new_data[idx] = neg_idx
                
                result = MeasurementArray(qc, new_data, parity_handles)
                insert_outvalues(eqn, context_dic, result)
                return
            
            elif isinstance(updates, ParityHandle):
                # Operand is not a MeasurementArray but updates is a ParityHandle
                # This happens when building an array from parity results in a loop
                
                # Get the shape of the operand
                if hasattr(operand, 'shape'):
                    size = int(np.prod(operand.shape))
                elif hasattr(operand, '__len__'):
                    size = len(operand)
                else:
                    size = 1
                
                # Initialize MeasurementArray with zeros
                new_data = np.zeros(size, dtype=np.int64)
                
                # Set the parity handle at the given index
                idx = get_scatter_index(indices)
                new_data[idx] = updates.index + MeasurementArray.PARITY_HANDLE_OFFSET
                
                result = MeasurementArray(qc, new_data, parity_handles)
                insert_outvalues(eqn, context_dic, result)
                return
            
            elif isinstance(updates, ProcessedMeasurement):
                # Operand is not a MeasurementArray but updates is ProcessedMeasurement
                insert_outvalues(eqn, context_dic, ProcessedMeasurement())
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
            result = handle_classical_processing(qc, invalues)
            if result is not None:
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name == "not":
            # Bitwise NOT - handled separately because it's a common operation
            result = handle_classical_processing(qc, invalues)
            if result is not None:
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name in ["and", "or", "xor"]:
            # Bitwise operations on boolean arrays
            result = handle_classical_processing(qc, invalues)
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
    
    # List to track parity operations during extraction (not attached to qc)
    parity_handles = []
    
    ammended_args = list(args) + [qc]
    
    if len(ammended_args) != len(jaspr.invars):
        raise Exception(
            "Supplied invalid number of arguments to Jaspr.to_qc "
            "(please exclude any static arguments, in particular callables)"
        )

    res = eval_jaxpr(
        jaspr, 
        eqn_evaluator=make_qc_extraction_eqn_evaluator(qc, parity_handles)
    )(*ammended_args)

    # Resolve MeasurementArrays to numpy arrays with dtype=object
    res = resolve_measurement_arrays(res)

    return res
