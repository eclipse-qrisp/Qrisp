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
       - Negative values: Measurement results (indices into qc.clbits)
         * -1 corresponds to the last clbit in the circuit
         * -2 corresponds to the second-to-last clbit
         * etc.
    
    2. **Transparent Extraction**: When extracting a single element that
       corresponds to a measurement, we return the actual Clbit object.
       This allows the extracted value to be used in subsequent operations
       (like controlled gates or parity primitives) that expect Clbit inputs.
    
    3. **Array Operations**: We intercept JAX array primitives and implement
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
    
    Attributes
    ----------
    qc : QuantumCircuit
        Reference to the quantum circuit being built. Needed to resolve
        negative indices back to actual Clbit objects.
    data : numpy.ndarray
        Array of integers encoding the boolean/measurement values.
    """
    
    def __init__(self, qc, data):
        """
        Initialize a MeasurementArray.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit being built.
        data : array-like
            Array of integers encoding boolean/measurement values.
        """
        self.qc = qc
        self.data = np.array(data, dtype=np.int64)
    
    def __len__(self):
        return len(self.data)
    
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
        Clbit, bool, or MeasurementArray
            - If extracting a single measurement element: returns the Clbit
            - If extracting a single known boolean: returns that boolean
            - If extracting a slice: returns a new MeasurementArray
        """
        if isinstance(key, (int, np.integer)):
            val = self.data[key]
            if val < 0:
                # Negative value indicates a measurement result.
                # The value directly serves as a negative index into qc.clbits.
                clbit_index = int(val)
                return self.qc.clbits[clbit_index]
            else:
                # Non-negative value is a known boolean (0 = False, 1 = True)
                return bool(val)
        elif isinstance(key, slice):
            return MeasurementArray(self.qc, self.data[key])
        else:
            raise TypeError(
                f"MeasurementArray indices must be integers or slices, not {type(key)}"
            )
    
    @classmethod
    def from_clbit(cls, qc, clbit):
        """
        Create a single-element MeasurementArray from a Clbit.
        
        This is used when a scalar Clbit needs to be broadcast into an array
        shape (e.g., via broadcast_in_dim).
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit containing the clbit.
        clbit : Clbit
            The classical bit to encode.
        
        Returns
        -------
        MeasurementArray
            Single-element array containing the encoded clbit reference.
        """
        # Find position of this clbit and convert to negative index
        clbit_idx = qc.clbits.index(clbit)
        neg_idx = clbit_idx - len(qc.clbits)
        return cls(qc, np.array([neg_idx], dtype=np.int64))
    
    @classmethod
    def from_value(cls, qc, value):
        """
        Create a single-element MeasurementArray from a known boolean value.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit (needed for consistency, not used for encoding).
        value : bool
            The boolean value to encode.
        
        Returns
        -------
        MeasurementArray
            Single-element array containing 0 (False) or 1 (True).
        """
        return cls(qc, np.array([int(bool(value))], dtype=np.int64))
    
    @classmethod
    def concatenate(cls, qc, arrays):
        """
        Concatenate multiple arrays/values into a single MeasurementArray.
        
        This handles the JAX `concatenate` primitive when any of the inputs
        contains measurement data.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit being built.
        arrays : list
            List of values to concatenate. Each can be:
            - MeasurementArray: data is extended directly
            - Clbit: encoded as negative index
            - bool/int: encoded as 0 or 1
        
        Returns
        -------
        MeasurementArray
            Concatenated array.
        """
        from qrisp import Clbit
        
        result_data = []
        for arr in arrays:
            if isinstance(arr, MeasurementArray):
                result_data.extend(arr.data)
            elif isinstance(arr, Clbit):
                clbit_idx = qc.clbits.index(arr)
                neg_idx = clbit_idx - len(qc.clbits)
                result_data.append(neg_idx)
            elif isinstance(arr, (bool, np.bool_)):
                result_data.append(int(arr))
            elif isinstance(arr, (int, np.integer)):
                result_data.append(int(arr))
            else:
                raise TypeError(
                    f"Cannot concatenate type {type(arr)} into MeasurementArray"
                )
        
        return cls(qc, np.array(result_data, dtype=np.int64))


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
            pjit_to_gate, 
            cond_to_cl_control
        )
        
        invalues = extract_invalues(eqn, context_dic)
        prim_name = eqn.primitive.name
        
        # -----------------------------------------------------------------
        # SECTION 4.1: Control Flow and Structural Primitives
        # -----------------------------------------------------------------
        
        if prim_name == "jit" and isinstance(eqn.params["jaxpr"], Jaspr):
            # Nested Jaspr (from @qache or similar) - convert to gate
            return pjit_to_gate(eqn, context_dic, qc_extraction_eqn_evaluator)
        
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
            # This is implemented as a classical operation in the circuit
            res = Clbit("cb_" + str(len(qc.clbits)))
            qc.clbits.insert(0, res)
            qc.append(ParityOperation(len(invalues)), clbits=invalues + [res])
            insert_outvalues(eqn, context_dic, res)
            return
        
        # -----------------------------------------------------------------
        # SECTION 4.3: Type Conversion (convert_element_type)
        # -----------------------------------------------------------------
        # JAX often inserts type conversions. For measurement data, we just
        # pass through the value unchanged since we're not doing real computation.
        
        elif prim_name == "convert_element_type":
            inval = context_dic[eqn.invars[0]]
            if isinstance(inval, (ProcessedMeasurement, Clbit, MeasurementArray)):
                # Measurement data passes through unchanged
                context_dic[eqn.outvars[0]] = inval
                return
            elif isinstance(inval, list) and len(inval) and isinstance(
                inval[0], (ProcessedMeasurement, Clbit)
            ):
                # List of measurement data passes through unchanged
                context_dic[eqn.outvars[0]] = inval
                return
            return True
        
        # -----------------------------------------------------------------
        # SECTION 4.4: Array Operations on Measurement Data
        # -----------------------------------------------------------------
        # These primitives handle JAX array operations when the arrays contain
        # measurement results. We use MeasurementArray to track Clbit references
        # through these operations.
        
        elif prim_name == "broadcast_in_dim":
            # broadcast_in_dim: Expands a scalar to an array shape
            # Example: jnp.array([m]) where m is a scalar measurement
            # JAX traces this as: broadcast_in_dim(m, shape=(1,))
            
            inval = invalues[0]
            shape = eqn.params["shape"]
            
            if isinstance(inval, Clbit):
                meas_arr = MeasurementArray.from_clbit(qc, inval)
                new_data = np.broadcast_to(meas_arr.data, shape)
                result = MeasurementArray(qc, new_data.flatten())
                insert_outvalues(eqn, context_dic, result)
                return
            elif isinstance(inval, MeasurementArray):
                new_data = np.broadcast_to(inval.data, shape)
                result = MeasurementArray(qc, new_data.flatten())
                insert_outvalues(eqn, context_dic, result)
                return
            elif isinstance(inval, (bool, np.bool_)):
                meas_arr = MeasurementArray.from_value(qc, inval)
                new_data = np.broadcast_to(meas_arr.data, shape)
                result = MeasurementArray(qc, new_data.flatten())
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name == "concatenate":
            # concatenate: Joins multiple arrays along an axis
            # Example: jnp.array([m0, m1, m2]) compiles to:
            #   broadcast_in_dim(m0) -> arr0
            #   broadcast_in_dim(m1) -> arr1
            #   broadcast_in_dim(m2) -> arr2
            #   concatenate(arr0, arr1, arr2)
            
            if any(contains_measurement_data(v) for v in invalues):
                result = MeasurementArray.concatenate(qc, invalues)
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name == "squeeze":
            # squeeze: Removes dimensions of size 1
            # Example: arr[0] on a 1D array often compiles to slice + squeeze
            # The slice extracts shape (1,), squeeze reduces to scalar
            
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                if len(inval.data) == 1:
                    # Single element - extract and return the actual value
                    # This is where we bridge back to Clbit!
                    result = inval[0]
                else:
                    result = inval
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name == "slice":
            # slice: Static slicing with known start/stop indices
            # Example: arr[0:2] or arr[i] where i is a constant
            
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                start_indices = eqn.params["start_indices"]
                limit_indices = eqn.params["limit_indices"]
                start = start_indices[0] if start_indices else 0
                stop = limit_indices[0] if limit_indices else len(inval)
                result = inval[start:stop]
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name == "dynamic_slice":
            # dynamic_slice: Slicing with runtime-determined start index
            # Example: arr[i] inside a loop where i is a loop variable
            
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                start_idx = int(invalues[1])
                slice_size = eqn.params["slice_sizes"][0]
                result = MeasurementArray(
                    qc, inval.data[start_idx:start_idx + slice_size]
                )
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        elif prim_name == "gather":
            # gather: General indexing operation (covers various indexing patterns)
            
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                indices = invalues[1]
                if hasattr(indices, 'item'):
                    idx = int(indices.item())
                else:
                    idx = int(indices)
                result = inval[idx]
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # -----------------------------------------------------------------
        # SECTION 4.5: Default Handling
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
        # SECTION 4.6: Fallback for Unhandled Measurement Operations
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

    return res
