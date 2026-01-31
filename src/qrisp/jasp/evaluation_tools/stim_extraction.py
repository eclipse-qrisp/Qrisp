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
"""

from qrisp.jasp import make_jaspr
from qrisp.circuit import Clbit
from qrisp.jasp.interpreter_tools.interpreters.qc_extraction_interpreter import ParityHandle

def extract_stim(func):
    """
    Decorator that extracts a Stim circuit from a Jasp-traceable function.
    
    This decorator enables high-performance Clifford circuit simulation by converting
    Jasp-traceable Qrisp functions into Stim circuits. It handles the translation of
    quantum measurements into Stim's measurement record indices, allowing you to map
    simulation results back to the original function's return values.
    
    .. note::
        
        Stim only supports Clifford operations. Functions containing non-Clifford
        gates (e.g., T, RZ, RX) will raise an error during conversion.
    
    .. warning::
        
        **Measurement post-processing limitation:** Advanced quantum types like 
        :ref:`QuantumFloat` apply post-processing to raw measurement results during 
        decoding. For example, a QuantumFloat might convert a raw integer into a fractional 
        value. This post-processing cannot be performed during the Stim extraction 
        because it requires transforming a list of classical bits in ways that involve
        classical post-processing steps that can not be represented by the Stim circuit
        format.
        
        For this reason, it is **recommended to use** :ref:`QuantumVariable` **instead 
        of QuantumFloat** (or similar advanced types) when working with ``extract_stim``. 
        QuantumVariable's decoder returns raw integer values without post-processing, 
        making it fully compatible with Stim's measurement record. You can then apply 
        any necessary transformations manually after sampling. Values that have
        been processed in this way are represented through the ``ProcessedMeasurement``
        class, which acts as a dummy representative.
        
        See the :meth:`Jaspr.to_qc <qrisp.jasp.Jaspr.to_qc>` documentation for more 
        details on this limitation.
    
    Parameters
    ----------
    func : callable
        A Jasp-traceable function that manipulates quantum variables and returns
        quantum measurement results and/or classical values.
    
    Returns
    -------
    callable
        A decorated function that returns:
        
        - **Single return value:** If `func` returns nothing, returns just the 
          `stim.Circuit` object.
        - **Multiple return values:** If `func` returns n values, returns a tuple 
          of (n+1) elements:
          
          * Elements 0 to n-1: The function's return values, where:
            
            - **Classical values** (integers, floats, etc.) are returned as-is.
            - **Quantum measurements** (measured QuantumVariables) are returned as 
              integers or tuples of integers indicating which measurement positions 
              in the Stim circuit correspond to this return value.
          
          * Element n: The `stim.Circuit` object.
    
    Examples
    --------
    
    **Example 1: Single return value**
    
    When the function has no return value, only the Stim circuit is returned:
    
    ::
        
        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp import extract_stim
        
        @extract_stim
        def bell_state():
            qv = QuantumVariable(2)
            h(qv[0])
            cx(qv[0], qv[1])
            measure(qv)
        
        stim_circuit = bell_state()
        print(stim_circuit)
        # Yields:
        # H 0
        # CX 0 1
        # M 0 1
    
    **Example 2: Multiple return values with measurement indices**
    
    When returning one or more values, quantum measurements are returned as measurement 
    indices, while classical values remain unchanged:
    
    ::
        
        from qrisp import QuantumFloat, h, cx, measure
        from qrisp.jasp import extract_stim
        
        @extract_stim
        def analyze_state(n):
            qf = QuantumVariable(n)
            h(qf)
            
            # Mid-circuit measurement
            first_qubit_result = measure(qf[0])
            
            # Classical computation
            classical_value = n * 2
            
            # Final measurement
            final_result = measure(qf)
            
            return classical_value, first_qubit_result, final_result
        
        classical_val, first_meas_idx, final_meas_indices, stim_circuit = analyze_state(3)
        
        print(f"Classical value: {classical_val}")  # 6 (unchanged)
        print(f"First qubit measurement index: {first_meas_idx}")  # e.g., 0
        print(f"Final measurement indices: {final_meas_indices}")  # e.g., (1, 2, 3)
    
    **Example 3: Sampling and slicing results**
    
    Use the measurement indices to extract specific results from Stim's samples:
    
    ::
        
        @extract_stim
        def prepare_entangled_state():
            qf1 = QuantumVariable(2)
            qf2 = QuantumVariable(3)
            
            # Prepare qf1 in superposition
            h(qf1)
            
            # Entangle qf2 with qf1[0]
            for i in range(3):
                cx(qf1[0], qf2[i])
            
            result1 = measure(qf1)
            result2 = measure(qf2)
            
            return result1, result2

        # Extract the circuit and measurement indices
        qf1_indices, qf2_indices, stim_circuit = prepare_entangled_state()

        print(f"qf1 measured at positions: {qf1_indices}")  # [0, 1]
        print(f"qf2 measured at positions: {qf2_indices}")  # [2, 3, 4]

        # Sample 1000 shots from the Stim circuit
        sampler = stim_circuit.compile_sampler()
        all_samples = sampler.sample(1000)  # Shape: (1000, 5) - 5 total measurements

        # Slice the samples to get only the results for each return value
        qf1_samples = all_samples[:, qf1_indices]  # Shape: (1000, 2)
        qf2_samples = all_samples[:, qf2_indices]  # Shape: (1000, 3)

        # Convert bit arrays to integers (little-endian)
        import numpy as np
        qf1_values = qf1_samples.dot(1 << np.arange(qf1_samples.shape[1]))
        qf2_values = qf2_samples.dot(1 << np.arange(qf2_samples.shape[1]))

        print(f"First 10 qf1 values: {qf1_values[:10]}")
        print(f"First 10 qf2 values: {qf2_values[:10]}")

        # Verify entanglement: when qf1[0]=0, all qf2 bits should be 0
        qf1_first_bit = qf1_samples[:, 0]
        assert np.all(qf2_samples[qf1_first_bit == 0] == 0)

    **Example 4: Using parity checks (Detectors)**

    You can use the :func:`~qrisp.jasp.parity` function to define parity checks within your circuit.
    When extracted to Stim, these are converted into ``DETECTOR`` instructions.

    ::

        from qrisp import QuantumVariable, h, cx, measure
        from qrisp.jasp import extract_stim, parity
        from qrisp.misc.stim_tools import stim_noise
        import stim

        @extract_stim
        def selective_noise_demo():
            # Create two QuantumVariables for independent Bell pairs
            bell_pair_1 = QuantumVariable(2)
            bell_pair_2 = QuantumVariable(2)
            
            h(bell_pair_1[0]); cx(bell_pair_1[0], bell_pair_1[1])
            h(bell_pair_2[0]); cx(bell_pair_2[0], bell_pair_2[1])
            
            # Apply deterministic X error to one of the qubits in the second pair
            stim_noise("X_ERROR", 1.0, bell_pair_2[0])
            
            m1_0 = measure(bell_pair_1[0]); m1_1 = measure(bell_pair_1[1])
            m2_0 = measure(bell_pair_2[0]); m2_1 = measure(bell_pair_2[1])
            
            # Detector 1: expectation=False implies we expect even parity
            d1 = parity(m1_0, m1_1, expectation=False)
            
            # Detector 2: Checks parity of second, noisy pair
            d2 = parity(m2_0, m2_1, expectation=False)
            
            return d1, d2

        d1, d2, stim_circuit = selective_noise_demo()
        sampler = stim_circuit.compile_detector_sampler()
        print(sampler.sample(1))
        # Yields [[False, True]] (False = no error/match, True = error/mismatch)

    **Example 5: Defining Observables**

    Similarly, :func:`~qrisp.jasp.parity` with ``expectation=None`` defines logical observables in Stim.

    ::

        @extract_stim
        def observable_demo():
            qv = QuantumVariable(2)
            h(qv)
            m0 = measure(qv[0]); m1 = measure(qv[1])
            
            # Define an observable O = Z_0 Z_1
            logical_obs = parity(m0, m1)
            return logical_obs

        obs_idx, stim_circuit = observable_demo()
        # stim_circuit contains OBSERVABLE_INCLUDE(0) ...

    """
    
    def return_func(*args):
        """
        Inner function that performs the actual Stim extraction.
        
        This function implements the conversion pipeline:
        1. Creates a Jaspr (Jax-like representation) from the traced function
        2. Converts the Jaspr to a QuantumCircuit via staticalization
        3. Extracts the Stim circuit from the QuantumCircuit
        4. Maps Qrisp Clbit objects to Stim measurement record indices
        """
        
        # Step 1: Create a Jaspr from the function with the given arguments
        # The Jaspr is a traced representation of the quantum program, similar to Jax's jaxpr.
        # This tracing process records the quantum operations without actually executing them.
        jaspr = make_jaspr(func)(*args)
        
        # Step 2: Convert the Jaspr to a QuantumCircuit
        # The "staticalization" process converts the traced representation back to a concrete
        # quantum circuit. The to_qc method returns a tuple containing:
        # - All return values from the original function (e.g., measured values, QuantumVariables)
        # - The QuantumCircuit object (always the last element)
        # So if the original function returns n values, staticalization_result has (n+1) elements.
        staticalization_result = jaspr.to_qc(*args)
        
        # Handle the simple case: function returns no value (i.e. the staticalization returns just a qc)
        if len(jaspr.outvars) == 1:
            # For single return values, staticalization_result is just the QuantumCircuit.
            # Convert it directly to Stim without needing to track the clbit mapping.
            return staticalization_result.to_stim()
        
        # Handle the complex case: function returns multiple values
        else:
            # Extract the QuantumCircuit (always the last element of the tuple)
            qc = staticalization_result[-1]
            
            # Convert the QuantumCircuit to Stim with mapping enabled.
            # - clbit_mapping: maps Clbit objects to Stim measurement record indices
            # - detector_mapping: maps parity record indices to Stim detector indices
            # - observable_mapping: maps parity record indices to Stim observable indices
            # We merge these into idx_mapping for unified lookup.
            stim_circ, clbit_mapping, detector_mapping, observable_mapping = qc.to_stim(return_measurement_map=True, return_detector_map=True, return_observable_map = True)
            idx_mapping = {**detector_mapping, **clbit_mapping, **observable_mapping}
            
            # Process all return values except the QuantumCircuit (last element)
            # We need to replace any Clbit objects with their corresponding Stim indices.
            new_result = []
            for i in range(len(staticalization_result)-1):
                
                val = staticalization_result[i]
                
                # Case 1: Value is a list of Clbit objects (e.g., from multi-qubit measurement)
                # Replace each Clbit with its corresponding Stim measurement index.
                # This happens when a QuantumFloat/QuantumVariable is measured and returns
                # a list of classical bits representing the measurement results.
                if isinstance(val, list) and len(val) and isinstance(val[0], Clbit):
                    new_val = tuple(idx_mapping[clbit] for clbit in val)
                    if len(new_val) == 1:
                        new_val = new_val[0]
                
                # Case 2: Value is a single Clbit object
                # Replace it with its Stim measurement index.
                # This happens when a single qubit is measured.
                elif isinstance(val, Clbit):
                    new_val = idx_mapping[val]
                
                # Case 3: Value is a ParityHandle (from parity operation)
                # Replace it with its Stim detector/observable index from the mapping.
                elif isinstance(val, ParityHandle):
                    new_val = idx_mapping[val.index]
                
                # Case 4: Value is something else (e.g., integer, float, ProcessedMeasurement)
                # Pass through unchanged. Classical values computed during the function
                # (not involving measurements) are returned as-is.
                else:
                    new_val = val
                    
                new_result.append(new_val)
            
            # Append the Stim circuit as the last element of the result tuple
            new_result.append(stim_circ)
        
            # Return all processed values plus the Stim circuit as a tuple
            return tuple(new_result)
    
    return return_func
