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

import numpy as np


class ProcessedMeasurement:
    """
    Placeholder object used to represent the result of classical post-processing
    of measurement results in the QuantumCircuit representation.
    
    Since QuantumCircuit objects cannot represent classical computation on measurement
    results, this class serves as a placeholder to emulate the computation without
    actually performing it.
    """
    pass


class MeasurementArray:
    """
    Represents an array of boolean values (possibly measurement results) during
    QuantumCircuit extraction.
    
    This class is used to handle JAX array operations on measurement results when
    lowering Jaspr to QuantumCircuit. Since Clbit objects are not JAX-compatible,
    we represent arrays containing measurements using integer encoding.
    
    Attributes
    ----------
    qc : QuantumCircuit
        Reference to the quantum circuit being built.
    data : numpy.ndarray
        Array of integers where:
        - 0 means known boolean value False
        - 1 means known boolean value True
        - Negative values indicate measurement results: -1 corresponds to the
          last classical bit in the quantum circuit, -2 to the second last, etc.
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
        
        Returns
        -------
        Clbit, bool, or MeasurementArray
            If extracting a single element that is a measurement, returns the Clbit.
            If extracting a single known boolean, returns that boolean.
            If extracting a slice, returns a new MeasurementArray.
        """
        from qrisp import Clbit
        
        if isinstance(key, (int, np.integer)):
            val = self.data[key]
            if val < 0:
                # Negative index means measurement result
                # -1 -> last clbit (index -1), -2 -> second last, etc.
                clbit_index = int(val)  # Already negative
                return self.qc.clbits[clbit_index]
            else:
                # Known boolean value
                return bool(val)
        elif isinstance(key, slice):
            return MeasurementArray(self.qc, self.data[key])
        else:
            raise TypeError(f"MeasurementArray indices must be integers or slices, not {type(key)}")
    
    @classmethod
    def from_clbit(cls, qc, clbit):
        """
        Create a MeasurementArray from a single Clbit.
        
        The Clbit is encoded as a negative index based on its position in qc.clbits.
        """
        # Find the index of this clbit in the circuit's clbit list
        # We use negative indexing: -1 for last, -2 for second last, etc.
        clbit_idx = qc.clbits.index(clbit)
        # Convert to negative index from end
        neg_idx = clbit_idx - len(qc.clbits)
        return cls(qc, np.array([neg_idx], dtype=np.int64))
    
    @classmethod
    def from_value(cls, qc, value):
        """
        Create a MeasurementArray from a known boolean value.
        """
        return cls(qc, np.array([int(bool(value))], dtype=np.int64))
    
    @classmethod
    def concatenate(cls, qc, arrays):
        """
        Concatenate multiple MeasurementArrays or values into one.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit being built.
        arrays : list
            List of MeasurementArray, Clbit, or boolean values to concatenate.
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
                # Assume it's a boolean-like value (0 or 1)
                result_data.append(int(arr))
            else:
                raise TypeError(f"Cannot concatenate type {type(arr)} into MeasurementArray")
        
        return cls(qc, np.array(result_data, dtype=np.int64))


def make_qc_extraction_eqn_evaluator(qc):
    
    def contains_measurement_data(val):
        """Check if a value contains Clbit or MeasurementArray data."""
        from qrisp import Clbit
        if isinstance(val, (Clbit, MeasurementArray, ProcessedMeasurement)):
            return True
        if isinstance(val, list) and len(val):
            return contains_measurement_data(val[0])
        return False

    def qc_extraction_eqn_evaluator(eqn, context_dic):
        """
        Equation evaluator for converting a Jaspr to a QuantumCircuit.
        
        This evaluator processes Jaxpr equations and builds a QuantumCircuit representation.
        It handles quantum primitives directly and creates ProcessedMeasurement placeholders
        for classical post-processing operations.
        
        Parameters
        ----------
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : dict
            Context dictionary mapping variables to their values.
        
        Returns
        -------
        bool or None
            Returns True to indicate default evaluation should be used,
            False to skip default evaluation, or None if the equation was fully processed.
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
        from qrisp.jasp.interpreter_tools.interpreters import pjit_to_gate, cond_to_cl_control
        invalues = extract_invalues(eqn, context_dic)
        if eqn.primitive.name == "jit" and isinstance(
            eqn.params["jaxpr"], Jaspr
        ):
            return pjit_to_gate(eqn, context_dic, qc_extraction_eqn_evaluator)
        elif eqn.primitive.name == "cond":
            return cond_to_cl_control(eqn, context_dic, qc_extraction_eqn_evaluator)
        elif eqn.primitive.name == "jasp.parity":
            res = Clbit("cb_" + str(len(qc.clbits)))
            qc.clbits.insert(0, res)
            qc.append(ParityOperation(len(invalues)), clbits = invalues + [res])
            insert_outvalues(eqn, context_dic, res)
        elif eqn.primitive.name == "while":
            return True
        elif eqn.primitive.name == "convert_element_type":
            inval = context_dic[eqn.invars[0]]
            if isinstance(inval, (ProcessedMeasurement, Clbit, MeasurementArray)):
                context_dic[eqn.outvars[0]] = inval
                return
            elif isinstance(inval, list) and len(inval) and isinstance(
                inval[0], (ProcessedMeasurement, Clbit)
            ):
                context_dic[eqn.outvars[0]] = inval
                return
            return True
        
        # Handle broadcast_in_dim: scalar -> array
        elif eqn.primitive.name == "broadcast_in_dim":
            inval = invalues[0]
            shape = eqn.params["shape"]
            
            if isinstance(inval, Clbit):
                # Create MeasurementArray from Clbit and broadcast to shape
                meas_arr = MeasurementArray.from_clbit(qc, inval)
                # Broadcast: replicate the single element to fill the shape
                new_data = np.broadcast_to(meas_arr.data, shape)
                result = MeasurementArray(qc, new_data.flatten())
                insert_outvalues(eqn, context_dic, result)
                return
            elif isinstance(inval, MeasurementArray):
                # Broadcast existing MeasurementArray
                new_data = np.broadcast_to(inval.data, shape)
                result = MeasurementArray(qc, new_data.flatten())
                insert_outvalues(eqn, context_dic, result)
                return
            elif isinstance(inval, (bool, np.bool_)):
                # Create MeasurementArray from known boolean
                meas_arr = MeasurementArray.from_value(qc, inval)
                new_data = np.broadcast_to(meas_arr.data, shape)
                result = MeasurementArray(qc, new_data.flatten())
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # Handle concatenate: join multiple arrays
        elif eqn.primitive.name == "concatenate":
            # Check if any input contains measurement data
            if any(contains_measurement_data(v) for v in invalues):
                result = MeasurementArray.concatenate(qc, invalues)
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # Handle squeeze: remove dimensions of size 1
        elif eqn.primitive.name == "squeeze":
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                # For MeasurementArray, squeeze just returns the same data
                # (we're working with 1D arrays internally)
                if len(inval.data) == 1:
                    # Single element - extract it
                    result = inval[0]
                else:
                    result = inval
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # Handle slice: extract a portion of an array
        elif eqn.primitive.name == "slice":
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                start_indices = eqn.params["start_indices"]
                limit_indices = eqn.params["limit_indices"]
                # For 1D case
                start = start_indices[0] if start_indices else 0
                stop = limit_indices[0] if limit_indices else len(inval)
                result = inval[start:stop]
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # Handle dynamic_slice: dynamic indexing
        elif eqn.primitive.name == "dynamic_slice":
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                # invalues[1:] are the start indices
                start_idx = int(invalues[1])
                slice_size = eqn.params["slice_sizes"][0]
                result = MeasurementArray(qc, inval.data[start_idx:start_idx + slice_size])
                insert_outvalues(eqn, context_dic, result)
                return
            else:
                return True
        
        # Handle gather: general indexing operation
        elif eqn.primitive.name == "gather":
            inval = invalues[0]
            if isinstance(inval, MeasurementArray):
                # For simple integer indexing, gather with specific params
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
        
        else:
            for val in invalues:
                if contains_measurement_data(val):
                    break
            else:
                if isinstance(eqn.primitive, QuantumPrimitive):
                    outvalues = eqn.primitive.impl(*invalues, **eqn.params)
                    insert_outvalues(eqn, context_dic, outvalues)
                    return
                else:
                    return True
            
        if len(eqn.outvars) == 0:
            return
        elif len(eqn.outvars) == 1 and not eqn.primitive.multiple_results:
            outvalues = ProcessedMeasurement()
        elif len(eqn.outvars) >= 1:
            outvalues = [ProcessedMeasurement() for _ in range(len(eqn.outvars))]
        
        insert_outvalues(eqn, context_dic, outvalues)
        
    return qc_extraction_eqn_evaluator


def jaspr_to_qc(jaspr, *args):
    """
    Converts a Jaspr into a QuantumCircuit if applicable.
    
    Circuit conversion of algorithms involving realtime computations is not possible.
    Any computations that perform classical postprocessing of measurements
    can not be reflected within the QuantumCircuit object itself and will
    generate an object of type ``ProcessedMeasurement``. These objects hold
    no further information and are simply used as placeholders to emulate
    the computation.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr object to convert.
    *args : tuple
        The arguments to call the Jaspr with.

    Returns
    -------
    return_values : tuple
        The return values of the Jaspr. QuantumVariable return types are
        returned as lists of Qubits.
    :ref:`QuantumCircuit`
        The resulting QuantumCircuit.

    Examples
    --------

    We create a simple script and inspect the QuantumCircuit:

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
        # Yields
        # qb_0: ──■───────
        #       ┌─┴─┐┌───┐
        # qb_1: ┤ X ├┤ T ├
        #       └───┘└───┘
        
    To demonstrate the behavior under measurement post-processing, we build
    a similar script:
        
    ::
        
        from qrisp.jasp.interpreter_tools.interpreters import ProcessedMeasurement
        
        def example_function(i):
        
            qf = QuantumFloat(i)
            cx(qf[0], qf[1])
            t(qf[1])
            
            meas_res = measure(qf)
            # Perform classical post processing
            meas_res *= 2
            return meas_res
        
        jaspr = make_jaspr(example_function)(2)
        
        meas_res, qc = jaspr_to_qc(jaspr, 2)
        print(isinstance(meas_res, ProcessedMeasurement))
        # True
    """
    from qrisp import QuantumCircuit
    from qrisp.jasp import eval_jaxpr

    qc = QuantumCircuit()
    ammended_args = list(args) + [qc]
    
    if len(ammended_args) != len(jaspr.invars):
        raise Exception(
            "Supplied invalid number of arguments to Jaspr.to_qc (please exclude any static arguments, in particular callables)"
        )

    res = eval_jaxpr(jaspr, eqn_evaluator=make_qc_extraction_eqn_evaluator(qc))(*(ammended_args))

    return res
