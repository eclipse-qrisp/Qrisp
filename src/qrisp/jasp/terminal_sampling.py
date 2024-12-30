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

import numpy as np

import jax.numpy as jnp

from qrisp.jasp.tracing_logic import qache
from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.interpreter_tools import extract_invalues, insert_outvalues, eval_jaxpr

def terminal_sampling(func = None, shots = None):
    """
    The ``terminal_sampling`` decorator is a performs a hybrid simulation and afterwards
    samples from the resulting quantum state.
    The idea behind this function is that it is very cheap for a classical simulator
    to sample from a given quantum state without simulating the whole state from
    scratch. For quantum simulators that simulate pure quantum computations
    (i.e. no classical steps) this is very established and usually achieved through
    a "shots" keyword. For hybrid simulators (like Jasp) it is not so straightforward 
    because mid-circuit measurements can alter the classical computation.
    
    In general, generating N samples from a hybrid program requires N executions
    of said programm. If it is however known that the quantum state is the same
    regardless of mid-circuit measurement outcomes, we can use the terminal sampling
    function. If this condition is not met, the ``terminal_sampling`` function
    will not return a valid distribution. A demonstration for this is given in the
    examples section.

    To use the terminal sampling decorator, a Jasp-compatible function returning
    some QuantumVariables has to be given as a parameter.
    
    Parameters
    ----------
    func : callable
        A Jasp compatible function returning QuantumVariables.
    shots : int, optional
        An integer specifying the amount of shots. The default is None, which 
        will result in probabilities being returned.

    Returns
    -------
    callable
        A function that returns a dictionary of measurement results similar to
        :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`.

    Examples
    --------
    
    We sample from a :ref:`QuantumFloat` that has been brought in a superposition.
    
    ::
        
        from qrisp import QuantumFloat, QuantumBool, h
        from qrisp.jasp import terminal_sampling

        @terminal_sampling(shots = 1000)
        def main(i):
            qf = QuantumFloat(8)
            qbl = QuantumBool()
            h(qf[i])
            cx(qf[i], qbl[0])
            return qf, qbl

        sampling_function = terminal_sampling(main, shots = 1000)
        
        print(main(0))
        print(main(1))
        print(main(2))

        # Yields:        
        {(1.0, True): 526, (0.0, False): 474}
        {(2.0, True): 503, (0.0, False): 497}
        {(4.0, True): 502, (0.0, False): 498}    
        
    **Example of invalid use**
    
    In this example we demonstrate a hybrid program that can not be properly sample
    via ``terminal_sampling``. The key ingredient here is a realtime component.
    
    ::
        
        from qrisp import QuantumBool, measure, control
        
        @terminal_sampling
        def main():
            
            qbl = QuantumBool()
            qf = QuantumFloat(4)
            
            # Bring qbl into superposition
            h(qbl)
            
            # Perform a measure
            cl_bl = measure(qbl)
            
            # Perform a conditional operation based on the measurement outcome
            with control(cl_bl):
                qf[:] = 1
                h(qf[2])
            
            return qf
        
        print(main())
        # Yields either {0.0: 1.0} or {1.0: 0.5, 5.0: 0.5} (with a 50/50 probability)
        
    The problem here is the fact that the distribution of the returned QuantumFloat
    is depending on the measurement outcome of the :ref:`QuantumBool`. The 
    ``terminal_sampling`` function performs this simulation (including the measurement)
    only once and simply samples from the final distribution.
    """
    
    if isinstance(func, int):
        shots = func
        func = None
    
    if func is None:
        return lambda x : terminal_sampling(x, shots)
    
    def return_function(*args, **kwargs):
        from qrisp.core import measure, QuantumVariable
        # The idea for realizing this function is to create an ammended function 
        # which first traces the user given function and then performs two steps:
        
        # 1. The measurements of the relevant DynamicQuantumArrays are performed.
        # 2. The decoding is performed.
        
        # These steps are traced into a Jasp, which is then evaluated with a
        # modified interpreter. This interpreter performs the usual simulation
        # logic but for the two final steps implements custom logic to make use
        # of the powerful sampling features of the Qrisp simulator infrastructure.
        
        
        # This function performs the described steps
        def ammended_function(*args, **kwargs):
            
            # Trace the user given function
            qv_tuple = func(*args, **kwargs)
            
            # Turn the result (if necessary) into a tuple
            if not isinstance(qv_tuple, tuple):
                qv_tuple = (qv_tuple,)
            
            for qv in qv_tuple:
                if not isinstance(qv, QuantumVariable):
                    raise Exception("Terminal sampling function didn't return QuantumVariables")
            
            # Trace the DynamicQubitArray measurements
            @qache
            def terminal_sampling_helper_1(*args):
                res_list = []
                for reg in args:
                    res_list.append(measure(reg))
                return tuple(res_list)
            
            measurement_ints = terminal_sampling_helper_1(*[qv.reg for qv in qv_tuple])
            
            # Trace the decoding
            @qache
            def terminal_sampling_helper_2(*meas_tuples):
                res_list = []
                for i in range(len(qv_tuple)):
                    res_list.append(meas_tuples[i+len(qv_tuple)].decoder(meas_tuples[i]))
                return tuple(res_list)
            decoded_values = terminal_sampling_helper_2(*(list(measurement_ints) + list(qv_tuple)))
            
            
            return decoded_values
        
        # Make the jaspr and flatten the environments
        jaspr = make_jaspr(ammended_function)(*args, **kwargs)
        flattened_jaspr = jaspr.flatten_environments()
        
        # This dictionary will contain the integer measurement resuts
        meas_res_dic = {}
        
        # This list will contain information about how many qubits are contained
        # in each QuantumVariable
        return_signature = []
        
        # This dictionary will contain the decoded result of the terminal_sampling
        decoded_meas_res_dic = {}
        
        # Set up the custom interpreter
        def eqn_evaluator(eqn, context_dic):
            
            # Implement custom logic for the last two steps
            if eqn.primitive.name == "pjit":
                invalues = extract_invalues(eqn, context_dic)
                
                
                if eqn.params["name"] == "terminal_sampling_helper_1":
                    
                    # Collect the qubits to be measured into a single list
                    qubits = []
                    for i in range(1, len(invalues)):
                        qubits.extend(invalues[i])
                        return_signature.append(len(invalues[i]))
                        
                    # Clear the buffer and copy the state to evaluate the measurement
                    invalues[0].apply_buffer()
                    quantum_state = invalues[0].copy()
                    
                    # Evaluate the measurement
                    # This returns a dictionary of the form {label int : count int}
                    # if shots is an int. If shots is None, count int instead is a
                    # float representing the probability.
                    
                    meas_res_dic.update(quantum_state.multi_measure(qubits, shots))

                    if shots is None:
                        # Round to prevent floating point errors of the simulation                    
                        norm = 0
                        for k, v in meas_res_dic.items():
                            meas_res_dic[k] = np.round(v, decimals = 8)
                            norm += meas_res_dic[k]
                        
                        for k, v in meas_res_dic.items():
                            meas_res_dic[k] = v/norm
                
                # Each int in the values of meas_res_dic represents the values
                # of all QuantumVariables. We therefore need to "split" the ints
                # into the appropriate parts and decode them.
                # Splitting means turning the int "1001001" into "100" and "1001".
                if eqn.params["name"] == "terminal_sampling_helper_2":
                    
                    # Iterate through the sampled values
                    for k, v in meas_res_dic.items():
                        
                        # We now evaluate the function that was previously traced
                        # to perform the decoding. The first few arguments of this
                        # function are the integers to be decoded.
                        
                        # We therefore use the traced Jaspr to perform the decoding
                        # by modifying the input values.
                        new_invalues = list(invalues)
                        
                        j = 0
                        for i in range(len(return_signature)):
                            # Split the integers into intervals ranging from 
                            # j to j + return_signature[i]
                            new_invalues[1+i] = (k & ((2**(return_signature[i])-1) << j))>>j
                            j += return_signature[i]
                        
                        # Evaluate the decoder
                        outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*new_invalues)
                        
                        # We now build the key vor the result dic
                        # For that we turn the jax types into the corresponding
                        # Python types.
                        
                        if len(outvalues) == 2:
                            key = outvalues[1]
                            if key.dtype == jnp.int32:
                                key = int(key)
                            elif key.dtype == jnp.float32:
                                key = float(key)
                            elif key.dtype == jnp.bool:
                                key = bool(key)
                            
                            decoded_meas_res_dic[key] = v
                        # If the user given function returned more than one
                        # value, the key is a tuple to be build up
                        else:
                            key_list = []
                            
                            for i in range(1, len(outvalues)):
                                key = outvalues[i]
                                if key.dtype == jnp.int32:
                                    key = int(key)
                                elif key.dtype == jnp.float32:
                                    key = float(key)
                                elif key.dtype == jnp.bool:
                                    key = bool(key)
                                key_list.append(key)        
                            
                            decoded_meas_res_dic[tuple(key_list)] = v
                            
                
                # Performs the default logic for simulating a Jaspr
                outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*invalues)
                if not isinstance(outvalues, (list, tuple)):
                    outvalues = [outvalues]
                insert_outvalues(eqn, context_dic, outvalues)
            else:
                return True
        
        # Initiate the simulation
        from qrisp.simulator import BufferedQuantumState
        args = [BufferedQuantumState()] + list(args)
        eval_jaxpr(flattened_jaspr, eqn_evaluator = eqn_evaluator)(*args)
    
        # Sort counts_list such the most probable values come first
        decoded_meas_res_dic = dict(sorted(decoded_meas_res_dic.items(), key=lambda item: item[0]))
        decoded_meas_res_dic = dict(sorted(decoded_meas_res_dic.items(), key=lambda item: -item[1]))
    
        return decoded_meas_res_dic
    
    return return_function
