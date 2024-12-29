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


import jax.numpy as jnp

from qrisp.jasp.tracing_logic import qache
from qrisp.jasp.jasp_expression import make_jaspr
from qrisp.jasp.interpreter_tools import extract_invalues, insert_outvalues, eval_jaxpr

def terminal_sampling(func, shots = None):
    """
    The ``terminal_sampling`` function performs a hybrid simulation and afterwards
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

    To use the terminal sampling function, a Jasp-compatible function returning
    a QuantumVariable has to be given as a parameter.

    Parameters
    ----------
    func : callable
        A Jasp compatible function returning a QuantumVariable.
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
        
        from qrisp import QuantumFloat, h
        from qrisp.jasp import terminal_sampling

        def main(i):
            qf = QuantumFloat(8)
            h(qf[i])
            return qf

        sampling_function = terminal_sampling(main, shots = 1000)
        
        print(sampling_function(0))
        print(sampling_function(1))
        print(sampling_function(2))

        # Yields:        
        # {0.0: 516, 1.0: 484}
        # {0.0: 527, 2.0: 473}
        # {0.0: 498, 4.0: 502}
    
    **Example of invalid use**
    
    In this example we demonstrate a hybrid program that can not be properly sample
    via ``terminal_sampling``. The key ingredient here is a realtime component.
    
    ::
        
        from qrisp import QuantumBool, measure, control
        
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
        
        print(terminal_sampling(main)())
        # Yields either {0.0: 1.0} or {1.0: 0.5, 5.0: 0.5} (with a 50/50 probability)
        
    The problem here is the fact that the distribution of the returned QuantumFloat
    is depending on the measurement outcome of the :ref:`QuantumBool`. The 
    ``terminal_sampling`` function performs this simulation (including the measurement)
    only once and simply samples from the final distribution.
    """
    
    def return_function(*args, **kwargs):
        from qrisp.core import measure
        
        
        def ammended_function(*args, **kwargs):
            qv_tuple = func(*args, **kwargs)
            
            if not isinstance(qv_tuple, tuple):
                qv_tuple = (qv_tuple,)
                
            @qache
            def terminal_sampling_helper_1(*args):
                res_list = []
                for reg in args:
                    res_list.append(measure(reg))
                return tuple(res_list)
            
            measurement_ints = terminal_sampling_helper_1(*[qv.reg for qv in qv_tuple])
            
            @qache
            def terminal_sampling_helper_2(*meas_tuples):
                res_list = []
                for i in range(len(qv_tuple)):
                    res_list.append(meas_tuples[i+len(qv_tuple)].decoder(meas_tuples[i]))
                return tuple(res_list)
            
            decoded_values = terminal_sampling_helper_2(*(list(measurement_ints) + list(qv_tuple)))
            return decoded_values
        
        jaspr = make_jaspr(ammended_function)(*args, **kwargs)
        flattened_jaspr = jaspr.flatten_environments()
        
        meas_res_dic = {}
        return_signature = []
        decoded_meas_res_dic = {}
        
        def eqn_evaluator(eqn, context_dic):
            
            if eqn.primitive.name == "pjit":
                invalues = extract_invalues(eqn, context_dic)
                
                if eqn.params["name"] == "terminal_sampling_helper_1":
                    invalues[0].apply_buffer()
                    quantum_state = invalues[0].copy()
                    qubits = []
                    for i in range(1, len(invalues)):
                        qubits.extend(invalues[i])
                        return_signature.append(len(invalues[i]))
                    
                    meas_res_dic.update(quantum_state.multi_measure(qubits, shots))
                
                if eqn.params["name"] == "terminal_sampling_helper_2":
                    
                    for k, v in meas_res_dic.items():
                        new_invalues = list(invalues)
                        
                        j = 0
                        for i in range(len(return_signature)):
                            new_invalues[1+i] = (k & ((2**(return_signature[i])-1) << j))>>j
                            j += return_signature[i]
                        
                        outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*new_invalues)
                        
                        if len(outvalues) == 2:
                            key = outvalues[1]
                            if key.dtype == jnp.int32:
                                key = int(key)
                            elif key.dtype == jnp.float32:
                                key = float(key)
                            elif key.dtype == jnp.bool:
                                key = bool(key)
                            
                            decoded_meas_res_dic[key] = v
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
                            
                    
                outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*invalues)
                if not isinstance(outvalues, (list, tuple)):
                    outvalues = [outvalues]
                insert_outvalues(eqn, context_dic, outvalues)
            else:
                return True
            
        from qrisp.simulator import BufferedQuantumState
        args = [BufferedQuantumState()] + list(args)
        eval_jaxpr(flattened_jaspr, eqn_evaluator = eqn_evaluator)(*args)
    
        return decoded_meas_res_dic
    
    return return_function
