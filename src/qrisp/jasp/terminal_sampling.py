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

def terminal_sampling(func, shots):
    
    def return_function(*args, **kwargs):
        from qrisp.core import measure
        
        @qache
        def terminal_sampling_helper_1(reg):
            return measure(reg)
        
        @qache
        def terminal_sampling_helper_2(i, qv):
            return qv.decoder(i)
        
        def ammended_function(*args, **kwargs):
            qv = func(*args, **kwargs)
            measurement_int = terminal_sampling_helper_1(qv.reg)
            decoded_value = terminal_sampling_helper_2(measurement_int, qv)
            return decoded_value
        
        jaspr = make_jaspr(ammended_function)(*args, **kwargs)
        flattened_jaspr = jaspr.flatten_environments()
        
        meas_res_dic = {}
        decoded_meas_res_dic = {}
        
        def eqn_evaluator(eqn, context_dic):
            
            if eqn.primitive.name == "pjit":
                invalues = extract_invalues(eqn, context_dic)
                
                if eqn.params["name"] == "terminal_sampling_helper_1":
                    invalues[0].apply_buffer()
                    
                    quantum_state = invalues[0].copy()
                    qubits = invalues[1]
                    
                    meas_ints, probs = quantum_state.multi_measure(qubits)
                    
                    for i in range(len(meas_ints)):
                        meas_res_dic[meas_ints[i]] = int(shots*probs[i])
                
                if eqn.params["name"] == "terminal_sampling_helper_2":
                    
                    for k, v in meas_res_dic.items():
                        new_invalues = list(invalues)
                        new_invalues[1] = k
                        outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*new_invalues)
                        
                        key = outvalues[1]
                        
                        if key.dtype == jnp.int32:
                            key = int(key)
                        elif key.dtype == jnp.float32:
                            key = float(key)
                        elif key.dtype == jnp.bool:
                            key = bool(key)
                        
                        decoded_meas_res_dic[key] = v
                    
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
