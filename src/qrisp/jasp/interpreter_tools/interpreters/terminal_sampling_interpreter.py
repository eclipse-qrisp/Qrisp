"""
\********************************************************************************
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
********************************************************************************/
"""

from functools import lru_cache
from random import shuffle

import numpy as np

import jax.numpy as jnp
import jax

from qrisp.jasp.interpreter_tools.abstract_interpreter import eval_jaxpr, extract_invalues, insert_outvalues, exec_eqn
from qrisp.jasp.interpreter_tools.interpreters.control_flow_interpretation import evaluate_while_loop

# The following function implements the behavior of the jaspify simulator for terminal sampling
# To understand the function consider the result of tracing a simple sampling task

# from qrisp.jasp import *
# from qrisp import *

# def inner_f():
#     qf = QuantumFloat(4)
#     h(qf[0])
#     return qf

# @make_jaxpr
# def main(i):
    
#     return expectation_value(inner_f, i)()

# print(main(100))

# While the resulting Jaxpr accurately describes a sampling task, the sampling
# is not as scalable as it could be, given the powerfull sampling features of the
# Qrisp simulator.

# When called via @jaspify and terminal_sampling = True, we need some custom logic
# to implement the terminal_sampling behavior.

# The traced Jaxpr contains the "sampling_body_func", which
# is (visually) partioned into three sections:
    
# 1. The evaluation of the user function, which executes the code of "inner_func"
# 2. The sampling helper 1 function. This function contains the measurement primitives.
# 3. The sampling helper 2 function. This function contains the decoding of the measurement results.

# These functions are evaluated with custom code. Of particular importance is step 2,
# which leverages the fast sampling features of the Qrisp simulator (instead of looping)
# over measurements.

# { lambda a:f64[1]; b:i64[]. let
#     c:f64[] = pjit[
#       name=expectation_value_eval_function
#       jaxpr={ lambda ; d:f64[1] e:i64[]. let
#           _:i64[] _:i64[] f:f64[1] = while[
#             body_jaxpr={ lambda ; g:i64[] h:i64[] i:f64[1]. let
#                 j:i64[] = add g 1
#                 k:QuantumCircuit = jasp.quantum_kernel 
#                 _:QuantumCircuit l:f64[1] = pjit[
#                   name=sampling_body_func
#                   jaxpr={ lambda ; m:QuantumCircuit n:i64[] o:f64[1]. let

# -------------------------------------------------------------------------------------

#                       p:QuantumCircuit q:QubitArray r:i64[] _:bool[] = pjit[
#                         name=user_func
#                         jaxpr={ lambda ; s:QuantumCircuit. let
#                             t:QuantumCircuit u:QubitArray = jasp.create_qubits s
#                               4
#                             v:Qubit = jasp.get_qubit u 0
#                             w:QuantumCircuit = jasp.h t v
#                           in (w, u, 0, False) }
#                       ] m

# -------------------------------------------------------------------------------------

#                       x:QuantumCircuit y:i64[] = pjit[
#                         name=sampling_helper_1
#                         jaxpr={ lambda ; z:QuantumCircuit ba:QubitArray. let
#                             bb:QuantumCircuit bc:i64[] = jasp.measure z ba
#                           in (bb, bc) }
#                       ] p q

# -------------------------------------------------------------------------------------

#                       bd:f64[] = pjit[
#                         name=sampling_helper_2
#                         jaxpr={ lambda ; be:i64[] bf:i64[]. let
#                             _:f64[] = convert_element_type[
#                               new_dtype=float64
#                               weak_type=False
#                             ] be
#                             bg:f64[] = pow 2.0 be
#                             bh:f64[] = convert_element_type[
#                               new_dtype=float64
#                               weak_type=False
#                             ] bf
#                             bi:f64[] = mul bh bg
#                           in (bi,) }
#                       ] r y

# -------------------------------------------------------------------------------------

#                       bj:f64[1] = broadcast_in_dim[
#                         broadcast_dimensions=()
#                         shape=(1,)
#                       ] bd
#                       bk:f64[1] = add o bj
#                       bl:QuantumCircuit = jasp.reset x q
#                       _:QuantumCircuit = jasp.delete_qubits bl q
#                     in (x, bk) }
#                 ] k g i
#               in (j, h, l) }
#             body_nconsts=0
#             cond_jaxpr={ lambda ; bm:i64[] bn:i64[] bo:f64[1]. let
#                 bp:bool[] = lt bm bn
#               in (bp,) }
#             cond_nconsts=0
#           ] 0 e d
#           bq:f64[1] = slice[limit_indices=(1,) start_indices=(0,) strides=None] f
#           br:f64[] = squeeze[dimensions=(0,)] bq
#           bs:f64[] = convert_element_type[new_dtype=float64 weak_type=False] e
#           bt:f64[] = div br bs
#         in (bt,) }
#     ] a b
#   in (c,) }

def terminal_sampling_evaluator(sampling_res_type):
    
    def sampling_eqn_evaluator(eqn, context_dic, eqn_evaluator = exec_eqn):
        
        invalues = extract_invalues(eqn, context_dic)
        
        
        # Extract the shot number
        if sampling_res_type == "ev":
            shots = invalues[0]
        elif sampling_res_type == "array":
            shots = invalues[0]
        elif sampling_res_type == "dict":
            shots = invalues[0]
            if shots == 0:
                invalues[0] = 1
            
        
        # We will use this dictionary to store the sampling results
        meas_res_dic = {}
        
        # This list will contain information about how many qubits are contained
        # in each QuantumVariable of the returned function
        return_signature = []
        
        decoded_meas_res = []
        
        def sampling_body_eqn_evaluator(eqn, context_dic):
            
            # We only do a single iteration (the actual iteration is performed within)
            # the simulator.
            if eqn.primitive.name == "while":
                return evaluate_while_loop(eqn, 
                                           context_dic, 
                                           eqn_evaluator = sampling_body_eqn_evaluator, 
                                           break_after_first_iter = True)
            
            
            if eqn.primitive.name == "pjit":
                invalues = extract_invalues(eqn, context_dic)
                
                function_name = eqn.params["name"]
                
                # sampling_body_func is called with the ev_eqn_evaluator
                if function_name == "sampling_body_func":
                    outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator=sampling_body_eqn_evaluator)(*invalues)
                    insert_outvalues(eqn, context_dic, outvalues)
                    return
                
                if function_name == "user_func":
                    pass
            
                # This case describes the logic to use the simulator sampling features
                if function_name == "sampling_helper_1":
                
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
                            meas_res_dic[k] = np.round(v, decimals = 7)
                            norm += meas_res_dic[k]
                        
                        for k, v in meas_res_dic.items():
                            meas_res_dic[k] = v/norm
                            
                    outvalues = [0]*len(eqn.outvars)
                    insert_outvalues(eqn, context_dic, outvalues)
    
                # Each int in the values of meas_res_dic represents the values
                # of all QuantumVariables. We therefore need to "split" the ints
                # into the appropriate parts and decode them.
                # Splitting means turning the int "1001001" into "100" and "1001".
                if function_name == "sampling_helper_2":
                    
                    if sampling_res_type == "ev":
                        sampling_res = jnp.zeros(len(eqn.outvars))
                    elif sampling_res_type == "array":
                        sampling_res = []
                        sampling_res_dict = {}
                    elif sampling_res_type == "dict":
                        sampling_res = {}
                        
                    # Compile the decoder
                    decoder = decoder_compiler(eqn.params["jaxpr"], eqn_evaluator)
                    
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
                            new_invalues[len(invalues)-len(return_signature)+i] = (k & ((2**(return_signature[i])-1) << j))>>j
                            j += return_signature[i]
                        
                        # Evaluate the decoder
                        #outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator = eqn_evaluator)(*new_invalues)
                        
                        outvalues = decoder(*new_invalues)
                        
                        
                        # We now build the key for the result dic
                        # For that we turn the jax types into the corresponding
                        # Python types.
                        if not isinstance(outvalues, tuple):
                            if sampling_res_type == "ev":
                                sampling_res += outvalues*v
                            elif sampling_res_type == "array":
                                #sampling_res.extend(v*[outvalues[0]])
                                key = outvalues[0]
                                if key.size == 1:
                                    sampling_res_dict[key.item()] = v
                                else:
                                    sampling_res_dict[tuple(np.array(key))] = v
                                    #sampling_res.extend(v*[key])
                            elif sampling_res_type == "dict":
                                key = outvalues
                                if not type(v) in [int, float]:
                                    if v.dtype in [np.float64, np.float32]:
                                        v = float(v.item())
                                    elif v.dtype in [np.int32, np.int64]:
                                        v = int(v.item())
                                    else:
                                        raise
                                sampling_res[key.item()] = v
                            
                        # If the user given function returned more than one
                        # value, the key is a tuple to be build up
                        else:
                            if sampling_res_type == "ev":
                                sampling_res += jnp.array(outvalues)*v
                            elif sampling_res_type == "array":
                                sampling_res.extend(v*[outvalues])
                            elif sampling_res_type == "dict":
                                if not type(v) in [int, float]:
                                    if v.dtype in [np.float64, np.float32]:
                                        v = float(v.item())
                                    elif v.dtype in [np.int32, np.int64]:
                                        v = int(v.item())
                                    else:
                                        raise
                                sampling_res[tuple(x.item() for x in outvalues)] = v
                                
                    if sampling_res_type == "array":
                        keys = np.array(list(sampling_res_dict.keys()))
                        counts = np.array(list(sampling_res_dict.values()))
                        sampling_res = np.repeat(keys, counts, axis=0)
                        np.random.shuffle(sampling_res)
                        
                    elif sampling_res_type == "ev":
                        sampling_res = sampling_res/shots
                        if sampling_res.shape[0] == 1:
                            sampling_res = sampling_res[0]
                    elif sampling_res_type == "dict":
                        # Sort the counts such the most probable values come first
                        sampling_res = dict(sorted(sampling_res.items(), key=lambda item: item[0]))
                        sampling_res = dict(sorted(sampling_res.items(), key=lambda item: -item[1]))
                        
                    decoded_meas_res.append(sampling_res)
            
            return eqn_evaluator(eqn, context_dic)
    
        # Execute the above defined interpreter
        sampling_body_jaxpr = eqn.params["jaxpr"].jaxpr

        outvalues = eval_jaxpr(sampling_body_jaxpr, eqn_evaluator = sampling_body_eqn_evaluator)(*invalues)
        
        
        
        if not isinstance(outvalues, (list, tuple)):
            outvalues = [outvalues]
        
        insert_outvalues(eqn, context_dic, decoded_meas_res)
            
    return sampling_eqn_evaluator


@lru_cache(maxsize = int(1E5))
def decoder_compiler(jaxpr, eqn_evaluator):
    return jax.jit(eval_jaxpr(jaxpr, eqn_evaluator = eqn_evaluator))