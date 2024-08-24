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

from jax import make_jaxpr
from jax.core import Literal


from qrisp.jisp import QuantumPrimitive, AbstractQubitArray

# Name translator from Qrisp gate naming to Catalyst gate naming
op_name_translation_dic = {"cx" : "CNOT",
                       "cy" : "CY", 
                       "cz" : "CZ", 
                       "crx" : "CRX",
                       "crz" : "CRZ",
                       "swap" : "SWAP",
                       "x" : "PauliX",
                       "y" : "PauliY",
                       "z" : "PauliZ",
                       "h" : "Hadamard",
                       "rx" : "RX",
                       "ry" : "RY",
                       "rz" : "RZ",
                       "s" : "S",
                       "t" : "T",
                       "p" : "Phasegate"}


def convert_to_catalyst_function(closed_jaxpr, args):
    """
    This function converts a Qrisp Jaxpr to the equivalent in Catalyst.
    
    The Qrisp and the Catalyst Jaxpr are differing in the way how they treat 
    quantum gates (that are fundamentally in-place) in the functional programming
    setting, where in-place operations are forbidden.
    
    The Catalyst Jaxpr creates new qubit objects for every quantum gate and returns these.
    Furthermore the qinsert primitive creates a new register object, representing the new
    register with the inserted qubit.
    
    The Qrisp Jaxpr treats qubit and registers as opaque objects that could however be
    represented by simple integers indexing qubits in a quantum circuit.
    Quantum gates are modelled as functions that take a quantum circuit and qubit object 
    (aka. tensor indices) and return a new quantum circuit object.
    
    This has the nice property that measurements are side effect free operations,
    which is not the case for the Catalyst model, since the induced collaps can 
    influence non-participating qubit objects.

    Parameters
    ----------
    closed_jaxpr : Jaxpr
        A Jaxpr coming from a traced Qrisp function.
    args : iterable
        The arguments for the Qrisp function.

    Returns
    -------
    Jaxpr
        A Jaxpr using Catalyst primitives.

    """
    from qrisp.circuit import Operation    
    from catalyst.jax_primitives import qalloc_p, qinst_p, qmeasure_p, qdevice_p, qextract_p, qinsert_p
    import catalyst
    import pennylane as qml
    # Extract the Jaxpr from the ClosedJaxpr
    jaxpr = closed_jaxpr

    # Initiate Catalyst backend
    # _, device_name, device_libpath, device_kwargs = catalyst.utils.runtime.extract_backend_info(
    # qml.device("lightning.qubit", wires=0))
    
    device = qml.device("lightning.qubit", wires=0)        
    program_features = catalyst.utils.toml.ProgramFeatures(shots_present=False)
    device_capabilities = catalyst.device.get_device_capabilities(device, program_features)
    backend_info = catalyst.device.extract_backend_info(device, device_capabilities)
    

    # In the following there are two different types of objects involved:
        
    # 1. The variable objects from the equations of the Jaxpr to be processed
    # 2. The tracer objects from the Jaxpr that is built up

    # This dictionary translates the variables (as found in the Qrisp Jaxpr) to the 
    # corresponding tracer objects for tracing the Catalyst Jaxpr
    context_dic = {}
    
    # Wrapper around the dictionary to also treat literals
    def var_to_tr(var):
        if isinstance(var, Literal):
            return var.val
        else:
            return context_dic[var]

    # Since the compiling model of both Jaxpr differ a bit, we have to perform
    # some non-trivial translation.
    
    # This dictionary contains qubit variable objects as keys and
    # register/integer pairs as values.
    
    # This is required to load the correct qubit from the Catalyst registers
    qubit_dictionary = {}
    
    
    # This function will be traced
    def tracing_function():
        
        # Initiate the backend
        qdevice_p.bind(
        rtd_lib=backend_info.lpath,
        rtd_name=backend_info.device_name,
        rtd_kwargs=str(backend_info.kwargs),
        )
        
        
        # Insert the appropriate variable/tracer relation of the arguments into the dictionary
        for i in range(len(args)):
            context_dic[jaxpr.invars[i+1]] = args[i]
            
        context_dic[jaxpr.invars[0]] = (qalloc_p.bind(20), 0)
            
        # Loop through equations and process Qrisp primitive accordingly
        for eqn in jaxpr.eqns:
            
            invars = eqn.invars
            outvars = eqn.outvars
            
            # This is the case that the primitive is not given by Qrisp
            if not isinstance(eqn.primitive, QuantumPrimitive):
                
                # Basically the only thing that needs to be done is to translate the
                # variables to tracers and bind the primitive to the tracer argument
                if eqn.primitive.multiple_results:
                    tracers = eqn.primitive.bind(*[var_to_tr(var) for var in eqn.invars], **eqn.params)
                else:
                    tracer = eqn.primitive.bind(*[var_to_tr(var) for var in eqn.invars], **eqn.params)
                    tracers = [tracer]
                
                # Subsequently the resulting tracers need to be inserted into the 
                # dictionary
                for var in eqn.outvars:
                    context_dic[var] = tracer
            
            else:
                # This is the Qrisp primitive case
                
                if eqn.primitive.name == "create_qubits":
                    
                    qreg, stack_size = var_to_tr(invars[0])
                    context_dic[outvars[1]] = (stack_size, var_to_tr(invars[1]))
                    context_dic[outvars[0]] = (qreg, stack_size + var_to_tr(invars[1]))
                    
                    
                elif eqn.primitive.name == "get_qubit":
                    context_dic[outvars[0]] = context_dic[invars[0]][0] + var_to_tr(invars[1])
                    
                elif isinstance(eqn.primitive, Operation) or eqn.primitive.name == "measure":
                    # This case is applies a quantum operation
                    op = eqn.primitive
                    
                    # For this the first step is to collect all the Catalyst qubit tracers
                    # that are required for the Operation
                    qb_vars = []
                    
                    qb_pos = []
                    if op.name == "measure":
                        
                        if isinstance(invars[1].aval, AbstractQubitArray):
                            
                            qubit_array_data = var_to_tr(invars[1])
                            pos = qubit_array_data[0]
                            length = qubit_array_data[1]
                            
                            for i in range(length):
                                qb_pos.append(pos + i)
                        else:
                            qb_pos.append(var_to_tr(invars[1]))
                    else:
                        
                        for i in range(op.num_qubits):
                            qb_vars.append(invars[i+1+len(op.params)])
                            qb_pos.append(var_to_tr(invars[i+1+len(op.params)]))
                    
                    num_qubits = len(qb_pos)
                    catalyst_register_tracer = var_to_tr(invars[0])[0]
                    catalyst_qb_tracers = []
                    for i in range(num_qubits):
                        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                                             qb_pos[i])
                        catalyst_qb_tracers.append(catalyst_qb_tracer)
                    
                    # We can now apply the gate primitive
                    
                    if op.name == "measure":
                        if len(catalyst_qb_tracers) == 1:
                            res_bl, res_qb = qmeasure_p.bind(*catalyst_qb_tracers)
                            res_qbs = [res_qb]
                            context_dic[outvars[1]] = res_bl
                        else:
                            res_qbs = []
                            meas_res = 0
                            for i in range(len(catalyst_qb_tracers)):
                                res_bl, res_qb = qmeasure_p.bind(catalyst_qb_tracers[i])
                                meas_res += 2**i*res_bl
                                res_qbs.append(res_qb)
                            
                            context_dic[outvars[1]] = meas_res
                                
                        
                    else:
                        res_qbs = exec_qrisp_op(op, catalyst_qb_tracers)
                        
                    
                    # Finally, we reinsert the qubits and update the register tracer
                    for i in range(num_qubits):

                        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                                             qb_pos[i],
                                                             res_qbs[i])
                        
                        
                    context_dic[outvars[0]] = (catalyst_register_tracer, var_to_tr(invars[0])[1])

        # Return the appropriate tracers
        return tuple(var_to_tr(var) for var in jaxpr.outvars[1:])
                    
    return tracing_function


def jispr_to_qir(jispr, args):
    import catalyst
    catalyst_function = convert_to_catalyst_function(jispr, args)
    catalyst_jaxpr = make_jaxpr(catalyst_function)()
    
    mlir_module, mlir_ctx = catalyst.jax_extras.jaxpr_to_mlir("jisp_function", catalyst_jaxpr)

    catalyst.utils.gen_mlir.inject_functions(mlir_module, mlir_ctx)

    jit_object = catalyst.QJIT(catalyst_function, catalyst.CompileOptions())
    jit_object.compiling_from_textual_ir = False
    jit_object.mlir_module = mlir_module

    compiled_fn = jit_object.compile()[0]
    return jit_object.qir
    
def jispr_to_mlir(jispr, args):
    import catalyst
    catalyst_function = convert_to_catalyst_function(jispr, args)
    catalyst_jaxpr = make_jaxpr(catalyst_function)()
    
    mlir_module, mlir_ctx = catalyst.jax_extras.jaxpr_to_mlir("jisp_function", catalyst_jaxpr)

    catalyst.utils.gen_mlir.inject_functions(mlir_module, mlir_ctx)

    jit_object = catalyst.QJIT(catalyst_function, catalyst.CompileOptions())
    jit_object.compiling_from_textual_ir = False
    jit_object.mlir_module = mlir_module

    compiled_fn = jit_object.compile()[0]
    return jit_object.mlir

def jispr_to_catalyst_jaxpr(jispr, args):
    import catalyst
    catalyst_function = convert_to_catalyst_function(jispr, args)
    return make_jaxpr(catalyst_function)().jaxpr
    

def qjit(function):
    import catalyst
    from qrisp.jisp import make_jispr
    
    def jitted_function(*args):
        
        qrisp_jaxpr = make_jispr(function)(*args)
        
        catalyst_function = convert_to_catalyst_function(qrisp_jaxpr, args)
        catalyst_jaxpr = make_jaxpr(catalyst_function)()
        
        mlir_module, mlir_ctx = catalyst.jax_extras.jaxpr_to_mlir(function.__name__, catalyst_jaxpr)

        catalyst.utils.gen_mlir.inject_functions(mlir_module, mlir_ctx)

        jit_object = catalyst.QJIT(catalyst_function, catalyst.CompileOptions())
        jit_object.compiling_from_textual_ir = False
        jit_object.mlir_module = mlir_module

        compiled_fn = jit_object.compile()[0]
        print(jit_object.mlir)

        return compiled_fn(*args)
    
    return jitted_function

def exec_qrisp_op(op, catalyst_qbs):
    
    if op.definition:
        defn = op.definition
        for instr in defn.data:
            qubits = instr.qubits
            qubit_indices = [defn.qubits.index(qb) for qb in qubits]
            temp_catalyst_qbs = [catalyst_qbs[i] for i in qubit_indices]
            res_qbs = exec_qrisp_op(instr.op, temp_catalyst_qbs)
            
            for i in range(len(qubit_indices)):
                catalyst_qbs[qubit_indices[i]] = res_qbs[i]
                
        return catalyst_qbs
    
    else:
            
        from catalyst.jax_primitives import qinst_p
        res_qbs = qinst_p.bind(*catalyst_qbs, 
                               op = op_name_translation_dic[op.name], 
                               qubits_len = op.num_qubits)
        
        return res_qbs
    
def exec_multi_measurement(catalyst_register, start, stop):
    
    from catalyst.jax_primitives import qmeasure_p
    from jax.lax import fori_loop
    
    def loop_body(i, acc, reg):
        qb = qextract_p.bind(reg, i)
        res_bl, res_qb = qmeasure_p.bind(qb)
        acc = acc + 2**i*res_bl
        reg = qinsert_p.bind(reg, i, res_qb)
        return acc, reg
    
    return fori_loop.bind(start, stop, loop_body, (0, catalyst_register))