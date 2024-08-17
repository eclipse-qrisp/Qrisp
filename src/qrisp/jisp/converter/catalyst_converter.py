# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:51:46 2024

@author: sea
"""
from jax import make_jaxpr
from jax.core import Literal


from qrisp.jisp import QuantumPrimitive, AbstractQubitArray


def convert_to_catalyst_jaxpr(closed_jaxpr, args):
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
    jaxpr = closed_jaxpr.jaxpr

    # Initiate Catalyst backend
    _, device_name, device_libpath, device_kwargs = catalyst.utils.runtime.extract_backend_info(
    qml.device("lightning.qubit", wires=0))
    
    # Name translator from Qrisp gate naming to Catalyst gate naming
    op_name_translation_dic = {"cx" : "CNOT", "x" : "PauliX", "h" : "Hadamard"}

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
    def tracing_function(*args):
        
        # Initiate the backend
        qdevice_p.bind(
        rtd_lib=device_libpath,
        rtd_name=device_name,
        rtd_kwargs=str(device_kwargs),
        )
        
        # Insert the appropriate variable/tracer relation of the arguments into the dictionary
        for i in range(len(args)):
            context_dic[jaxpr.invars[i]] = args[i]
            
        # Loop through equations and process Qrisp primitive accordingly
        for eqn in jaxpr.eqns:
            
            invars = eqn.invars
            outvars = eqn.outvars
            
            # This is the case that the primitive is not given by Qrisp
            if not isinstance(eqn.primitive, QuantumPrimitive):
                
                # Basically the only thing that needs to be done is to translate the
                # variables to tracers and bind the primitive to the tracer argument
                if eqn.primitive.multiple_results:
                    tracers = eqn.primitive.bind(*[var_to_tr(var) for var in eqn.invars])
                else:
                    tracer = eqn.primitive.bind(*[var_to_tr(var) for var in eqn.invars])
                    tracers = [tracer]
                
                # Subsequently the resulting tracers need to be inserted into the 
                # dictionary
                for var in eqn.outvars:
                    context_dic[var] = tracer
            
            else:
                # This is the Qrisp primitive case
                if eqn.primitive.name == "qdef":
                    context_dic[outvars[0]] = (qalloc_p.bind(10), 0)
                
                
                elif eqn.primitive.name == "create_qubits":
                    
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
                            qb_vars.append(invars[i+1])
                            qb_pos.append(var_to_tr(invars[i+1]))
                    
                    num_qubits = len(qb_pos)
                    catalyst_register_tracer = var_to_tr(invars[0])[0]
                    catalyst_qb_tracers = []
                    for i in range(num_qubits):
                        catalyst_qb_tracer = qextract_p.bind(catalyst_register_tracer, 
                                                             qb_pos[i])
                        catalyst_qb_tracers.append(catalyst_qb_tracer)
                    
                    # We can now apply the gate primitive
                    
                    if op.name == "measure":
                        res_bl, res_qb = qmeasure_p.bind(*catalyst_qb_tracers)
                        res_qbs = [res_qb]
                        context_dic[outvars[1]] = res_bl
                        
                    else:
                    
                        res_qbs = qinst_p.bind(*catalyst_qb_tracers, 
                                               op = op_name_translation_dic[op.name], 
                                               qubits_len = op.num_qubits)
                        
                    
                    # Finally, we reinsert the qubits and update the register tracer
                    for i in range(num_qubits):

                        catalyst_register_tracer = qinsert_p.bind(catalyst_register_tracer, 
                                                             qb_pos[i],
                                                             res_qbs[i])
                        
                        
                    context_dic[outvars[0]] = (catalyst_register_tracer, var_to_tr(invars[0])[1])

        # Return the appropriate tracers
        return tuple(var_to_tr(var) for var in jaxpr.outvars)
                    
    return make_jaxpr(tracing_function)(*args)

def qjit(function):
    import catalyst
    
    def jitted_function(*args):
        
        qrisp_jaxpr = make_jaxpr(function)(*args)
        
        catalyst_jaxpr = convert_to_catalyst_jaxpr(qrisp_jaxpr, args)
        
        mlir_module, mlir_ctx = catalyst.jax_extras.jaxpr_to_mlir(function.__name__, catalyst_jaxpr)

        catalyst.utils.gen_mlir.inject_functions(mlir_module, mlir_ctx)

        def dummy():
            return 0

        jit_object = catalyst.QJIT(dummy, catalyst.CompileOptions())
        jit_object.compiling_from_textual_ir = False
        jit_object.mlir_module = mlir_module

        compiled_fn = jit_object.compile()[0]
        print(jit_object.qir)

        return compiled_fn(*args)
    
    return jitted_function