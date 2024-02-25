# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:29:14 2024

@author: sea
"""

import numpy as np
from catalyst.jax_primitives import *
from jax import jit, make_jaxpr
from catalyst.utils.contexts import EvaluationMode, EvaluationContext
import catalyst
import pennylane as qml

_, device_name, device_libpath, device_kwargs = catalyst.utils.runtime.extract_backend_info(
    qml.device("lightning.qubit", wires=0)
)

def catalyst_pipeline(test_f, intended_args = []):
    
    def wrapper(*args):
        qdevice_p.bind(
        rtd_lib=device_libpath,
        rtd_name=device_name,
        rtd_kwargs=str(device_kwargs),
        )
        return test_f(*args)
    
    jaxpr = make_jaxpr(wrapper)(*intended_args)
    
    # for eq in jaxpr.eqns:
        # print(eq)
    print(jaxpr)
    
    mlir_module, mlir_ctx = catalyst.utils.jax_extras.jaxpr_to_mlir(test_f.__name__, jaxpr)
    
    catalyst.utils.gen_mlir.inject_functions(mlir_module, mlir_ctx)
    #print(mlir_module)
    
    jit_object = catalyst.QJIT("test", catalyst.CompileOptions())
    jit_object.compiling_from_textual_ir = False
    jit_object.mlir_module = mlir_module
    
    compiled_fn = jit_object.compile()
    # print(jit_object.qir)
    
    return compiled_fn



def create_trivial_circuit():
    

    reg = qalloc_p.bind(2)
    qb0_0 = qextract_p.bind(reg, 0)
    qb1_0 = qextract_p.bind(reg, 1)
    # qb0_1, = qinst_p.bind(qb0_0, op="PauliX", qubits_len=1)
    qb0_1, = qinst_p.bind(qb0_0, op="Hadamard", qubits_len=1)
    qb0_2, qb1_1 = qinst_p.bind(qb0_1, qb1_0, op="CNOT", qubits_len=2)
    
    # return qb0_0, qb1_0
    return qb0_2, qb1_1


# How to use this in the following functions without tracing again?
#  jitted_created_trivial_circuit = catalyst.qjit(create_trivial_circuit)

def measurement_test():

    
    qb_0, qb_1 = create_trivial_circuit()

    meas_res, qb0_2 = qmeasure_p.bind(qb_0)
    return meas_res


def statevector_test():
    
    qb_0, qb_1 = create_trivial_circuit()

    state_var = compbasis_p.bind(qb_0, qb_1)
    num_value = state_p.bind(state_var, shape=(4,))
 
    return num_value
    
def counts_test():
    
    qb_0, qb_1 = create_trivial_circuit()
    print(type(qb_0))
    test_obs = compbasis_p.bind(qb_0, qb_1)
    counts_a, counts_b = counts_p.bind(test_obs, shots = 100, shape=(2**2,))

    
    return counts_a, counts_b


# print(catalyst_pipeline(measurement_test)())
# print(catalyst_pipeline(statevector_test)())

# Crashes without error

print(catalyst_pipeline(counts_test)())
print(jax.make_jaxpr(counts_test))


from catalyst.utils.jax_extras import wrap_init

inner_jaxpr = jax.make_jaxpr(create_trivial_circuit)()

def measurement_test():
    # qb_2, qb_3 = func_p.bind(wrap_init(create_trivial_circuit), fn=create_trivial_circuit)
    qb_2, qb_3 = create_trivial_circuit()
    meas_res, qb0_2 = qmeasure_p.bind(qb_0)
    return meas_res

# jax.make_jaxpr(measurement_test)()

#%%


# Attempt to measure a dynamic amount of qubits

def create_less_trivial_circuit(qubit_amount):
    
    reg = qalloc_p.bind(qubit_amount)
    qb0_0 = qextract_p.bind(reg, 0)
    qb1_0 = qextract_p.bind(reg, 1)
    qb0_1, = qinst_p.bind(qb0_0, op="Hadamard", qubits_len=1)
    qb0_2, qb1_1 = qinst_p.bind(qb0_1, qb1_0, op="CNOT", qubits_len=2)
    
    qinsert_p.bind(reg, 0, qb0_1)
    qinsert_p.bind(reg, 1, qb1_1)
    
    return reg

def extraction_helper(reg, amount):
    qb_list = []
    for i in range(amount):
        qb = qextract_p.bind(reg, i)
        qb_list.append(qb)
    return compbasis_p.bind(*qb_list)
        
from jax import lax    

def measurement_test(qubit_amount):
    
    reg = create_less_trivial_circuit(qubit_amount)
    
    obs = lax.switch(qubit_amount, [lambda reg : extraction_helper(reg, i) for i in range(5)], reg)
    counts_a, counts_b = counts_p.bind(obs, shots = 100, shape=(2**qubit_amount,))
    
    return counts_a, counts_b

catalyst_pipeline(measurement_test, intended_args = [4])

#%%


import jax
import importlib
import qrisp
importlib.reload(qrisp.quantum_variable)
importlib.reload(qrisp.quantum_session)
from qrisp import *
from qrisp.core import QuantumSession
QuantumVariable = qrisp.quantum_variable.QuantumVariable
# importlib.reload(qrisp.quantum_session)
# from qrisp import QuantumVariable
from catalyst.jax_primitives import *

i = 2
def test_f():
    
    qv = QuantumVariable(i)
    h(qv[0])
    cx(qv[0], qv[1])
    
    return qv.get_measurement()


print(jax.make_jaxpr(test_f))

catalyst_pipeline(test_f)()


