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
    
    mlir_module, mlir_ctx = catalyst.utils.jax_extras.jaxpr_to_mlir(test_f.__name__, jaxpr)
    
    catalyst.utils.gen_mlir.inject_functions(mlir_module, mlir_ctx)
    #print(mlir_module)
    
    jit_object = catalyst.QJIT("test", catalyst.CompileOptions())
    jit_object.compiling_from_textual_ir = False
    jit_object.mlir_module = mlir_module
    
    compiled_fn = jit_object.compile()
    # print(jit_object.qir)
    
    return compiled_fn


from qrisp import *

i = 2
def test_f():
    
    qv = QuantumVariable(i)
    h(qv[0])
    cx(qv[0], qv[1])
    
    return qv.get_measurement()


print(make_jaxpr(test_f)())
"""
{ lambda ; . let
    a:AbstractQreg() = qalloc 2
    b:AbstractQbit() = qextract a 0
    c:AbstractQbit() = qinst[op=Hadamard qubits_len=1] b
    d:AbstractQreg() = qinsert a 0 c
    e:AbstractQbit() = qextract d 0
    f:AbstractQbit() = qextract d 1
    g:AbstractQbit() h:AbstractQbit() = qinst[op=CNOT qubits_len=2] e f
    i:AbstractQreg() = qinsert d 0 g
    j:AbstractQreg() = qinsert i 1 h
    k:AbstractQbit() = qextract j 0
    l:AbstractQbit() = qextract j 1
    m:AbstractObs(num_qubits=2,primitive=compbasis) = compbasis k l
    n:f64[4] o:i64[4] = counts[shape=(4,) shots=100] m
  in (n, o) }
"""


print(catalyst_pipeline(test_f)())
"""
[array([0., 1., 2., 3.]), array([46,  0,  0, 54])]
Bell-State 🥳
"""
pass


