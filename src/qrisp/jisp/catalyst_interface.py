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

import pennylane as qml
import catalyst
from catalyst.jax_primitives import qalloc_p, qdevice_p, AbstractQreg


from qrisp.jisp import AbstractQubitArray, AbstractQubit, AbstractQuantumCircuit
from qrisp.jisp.interpreter_tools.interpreters.catalyst_interpreter import catalyst_eqn_evaluator
from qrisp.jisp import eval_jaxpr

def jispr_to_catalyst_jaxpr(jispr):
    
    args = []
    for invar in jispr.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            args.append((AbstractQreg(), 0))
        elif isinstance(invar.aval, AbstractQubitArray):
            args.append((0,0))
        elif isinstance(invar.aval, AbstractQubit):
            args.append(0)
        else:
            args.append(invar.aval)
        
    return make_jaxpr(eval_jaxpr(jispr, eqn_evaluator = catalyst_eqn_evaluator))(*args)


def jispr_to_catalyst_function(jispr):
    
    # Initiate Catalyst backend
    device = qml.device("lightning.qubit", wires=0)        
    program_features = catalyst.utils.toml.ProgramFeatures(shots_present=False)
    device_capabilities = catalyst.device.get_device_capabilities(device, program_features)
    backend_info = catalyst.device.extract_backend_info(device, device_capabilities)
    
    def catalyst_function(*args):
        #Initiate the backend
        qdevice_p.bind(
        rtd_lib=backend_info.lpath,
        rtd_name=backend_info.device_name,
        rtd_kwargs=str(backend_info.kwargs),
        )
        
        qreg = qalloc_p.bind(20)
        
        args = list(args)
        args.insert(0, (qreg, 0))
        
        return eval_jaxpr(jispr, eqn_evaluator = catalyst_eqn_evaluator)(*args)[1:]
    
    return catalyst_function


def jispr_to_catalyst_qjit(jispr, function_name = "jispr_function"):
    
    def inner_function(*args):
        
        catalyst_function = jispr_to_catalyst_function(jispr)
        catalyst_function.__name__ = function_name
        jit_object = catalyst.QJIT(catalyst_function, catalyst.CompileOptions())
        jit_object.jaxpr = make_jaxpr(catalyst_function)(*args)
        jit_object.workspace = jit_object._get_workspace()
        jit_object.mlir_module, jit_object.mlir = jit_object.generate_ir()
        jit_object.compiled_function, jit_object.qir = jit_object.compile()
        return jit_object
    
    return inner_function
    

def qjit(function):
    
    from qrisp.jisp import make_jispr
    
    def jitted_function(*args):
        jispr = make_jispr(function)(*args)
        qjit_obj = jispr_to_catalyst_qjit(jispr, function_name = function.__name__)(*args)
        return qjit_obj.compiled_function(*args)
    
    return jitted_function


def jispr_to_qir(jispr, args):
    qjit_obj = jispr_to_catalyst_qjit(jispr)(*args)
    return qjit_obj.qir
    
def jispr_to_mlir(jispr, args):
    qjit_obj = jispr_to_catalyst_qjit(jispr)(*args)
    return qjit_obj.mlir