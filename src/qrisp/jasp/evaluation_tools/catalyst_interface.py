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

from functools import lru_cache

from jax import make_jaxpr
from jax.extend.core import Literal
import jax.numpy as jnp

import pennylane as qml
import catalyst
from catalyst.jax_primitives import qalloc_p, device_init_p, AbstractQreg
from catalyst.jax_extras.patches import patched_make_eqn
from catalyst.utils.patching import Patcher
from jax.interpreters.partial_eval import DynamicJaxprTrace


from qrisp.jasp import (
    AbstractQubitArray,
    AbstractQubit,
    AbstractQuantumCircuit,
    eval_jaxpr,
    Jlist,
)
from qrisp.jasp.interpreter_tools.interpreters.catalyst_interpreter import (
    catalyst_eqn_evaluator,
)


def jaspr_to_catalyst_jaxpr(jaspr):
    """
    Converts a jaspr into a Catalyst Jaxpr.

    Since the jasp modelling aproach of quantum computation differs a bit from
    the Catalyst Jaxpr model, we have to translate between the models.

    The AbstractQreg datastructure in Catalyst is treated as a stack of qubits,
    that can each be adressed by an integer. We therefore make the following
    conceptual replacements.


    AbstractQubit -> A simple integer denoting the position of the Qubit in the stack.

    AbstractQubitArray -> A tuple of two integers. The first integer indicates
                        the "starting position" of the Qubits of the QubitArray
                        in the stack, and the second integer denotes the length
                        of the QubitArray.

    AbstractQuantumCircuit -> A tuple of a AbstractQreg and an integer i. The integer
                            denotes the current "stack size", ie. if a new
                            QubitArray of size l is allocated it will be an
                            interval of qubits starting at position i and the
                            new tuple representing the new AbstractQuantumCircuit
                            will have i_new = i + l

    Parameters
    ----------
    jaspr : qrisp.jasp.jaspr
        The input jaspr.

    Returns
    -------
    jax.core.Jaxpr
        The output Jaxpr using catalyst primitives.

    """

    # Translate the input args according to the above rules.
    args = []
    for invar in jaspr.jaxpr.invars:
        if isinstance(invar.aval, AbstractQuantumCircuit):
            # We initialize with the inverted list [... 3, 2, 1, 0] since the
            # pop method of the dynamic list always removes the last element
            args.append((AbstractQreg(), Jlist(jnp.arange(30, 0, -1), max_size=30)))
        elif isinstance(invar.aval, AbstractQubitArray):
            args.append(Jlist())
        elif isinstance(invar.aval, AbstractQubit):
            args.append(jnp.asarray(0, dtype="int64"))
        elif isinstance(invar, Literal):
            if isinstance(invar.val, int):
                args.append(jnp.asarray(invar.val, dtype="int64"))
            if isinstance(invar.val, float):
                args.append(jnp.asarray(invar.val, dtype="f32"))
        else:
            args.append(invar.aval)

    # Call the Catalyst interpreter
    
    # Hotfix according to: https://github.com/PennyLaneAI/catalyst/issues/2394#issuecomment-3752134787
    with Patcher((DynamicJaxprTrace, "make_eqn", patched_make_eqn)):
        return make_jaxpr(eval_jaxpr(jaspr, eqn_evaluator=catalyst_eqn_evaluator))(*args)

def jaspr_to_catalyst_function(jaspr, device=None):

    # This function takes a jaspr and returns a function that performs a sequence
    # of .bind calls of Catalyst primitives, such that the function (when compiled)
    # by Catalyst reproduces the semantics of jaspr

    # Initiate Catalyst backend info
    if device==None:
        device = qml.device("lightning.qubit", wires=0)

    backend_info = catalyst.device.extract_backend_info(device)

    def catalyst_function(*args):
        # Initiate the backend
        device_init_p.bind(
            0,
            rtd_lib=backend_info.lpath,
            rtd_name=backend_info.c_interface_name,
            rtd_kwargs=str(backend_info.kwargs),
            auto_qubit_management = True
        )

        # Create the AbstractQreg
        qreg = qalloc_p.bind(20)

        # Insert the Qreg into the list of arguments (such that it is used by the
        # Catalyst interpreter.
        args = list(args)

        # We initialize with the inverted list [... 3, 2, 1, 0] since the
        # pop method of the dynamic list always removes the last element
        args.append((qreg, Jlist(jnp.arange(30, 0, -1), max_size=30)))

        # Call the catalyst interpreter. The first return value will be the AbstractQreg
        # tuple, which is why we exclude it from the return values
        
        # Hotfix according to: https://github.com/PennyLaneAI/catalyst/issues/2394#issuecomment-3752134787
        with Patcher((DynamicJaxprTrace, "make_eqn", patched_make_eqn)):
            return eval_jaxpr(jaspr, eqn_evaluator=catalyst_eqn_evaluator)(*args)[:-1]

    return catalyst_function


@lru_cache(int(1e5))
def jaspr_to_catalyst_qjit(jaspr, function_name="jaspr_function", device=None):
    # This function takes a jaspr and turns it into a Catalyst QJIT object.
    # Perform the code specified by the Catalyst developers
    catalyst_function = jaspr_to_catalyst_function(jaspr, device=device)
    catalyst_function.__name__ = function_name
    jit_object = catalyst.QJIT(catalyst_function, catalyst.CompileOptions())
    jit_object.jaxpr = make_jaxpr(catalyst_function)(
        *[invar.aval for invar in jaspr.invars[:-1]]
    )
    jit_object.workspace = jit_object._get_workspace()
    temp = jit_object.generate_ir()
    if isinstance(temp, tuple):
        raise Exception("Please upgrade to pennylane-catalyst>=0.11.0")
    jit_object.mlir_module = temp
    jit_object.compiled_function, _ = jit_object.compile()
    return jit_object


def jaspr_to_qir(jaspr):
    # This function returns the QIR code for a given jaspr
    qjit_obj = jaspr_to_catalyst_qjit(jaspr)
    return qjit_obj.qir


def jaspr_to_mlir(jaspr):
    # This function returns the MLIR code for a given jaspr
    qjit_obj = jaspr_to_catalyst_qjit(jaspr)
    return qjit_obj.mlir
