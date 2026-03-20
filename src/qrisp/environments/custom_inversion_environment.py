"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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

import inspect

import jax
import jax.numpy as jnp

from qrisp.environments.quantum_environments import QuantumEnvironment
from qrisp.environments.gate_wrap_environment import GateWrapEnvironment
from qrisp.circuit import Operation, QuantumCircuit, Instruction
from qrisp.environments.iteration_environment import IterationEnvironment
from qrisp.core import merge

from qrisp.jasp import (
    check_for_tracing_mode,
    qache,
    AbstractQubit,
    make_jaspr,
    get_last_equation,
)


def custom_inversion(*func, **cusi_kwargs):
    """

    The ``custom_inversion`` decorator enables registering a specialized subroutine for the inverted version
    of the decorated function. This decorator is crucial for functions where the inversion logic cannot be derived
    simply by reversing the gate order (e.g., subroutines involving measurements or dynamic classical control). 
    In such scenarios, the user can explicitly define how the function should behave when inverted, ensuring that
    the correct logic is applied in both forward and backward contexts.

    To make use of the decorator, the decorated function is required to support a keyword argument ``inv``,
    which receives a static boolean. This boolean indicates whether the forward or the backward version of the function
    should be executed. Once defined, the function with decorator applied **does not** need to be called with the ``inv`` keyword. 
    Instead the backward version will be called automatically, if the function is called within an :ref:`InversionEnvironment`.

    .. warning::
        
        Custom inversion is currently only available in dynamic mode.



    For more details consult the examples section. 

    Parameters
    ----------
    func : function
        A function of QuantumVariables, which has the ``inv`` keyword.

    Returns
    -------
    adaptive_inversion_function : function
        A function which will execute it's custom inverted version, if called
        within the custom inversion context.

    Examples
    ----------
    We demonstrate the use of the ``custom_inversion`` decorator with a simple example.
    
    In this example, we define a function that implements Gidney's logical AND operation in the forward direction and uncomputes the
    logical AND in the backward direction. As `defined by Gidney <https://arxiv.org/abs/1709.06648>`_ , the forward and backward
    implementations of the logical AND are not simply inverses of each other. Thus, one cannot use the general :ref:`InversionEnvironment`
    to automatically apply the inverse of the forward implementation. Instead, we use the ``custom_inversion`` decorator to explicitly
    define both the forward and backward implementations of the logical AND operation.

    The ``gidney_mcx_impl`` and ``gidney_mcx_inv_impl`` functions are the pre-defined implementations of the forward and backward versions
    of Gidney's logical AND operation, respectively. We define ``gidney_mcx`` function along with the ``custom_inversion`` decorator, such
    that we simply apply the logical AND operation and then uncompute it using the custom inverse. The final state of the target qubit is
    returned to its initial state, which is the expected behavior for this example.
     
    ::

        from qrisp import QuantumFloat, custom_inversion, invert, make_jaspr, measure
        from qrisp.core import x
        from qrisp import gidney_mcx_impl, gidney_mcx_inv_impl

        @custom_inversion
        def gidney_mcx(a, b, c, inv=False):
            if not inv:
                # Forward: In-place AND operation
                gidney_mcx_impl(a, b, c)
            else:
                # Inverse: uncomputation of logical AND
                gidney_mcx_inv_impl(a, b, c)

        def main():
            # Initialize QuantumFloats
            a = QuantumFloat(1, name="a")
            b = QuantumFloat(1, name="b")
            c = QuantumFloat(1, name="c")

            # Prepare inputs (apply bit-flip to controlling qubits a and b)
            x(a[0])
            x(b[0])

            # Apply Logical AND
            gidney_mcx(a[0], b[0], c[0])

            # Uncompute using the custom inverse
            with invert():
                gidney_mcx(a[0], b[0], c[0])

            return measure(c)

        # Trace and execute
        jaspr = make_jaspr(main)()
        print("Result:", jaspr())
        # Expected Output: 0
    
    
    .. code-block:: python
        
        Result: 0.0
        

 


    """

    if len(func) == 0:
        return lambda x: custom_inversion(x, **cusi_kwargs)
    else:
        func = func[0]

    # The idea to realize the custom control feature in traced mode is to
    # first trace the non-controlled version into a pjit primitive using
    # the qache feature and the trace the controlled version.
    # The controlled version is then stored in the params attribute

    # Qache the function (in non-traced mode, this has no effect)

    # Make sure the inv keyword argument is treated as a static argument
    new_static_argnames = list(cusi_kwargs.get("static_argnames", []))
    new_static_argnames.append("inv")
    cusi_kwargs["static_argnames"] = new_static_argnames

    qached_func = qache(func, **cusi_kwargs)

    def adaptive_inversion_function(*args, **kwargs):

        if not check_for_tracing_mode():
            return func(*args, **kwargs)

        else:

            args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], bool):
                    args[i] = jnp.array(args[i], dtype=jnp.bool)
                elif isinstance(args[i], int):
                    args[i] = jnp.array(args[i], dtype=jnp.int64)
                elif isinstance(args[i], float):
                    args[i] = jnp.array(args[i], dtype=jnp.float64)
                elif isinstance(args[i], complex):
                    args[i] = jnp.array(args[i], dtype=jnp.complex64)

            # Call the (qached) function
            res = qached_func(*args, inv=False, **kwargs)

            # Retrieve the pjit equation
            jit_eqn = get_last_equation()

            if not jit_eqn.params["jaxpr"].inv_jaspr:
                # Trace the inverted version

                def ammended_func(*args, **kwargs):
                    new_kwargs = dict(kwargs)
                    return func(*args, inv=True, **new_kwargs)

                inverted_jaspr = make_jaspr(ammended_func)(*args, **kwargs)

                # Store controlled version
                jit_eqn.params["jaxpr"].inv_jaspr = inverted_jaspr
                inverted_jaspr.inv_jaspr = jit_eqn.params["jaxpr"]

        return res

    return adaptive_inversion_function
