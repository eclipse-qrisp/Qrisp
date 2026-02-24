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
    The ``custom_inversion`` decorator allows a user to specify the custom inverted version of
    the decorated function. If this function is called with the ``inv`` keyword set to
    ``True``, the custom inverted version is executed instead. There is one version of
    the function when ``inv`` is ``False`` (forward version) and another version when ``inv``
    is ``True`` (backward version).

    This decorator is crucial for functions where the inversion logic cannot be 
    derived simply by reversing the gate order (e.g., operations involving 
    measurements or dynamic classical control). In such scenarios, the user can explicitly
    define how the function should behave when inverted, ensuring that the correct logic is applied in
    both forward and backward contexts. Here, it is not as straightforward as using :ref:`InversionEnvironment`,
    the general inversion environment. 

    In order to use the ``custom_inversion`` decorator, you need to add the ``inv``
    keyword to your function signature. If called within an inversion context,
    this keyword will receive the corresponding boolean value indicating whether the function
    is being inverted.

    For more details consult the examples section.


    Parameters
    ----------
    func : function
        A function of QuantumVariables, which has the ``inv`` keyword.

    Returns
    -------
    adaptive_inversion_function : function
        A function which will execute it's inverted version, if called
        within the custom inversion context.

    Examples
    ----------

    We create an in-place addition function, which adds 10 to the input when called in the forward direction
    and subtracts 10 from the input when called in the backward direction. 

    ::

        from qrisp import QuantumFloat, custom_inversion, reset

        @custom_inversion
        def load_constant(qf, inv=False):

            if not inv:
                # Forward: In-place addition
                qf += 10
                print("Forward: In-place addition of 10")
            else:
                # Inverse: Go back to previous value
                # reverse in-place addition with -10
                qf += -10
                print("Backward: Go back to 0 by in-place addition of -10")

    To test the function, we can call it in both forward and backward contexts on a `QuantumFloat`.

    ::

        qf = QuantumFloat(0.0)

        # Forward call
        load_constant(qf)
        print("Value:", qf)

        # Backward call
        load_constant(qf, inv=True)
        print("Value:", qf)



    .. code-block:: none

        Forward: In-place addition of 10
        Value: {10: 1.0}  

        Backward: Go back to 0 by in-place addition of -10
        Value: {0: 1.0}  

        

 


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
