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
    The ``custom_control`` decorator allows to specify the controlled version of
    the decorated function. If this function is called within a :ref:`ControlEnvironment`
    or a :ref:`ConditionEnvironment` the controlled version is executed instead.

    Specific controlled versions of quantum functions are very common in many
    scientific publications. This is because the general control procedure can
    signifcantly increase resource demands.

    In order to use the ``custom_control`` decorator, you need to add the ``ctrl``
    keyword to your function signature. If called within a controlled context,
    this keyword will receive the corresponding control qubit.

    For more details consult the examples section.


    Parameters
    ----------
    func : function
        A function of QuantumVariables, which has the ``ctrl`` keyword.

    Returns
    -------
    adaptive_control_function : function
        A function which will execute it's controlled version, if called
        within a :ref:`ControlEnvironment` or a :ref:`ConditionEnvironment`.

    Examples
    --------

    We create a swap function with custom control.

    ::

        from qrisp import mcx, cx, custom_control

        @custom_control
        def swap(a, b, ctrl = None):

            if ctrl is None:

                cx(a, b)
                cx(b, a)
                cx(a, b)

            else:

                cx(a, b)
                mcx([ctrl, b], a)
                cx(a, b)


    Test the non-controlled version:

    ::

        from qrisp import QuantumBool

        a = QuantumBool()
        b = QuantumBool()

        swap(a, b)

        print(a.qs)


    .. code-block:: none

        QuantumCircuit:
        --------------
                  ┌───┐
        a.0: ──■──┤ X ├──■──
             ┌─┴─┐└─┬─┘┌─┴─┐
        b.0: ┤ X ├──■──┤ X ├
             └───┘     └───┘
        Live QuantumVariables:
        ---------------------
        QuantumBool a
        QuantumBool b


    Test the controlled version:

    ::

        from qrisp import control

        a = QuantumBool()
        b = QuantumBool()
        ctrl_qbl = QuantumBool()

        with control(ctrl_qbl):

            swap(a,b)

        print(a.qs.transpile(1))

    .. code-block:: none

                         ┌───┐
               a.0: ──■──┤ X ├──■──
                    ┌─┴─┐└─┬─┘┌─┴─┐
               b.0: ┤ X ├──■──┤ X ├
                    └───┘  │  └───┘
        ctrl_qbl.0: ───────■───────


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
