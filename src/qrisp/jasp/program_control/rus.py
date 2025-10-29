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
from jax.lax import cond, while_loop

from qrisp.circuit import XGate
from qrisp.jasp import (
    AbstractQubitArray,
    DynamicQubitArray,
    TracingQuantumSession,
    qache,
)
from qrisp.jasp.primitives import (
    Measurement_p,
    OperationPrimitive,
    delete_qubits_p,
    get_qubit_p,
    get_size_p,
    reset_p,
)


def RUS(*trial_function, **jit_kwargs):
    r"""
    Decorator to deploy repeat-until-success (RUS) components. At the core,
    RUS repeats a given quantum subroutine followed by a qubit measurement until
    the measurement returns the value ``1``. This step is prevalent
    in many important algorithms, among them the
    `HHL algorithm <https://arxiv.org/abs/0811.3171>`_ or the
    `LCU procedure <https://arxiv.org/abs/1202.5822>`_.

    Within Jasp, RUS steps can be realized by providing the quantum subroutine
    as a "trial function", which returns a boolean value (the repetition condition) and
    possibly other return values.

    It is important to note that the trial function can not receive quantum
    arguments. This is because after each trial, a new copy of these arguments
    would be required to perform the next iteration, which is prohibited by
    the no-clone theorem. It is however legal to provide classical arguments.

    Parameters
    ----------
    trial_function : callable
        A function returning a boolean value as the first return value. More
        return values are possible.
    static_argnums : int or list[int], optional
        A list of integers specifying which arguments are considered static in
        the sense of `jax.jit <https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html>`_.
        The first argument is indicated by 1, the second by 2, etc. The default
        is ``[]``.
    static_argnames : str or list[str], optional
        A list of strings specifying which arguments are considered static in
        the sense of `jax.jit <https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html>`_.
        The default is ``[]``.

    Returns
    -------
    callable
        A function that performs the RUS protocol with the trial function. The
        return values of this function are the return values of the trial function
        WITHOUT the boolean value.

    Examples
    --------

    To demonstrate the RUS behavior, we initialize a GHZ state

    .. math::

        \ket{\psi} = \frac{\ket{00000} + \ket{11111}}{\sqrt{2}}

    and measure the first qubit into a boolean value. This will be the value
    to cancel the repetition. This will collapse the GHZ state into either
    $\ket{00000}$ (which will cause a new repetition) or $\ket{11111}$, which
    cancels the loop. After the repetition is canceled we are therefore
    guaranteed to have the latter state.

    ::

        from qrisp.jasp import RUS, make_jaspr
        from qrisp import QuantumFloat, h, cx, measure

        @RUS
        def rus_trial_function():
            qf = QuantumFloat(5)
            h(qf[0])

            for i in range(1, 5):
                cx(qf[0], qf[i])

            cancelation_bool = measure(qf[0])
            return cancelation_bool, qf

        def call_RUS_example():

            qf = rus_trial_function()

            return measure(qf)

    Create the ``jaspr`` and simulate:

    ::

        jaspr = make_jaspr(call_RUS_example)()
        print(jaspr())
        # Yields, 31 which is the decimal version of 11111

    **Static arguments**

    To demonstrate the specification of static arguments, we will realize implement a
    simple `linear combination of unitaries <https://arxiv.org/abs/1202.5822>`_.

    Our implementation initializes a state of the form

    .. math::

        \left( \sum_{i = 0}^N c_i U_i \right) \ket{0}.

    We achieve this by specifying a set of unitaries $U_i$ in the form of a
    tuple of functions, each processing a :ref:`QuantumFloat`.

    The coefficients $c_i$ are specified through a function preparing the state

    .. math::

        \ket{\psi} = \sum_{i = 0}^N c_i \ket{i}

    For the state preparation function we specify two options to experiment with.
    A two qubit uniform superposition and a function that brings only the first
    qubit into superpostion.

    ::

        def state_prep_full(qv):
            h(qv[0])
            h(qv[1])

        def state_prep_half(qv):
            h(qv[0])

    For the first one we have $c_0 = c_1 = c_2 = c_3 = \sqrt{0.25}$. The second one
    gives $c_0 = c_1 = \sqrt{0.5}$ and $c_2 = c_3 = 0$.

    The next step is to define the unitaries $U_i$ in the form of a tuple
    of functions.

    ::

        from qrisp.jasp import *
        from qrisp import *

        def case_function_0(x):
            x += 3

        def case_function_1(x):
            x += 4

        def case_function_2(x):
            x += 5

        def case_function_3(x):
            x += 6

        case_functions = (case_function_0,
                          case_function_1,
                          case_function_2,
                          case_function_3)


    These functions each represent the unitary:

    .. math::

        U_i \ket{0} = \ket{i+3}

    Executing a linear combination of unitaries therefore gives

    .. math::

        \left( \sum_{i = 0}^N c_i U_i \right) \ket{0} = \sum_{i = 0}^N c_i \ket{i+3}


    Now we implement the LCU procedure.

    ::

        # Specify the corresponding arguments of the block encoding as "static",
        # i.e. compile time constants.

        @RUS(static_argnums = [2,3])
        def block_encoding(return_size, state_preparation, case_functions):

            # This QuantumFloat will be returned
            qf = QuantumFloat(return_size)

            # Specify the QuantumVariable that indicates, which
            # case to execute
            n = int(np.ceil(np.log2(len(case_functions))))
            case_indicator = QuantumFloat(n)

            # Turn into a list of qubits
            case_indicator_qubits = [case_indicator[i] for i in range(n)]

            # Perform the LCU protocoll
            with conjugate(state_preparation)(case_indicator):
                for i in range(len(case_functions)):
                    with control(case_indicator_qubits, ctrl_state = i):
                        case_functions[i](qf)

            # Compute the success condition
            success_bool = (measure(case_indicator) == 0)

            return success_bool, qf

    Finally, evaluate via the :ref:`terminal_sampling <terminal_sampling>`
    feature:

    ::

        @terminal_sampling
        def main():
            return block_encoding(4, state_prep_full, case_functions)

        print(main())
        # Yields: {3.0: 0.25, 4.0: 0.25, 5.0: 0.25, 6.0: 0.25}

    Evaluate the other state preparation function

    ::

        @terminal_sampling
        def main():
            return block_encoding(4, state_prep_half, case_functions)

        print(main())
        # Yields: {3.0: 0.5, 4.0: 0.5}

    As expected, the full state preparation function yields a state proportional
    to

    .. math::

        \ket{3} + \ket{4} + \ket{5} + \ket{6}.

    The second state preparation gives us

    .. math::

        \ket{3} + \ket{4}.

    """
    if len(trial_function) == 0:
        return lambda x: RUS(x, **jit_kwargs)
    else:
        trial_function = trial_function[0]

    # The idea for implementing this feature is to execute the function once qached
    # to collect the output QuantumVariable object.
    # From the infered output signature the q_while_loop is constructed

    def return_function(*trial_args):

        from qrisp.core import recursive_qv_search, reset
        from qrisp.jasp import q_cond, q_while_loop

        abs_qs = TracingQuantumSession.get_instance()

        initial_gc_mode = abs_qs.gc_mode

        # Set the garbage collection mode to temporarily auto to collect
        # any ancillas that have not been deleted.
        abs_qs.gc_mode = "auto"
        # Execute the function
        qached_function = qache(trial_function, **jit_kwargs)
        first_iter_res = qached_function(*trial_args)

        abs_qs.gc_mode = initial_gc_mode

        # Filter out the static arguments
        if "static_argnums" in jit_kwargs:
            static_argnums = jit_kwargs["static_argnums"]
            if isinstance(static_argnums, int):
                static_argnums = [static_argnums]
        else:
            static_argnums = []

        if "static_argnames" in jit_kwargs:
            argname_list = inspect.getfullargspec(trial_function)
            for i in range(len(argname_list)):
                if argname_list[i] in jit_kwargs["static_argnames"]:
                    static_argnums.append(i)

        dynamic_args = []

        for i in range(len(trial_args)):
            if i not in static_argnums:
                dynamic_args.append(trial_args[i])

        n_arg_vals = len(dynamic_args)

        # Next we construct the body of the loop
        # The q_while_loop receives a tuple of arguments and
        # also returns a tuple with the same signature.
        # We therefore combine the results of the first iteration with
        # the arguments to execute the loop.

        combined_args = tuple(list(dynamic_args) + list(first_iter_res))

        # This is the body function of the while loop
        def body_fun(args):

            # The first step is to reset and delete the results
            # from the previous iteration
            qv_results = recursive_qv_search(args[n_arg_vals:])

            abs_qs = TracingQuantumSession.get_instance()
            for qv in qv_results:
                abs_qs.register_qv(qv, None)
                reset(qv)
                qv.delete()

            # We now construct the arguments for the function call of
            # the current iteration.
            # For this, we combine the dynamic arguments with the static
            # arguments.
            dynamic_args = list(args[:n_arg_vals])

            new_trial_args = []
            for i in range(len(trial_args)):
                if i not in static_argnums:
                    new_trial_args.append(dynamic_args.pop(0))
                else:
                    new_trial_args.append(trial_args[i])

            # Set the garbage collection mode to auto and call the function
            abs_qs.gc_mode = "auto"
            trial_res = qached_function(*new_trial_args)
            abs_qs.gc_mode = initial_gc_mode

            # Update the tuple with initial args and the new results
            combined_args = tuple(list(args[:n_arg_vals]) + list(trial_res))
            return combined_args

        # This is the loop cancelation condition
        def cond_fun(val):
            # The loop cancelation index is located at the second position of the
            # return value tuple
            return ~val[n_arg_vals]

        # We now evaluate the loop

        # If the first iteration was already successful, we simply return the results
        # To realize this behavior we use a q_cond primitive
        def true_fun(combined_args):
            return combined_args

        # If the first iteration was not successfull, we start the loop
        def false_fun(combined_args):
            # Here is the while_loop
            return q_while_loop(cond_fun, body_fun, init_val=combined_args)

        # Evaluate everything
        combined_res = q_cond(first_iter_res[0], true_fun, false_fun, combined_args)

        # Return the results
        if len(first_iter_res) == 2:
            return combined_res[n_arg_vals + 1]
        else:
            return combined_res[n_arg_vals + 1 :]

    return return_function
