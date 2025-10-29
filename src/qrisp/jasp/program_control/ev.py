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

import jax
import jax.numpy as jnp

from qrisp.jasp.tracing_logic import quantum_kernel

# The following function implements the expectation_value feature.
# The basic functionality would be relatively straightforward to implement,
# however there are some complications. The reason for that is that the resulting
# jaxpr should be "readable" by the terminal sampling interpreter.
# Terminal sampling means that instead of performing the simulations "shots"-times
# it is performed once and the shots are then sampled from that distribution.
# Naturally this implies a massive performance increase, which is why a lot
# of effort is spent to realize a smooth implementation.

# The underlying idea to make the feature easily "readable" by the terminal
# sampling interpreter is to structure one iteration of sampling into three
# steps.

# 1. Evaluating the user function, which generates the distribution.
# 2. Sampling from that distribution via the "measure" function.
# 3. Decoding and postprocessing the measurement results.

# For the final two steps we deploy some custom logic to realize the terminal
# sampling behavior. To simplify the automatic processing of these steps,
# we capture each into individual pjit calls.

# The terminal sampling interpreter then identifies each steps via the
# eqn.params["name"] attribute and executes the custom logic.


def expectation_value(state_prep, shots, return_dict=False, post_processor=None):
    r"""
    The ``expectation_value`` function allows to estimate the expectation value
    from a state that is specified by a preparation procedure. This preparation
    procedure can be supplied via a Python function that returns one or
    more :ref:`QuantumVariables <QuantumVariable>`.

    Parameters
    ----------
    state_prep : callable
        A function returning one or more :ref:`QuantumVariables <QuantumVariable>`.
        The expectation value from this state will be computed.
        The state preparation function can only take classical values as arguments.
        This is because a quantum value would need to be copied for each sampling
        iteration, which is prohibited by the no-cloning theorem.
    shots : int or jax.core.Tracer
        The amount of samples to take to compute the expectation value.
    post_processor : callable, optional
        A classical Jax traceable function to apply to the results
        directly after measuring. By default no post processing is applied.

    Raises
    ------
    Exception
        Tried to sample from state preparation function taking a quantum value

    Returns
    -------
    callable
        A function returning a Jax array containing the expectation value.

    Examples
    --------

    We prepare the state

    .. math::

        \ket{\psi_k} = \frac{1}{\sqrt{2}} \left(\ket{0}\ket{0}\ket{\text{False}} + \ket{k}\ket{k}\ket{\text{True}}\right)

    ::

        from qrisp import *
        from qrisp.jasp import *


        def state_prep(k):
            a = QuantumFloat(4)
            b = QuantumFloat(4)

            qbl = QuantumBool()
            h(qbl)

            with control(qbl[0]):
                a[:] = k

            cx(a, b)

            return a, b

    And compute the expectation value of the QuantumFloats

    ::

        @jaspify
        def main(k):

            ev_function = expectation_value(state_prep, shots = 50)

            return ev_function(k)

        print(main(3))
        # Yields
        # [1.44 1.44]

    The true value 1.5 is not reached because of `shot noise <https://en.wikipedia.org/wiki/Shot_noise>`_.
    To improve the approximation, feel free to increase the shots!

    To demonstrate the ``post_processor`` keyword we define a simple post processing
    function

    ::

        def post_processor(x, y):
            return x*y

        @jaspify
        def main(k):

            ev_function = expectation_value(state_prep, shots = 50)

            return ev_function(k)

        print(main(3))
        # Yields
        # 4.338

    This result is expected because the inputs of ``post_processor`` are
    either (0,0) or (3,3) with 50% probability, so we get

    .. math::

        4.5 = \frac{3\cdot 3 + 0\cdot 0}{2}


    """

    from qrisp.core import QuantumVariable, measure
    from qrisp.jasp import make_tracer, qache

    if isinstance(shots, int):
        shots = make_tracer(shots)

    if post_processor is None:

        def identity(*args):
            return args

        post_processor = identity

    # Qache the user function
    @qache
    def user_func(*args):
        return state_prep(*args)

    # This function performs the logic to evaluate the expectation value
    def expectation_value_eval_function(*args, shots=0):

        for arg in args:
            if isinstance(arg, QuantumVariable):
                raise Exception(
                    "Tried to sample from state preparation function taking a quantum value"
                )

        # We now construct a loop to evaluate the expectation value via adding
        # the decoded and postprocessed measurement result into an accumulator.
        # The following function is the loop body, which is kernelized.
        @quantum_kernel
        def sampling_body_func(i, args):

            # Evaluate the user function
            acc = args[0]
            qv_tuple = user_func(*args[1:])

            if not isinstance(qv_tuple, tuple):
                qv_tuple = (qv_tuple,)

            # Ensure all results are QuantumVariables
            for qv in qv_tuple:
                if not isinstance(qv, QuantumVariable):
                    raise Exception(
                        "Tried to sample from function not returning a QuantumVariable"
                    )

            # Trace the DynamicQubitArray measurements
            # Since we execute the measurements on the .reg attribute, no decoding
            # is applied. The decoding happens in sampling_helper_2
            @qache
            def sampling_helper_1(*args):
                res_list = []
                for reg in args:
                    res_list.append(measure(reg))
                return tuple(res_list)

            measurement_ints = sampling_helper_1(*[qv.reg for qv in qv_tuple])

            # Trace the decoding
            @jax.jit
            def sampling_helper_2(*meas_ints):
                res_list = []
                for i in range(len(qv_tuple)):
                    res_list.append(qv_tuple[i].jdecoder(meas_ints[i]))

                # Apply the post processing
                return post_processor(*res_list)

            decoded_values = sampling_helper_2(*(list(measurement_ints)))

            if isinstance(decoded_values, tuple) and len(decoded_values) != 1:
                # Save the return amount (for more details check the comment of the)
                # initialization command of return_amount
                return_amount.append(len(decoded_values))
                if acc.shape[0] == 1:
                    raise AuxException()

            # Turn into jax array and add to the accumulator
            meas_res = jnp.array(decoded_values)
            acc += meas_res

            # Return the updated accumulator for the next loop iteration.
            return (acc, *args[1:])

        # This list captures the amount of return values. The strategy here is
        # to initially assume only one QuantumVariable is returned, which is then
        # added to the expectation value accumulator. If more than one is returned,
        # the amount is saved in this list and an exception is raised, which
        # subsequently causes another call but this time with the correct accumulator
        # dimension.

        return_amount = []

        try:
            # loop_res = jax.lax.fori_loop(0, shots, sampling_body_func, (jax.lax.broadcast(0., (1,)), *args))
            loop_res = jax.lax.fori_loop(
                0, shots, sampling_body_func, (jnp.zeros(1), *args)
            )
            return loop_res[0][0] / shots
        except AuxException:
            loop_res = jax.lax.fori_loop(
                0, shots, sampling_body_func, (jnp.zeros(return_amount), *args)
            )
            return loop_res[0] / shots

    if return_dict:
        expectation_value_eval_function.__name__ = "dict_sampling_eval_function"

    def return_function(*args):
        return jax.jit(expectation_value_eval_function)(*args, shots=shots)

    return return_function


class AuxException(Exception):
    pass
