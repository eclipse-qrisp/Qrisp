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

from qrisp.jasp.tracing_logic import quantum_kernel, check_for_tracing_mode

# The following function implements the sample feature.

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


def sample(state_prep=None, shots=0, post_processor=None):
    r"""
    The ``sample`` function allows to take samples from a state that is specified
    by a preparation procedure. This preparation procedure can be supplied via
    a Python function that returns one or more :ref:`QuantumVariables <QuantumVariable>`.

    The samples are returned in the form of a
    `Jax Array <https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html>`_
    which is shaped according to the ``shots`` parameter. Because of this, shots
    can only be a **static integer** (no dynamic values!). If you want to sample
    with a dynamic shot amount, look into :ref:`expectation_value`.


    Parameters
    ----------
    state_prep : callable
        A function returning one or more :ref:`QuantumVariables <QuantumVariable>`.
        The state from this QuantumVariables will be sampled.
        The state preparation function can only take classical values as arguments.
        This is because a quantum value would need to be copied for each sampling
        iteration, which is prohibited by the no-cloning theorem.
    shots : int
        The amounts of samples to take.
    post_processor : callable, optional
        A function to apply to the samples directly after measuring. By default no post processing is applied.

    Raises
    ------
    Exception
        Tried to sample with dynamic shots value (static integer required)
    Exception
        Tried to sample from state preparation function taking a quantum value

    Returns
    -------
    callable
        A classical, Jax traceable function returning a jax array containing
        the measurement results of each shot.

    Examples
    --------

    We prepare the state

    .. math::

        \ket{\psi} = \frac{1}{\sqrt{2}} \left(\ket{0}\ket{0}\ket{\text{True}} + \ket{k}\ket{k}\ket{\text{True}})\right)

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

    And subsequently sample from the QuantumFloats:

    ::

        @jaspify
        def main(k):

            sampling_function = sample(state_prep,
                                       shots = 10)

            return sampling_function(k)

        print(main(3))

        # Yields
        # [[3. 3.]
        #  [0. 0.]
        #  [0. 0.]
        #  [3. 3.]
        #  [0. 0.]
        #  [0. 0.]
        #  [3. 3.]
        #  [3. 3.]
        #  [0. 0.]
        #  [0. 0.]]

    To demonstrate the post processing feature, we write a simple post
    processing function:

    ::

        def post_processor(x, y):
            return 2*x + y//2

        @jaspify
        def main(k):

            sampling_function = sample(state_prep,
                                       shots = 10,
                                       post_processor = post_processor)

            return sampling_function(k)

        print(main(4))
        # Yields
        # [10. 10.  0.  0.  0.  0.  0.  0. 10. 10.]

    """

    from qrisp.jasp import qache
    from qrisp.core import QuantumVariable, measure

    if isinstance(state_prep, int):
        shots = state_prep
        state_prep = None

    if state_prep is None:
        return lambda x: sample(x, shots, post_processor=post_processor)

    if post_processor is None:

        def identity(*args):
            if len(args) == 1:
                return args[0]
            return args

        post_processor = identity

    if isinstance(shots, jax.core.Tracer):
        raise Exception(
            "Tried to sample with dynamic shots value (static integer required)"
        )
    elif not isinstance(shots, int):
        raise Exception(
            f"Tried to sample with shots value of non-integer type {type(shots)}"
        )

    # Qache the user function
    @qache
    def user_func(*args):
        return state_prep(*args)

    # This function evaluates the sampling process
    @jax.jit
    def sampling_eval_function(*args, tracerized_shots=0):

        for arg in args:
            if isinstance(arg, QuantumVariable):
                raise Exception(
                    "Tried to sample from state preparation function taking a quantum value"
                )

        # We now construct a loop to collect the samples by
        # inserting the postprocessed measurement result into an array.
        # The following function is the loop body, which is kernelized.
        @quantum_kernel
        def sampling_body_func(i, args):

            acc = args[0]

            # Evaluate the user function
            qv_tuple = user_func(*args[1:])

            if not isinstance(qv_tuple, tuple):
                qv_tuple = (qv_tuple,)

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
            def sampling_helper_2(acc, i, *meas_ints):
                decoded_values = []
                for j in range(len(qv_tuple)):
                    decoded_values.append(qv_tuple[j].jdecoder(meas_ints[j]))

                if len(qv_tuple) > 1:
                    decoded_values = post_processor(*decoded_values)
                else:
                    decoded_values = post_processor(*decoded_values)

                if isinstance(decoded_values, tuple):
                    # Save the return amount (for more details check the comment of the)
                    # initialization command of return_amount
                    return_amount.append(len(decoded_values))
                    if len(acc.shape) == 1:
                        raise AuxException()

                # Insert into the accumulating array
                acc = acc.at[i].set(decoded_values)

                return acc

            acc = sampling_helper_2(acc, i, *measurement_ints)

            return (acc, *args[1:])

        # This list captures the amount of return values. The strategy here is
        # to initially assume only one QuantumVariable is returned, which is then
        # added to the expectation value accumulator. If more than one is returned,
        # the amount is saved in this list and an exception is raised, which
        # subsequently causes another call but this time with the correct accumulator
        # dimension.

        return_amount = []

        try:
            loop_res = jax.lax.fori_loop(
                0, tracerized_shots, sampling_body_func, (jnp.zeros(shots), *args)
            )
            return loop_res[0]
        except AuxException:
            loop_res = jax.lax.fori_loop(
                0,
                tracerized_shots,
                sampling_body_func,
                (jnp.zeros((shots, return_amount[0])), *args),
            )
            return loop_res[0]

    from qrisp.jasp import terminal_sampling

    def return_function(*args):

        if check_for_tracing_mode():
            return sampling_eval_function(*args, tracerized_shots=shots)
        else:
            return terminal_sampling(state_prep, shots)(*args)

    return return_function


class AuxException(Exception):
    pass
