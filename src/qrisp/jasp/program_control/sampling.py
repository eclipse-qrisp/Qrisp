"""********************************************************************************
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

import jax
import jax.numpy as jnp

from qrisp.jasp.tracing_logic import check_for_tracing_mode, quantum_kernel

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
    r"""The ``sample`` function allows to take samples from a quantum computation
    specified by a *sampling kernel* — a Python function that receives only
    classical arguments and returns arbitrary values.  Any
    :ref:`QuantumVariables <QuantumVariable>` in the return are automatically
    measured and decoded.

    The samples are returned in the form of a
    `Jax Array <https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html>`_
    which is shaped according to the ``shots`` parameter. Because of this, shots
    can only be a **static integer** (no dynamic values!). If you want to sample
    with a dynamic shot amount, look into :ref:`expectation_value`.

    Sample calls can be efficiently simulated via terminal sampling by setting the
    corresponding keyword within :func:`~qrisp.jasp.jaspify` to ``True``.

    .. note::

        **Terminal sampling** (``terminal_sampling=True`` inside
        :func:`~qrisp.jasp.jaspify`) does **not** support sampling kernels
        that return classical values alongside quantum variables, or kernels
        that return *only* classical values.  Use ``terminal_sampling=False``
        for those cases.

    Parameters
    ----------
    state_prep : callable
        A sampling kernel — a function receiving only classical arguments and
        returning one or more :ref:`QuantumVariables <QuantumVariable>`,
        classical measurement results, or a mixture of both.
        The function may **not** receive quantum arguments because a quantum
        value would need to be copied for each sampling iteration, which is
        prohibited by the no-cloning theorem.
    shots : int
        The amounts of samples to take.
    post_processor : callable, optional
        A function to apply to the samples directly after measuring. By default
        no post processing is applied.

    Raises
    ------
    Exception
        Tried to sample with dynamic shots value (static integer required)
    Exception
        Tried to sample from sampling kernel taking a quantum value
    Exception
        Tried to use terminal sampling with a kernel that returns classical
        values (use ``terminal_sampling=False`` instead)

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

    **Sampling kernels returning classical values**

    A sampling kernel may also return classical values from mid-circuit
    measurements alongside (or instead of) quantum variables:

    ::

        def mixed_kernel():
            qf = QuantumFloat(4)
            h(qf[0])
            h(qf[1])
            mes = measure(qf[1])      # classical measurement result
            return qf, mes            # mixed: quantum + classical

        @jaspify
        def main():
            return sample(mixed_kernel, shots=20)()

        print(main())
        # Yields e.g.:
        # [[0. 0.]
        #  [0. 0.]
        #  [1. 0.]
        #  ...]

    .. note::

        The above example uses ``@jaspify`` (which defaults to
        ``terminal_sampling=False``).  Using ``@jaspify(terminal_sampling=True)``
        with a mixed-returns kernel will raise an error.

    """
    from qrisp.core import QuantumVariable, measure
    from qrisp.jasp import qache

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
        raise Exception("Tried to sample with dynamic shots value (static integer required)")
    elif not isinstance(shots, int):
        raise Exception(f"Tried to sample with shots value of non-integer type {type(shots)}")

    # Qache the user function
    @qache
    def user_func(*args):
        return state_prep(*args)

    # This function evaluates the sampling process
    @jax.jit
    def sampling_eval_function(*args, tracerized_shots=0):

        for arg in args:
            if isinstance(arg, QuantumVariable):
                raise Exception("Tried to sample from state preparation function taking a quantum value")

        # We now construct a loop to collect the samples by
        # inserting the postprocessed measurement result into an array.
        # The following function is the loop body, which is kernelized.
        @quantum_kernel
        def sampling_body_func(i, args):

            acc = args[0]

            # Evaluate the user function
            result_tuple = user_func(*args[1:])

            if not isinstance(result_tuple, tuple):
                result_tuple = (result_tuple,)

            # Build a per-position mask: QuantumVariable -> True, classical -> False.
            is_quantum = [isinstance(x, QuantumVariable) for x in result_tuple]

            # Separate quantum and classical returns.
            qv_tuple = tuple(x for x, is_q in zip(result_tuple, is_quantum) if is_q)
            classical_tuple = tuple(x for x, is_q in zip(result_tuple, is_quantum) if not is_q)

            if qv_tuple:
                # ----------------------------------------------------------
                # Stage 2: measure quantum registers only.
                # ----------------------------------------------------------
                @qache
                def sampling_helper_1(*args):
                    res_list = []
                    for reg in args:
                        res_list.append(measure(reg))
                    return tuple(res_list)

                measurement_ints = sampling_helper_1(*[qv.reg for qv in qv_tuple])

                # ----------------------------------------------------------
                # Stage 3: decode quantum, interleave with classical values,
                # apply post-processing.  Classical values are passed as
                # explicit arguments (before measurement ints).  When present
                # the helper is named sampling_helper_2_mixed so that the
                # terminal-sampling guard in jaspification.py can detect it.
                # ----------------------------------------------------------
                if classical_tuple:
                    def sampling_helper_2_mixed(*args):
                        n_classical = len(classical_tuple)
                        classical_vals = args[:n_classical]
                        meas_ints = args[n_classical:]

                        decoded_q = []
                        for j in range(len(qv_tuple)):
                            decoded_q.append(qv_tuple[j].jdecoder(meas_ints[j]))

                        full = []
                        q_idx = 0
                        c_idx = 0
                        for is_q in is_quantum:
                            if is_q:
                                full.append(decoded_q[q_idx])
                                q_idx += 1
                            else:
                                full.append(classical_vals[c_idx])
                                c_idx += 1

                        if len(full) > 1:
                            result = post_processor(*full)
                        else:
                            result = post_processor(*full)

                        if isinstance(result, tuple):
                            return_amount.append(len(result))
                            if len(acc.shape) == 1:
                                raise AuxException()
                        return result
                    sampling_helper_2 = jax.jit(sampling_helper_2_mixed)
                else:
                    def sampling_helper_2(*meas_ints):
                        decoded_q = []
                        for j in range(len(qv_tuple)):
                            decoded_q.append(qv_tuple[j].jdecoder(meas_ints[j]))

                        full = []
                        q_idx = 0
                        c_idx = 0
                        for is_q in is_quantum:
                            if is_q:
                                full.append(decoded_q[q_idx])
                                q_idx += 1
                            else:
                                full.append(classical_tuple[c_idx])
                                c_idx += 1

                        if len(full) > 1:
                            result = post_processor(*full)
                        else:
                            result = post_processor(*full)

                        if isinstance(result, tuple):
                            return_amount.append(len(result))
                            if len(acc.shape) == 1:
                                raise AuxException()
                        return result
                    sampling_helper_2 = jax.jit(sampling_helper_2)

                decoded_values = sampling_helper_2(*classical_tuple, *measurement_ints)

            else:
                # ----------------------------------------------------------
                # No quantum returns — pure classical.  No measurement or
                # decoding needed; just apply post-processing directly.
                # ----------------------------------------------------------
                if len(classical_tuple) > 1:
                    result = post_processor(*classical_tuple)
                else:
                    result = post_processor(*classical_tuple)

                if isinstance(result, tuple):
                    return_amount.append(len(result))
                    if len(acc.shape) == 1:
                        raise AuxException()
                decoded_values = result

            # Insert into the accumulating array
            acc = acc.at[i].set(decoded_values)

            return (acc, *args[1:])

        # This list captures the amount of return values. The strategy here is
        # to initially assume only one QuantumVariable is returned, which is then
        # added to the expectation value accumulator. If more than one is returned,
        # the amount is saved in this list and an exception is raised, which
        # subsequently causes another call but this time with the correct accumulator
        # dimension.

        return_amount = []

        try:
            loop_res = jax.lax.fori_loop(0, tracerized_shots, sampling_body_func, (jnp.zeros(shots), *args))
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
