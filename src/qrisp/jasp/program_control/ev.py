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
from jax.tree_util import tree_leaves, tree_structure, tree_unflatten

from qrisp.jasp.tracing_logic import quantum_kernel


@jax.jit
def _backend_shots_marker(val):
    """Identity marker so that ``backend_sampler`` can reliably locate the
    shot count inside a traced expectation_value Jaxpr."""
    return val


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


def expectation_value(sampling_kernel, shots, return_dict=False, post_processor=None):
    r"""The ``expectation_value`` function allows to estimate the expectation value
    from a *sampling kernel* — a Python function that receives only classical
    arguments and returns arbitrary values.  Any
    :ref:`QuantumVariables <QuantumVariable>` in the return are automatically
    measured and decoded; classical values are interleaved in-place.

    Parameters
    ----------
    sampling_kernel : callable
        A sampling kernel — a function receiving only classical arguments and
        returning one or more :ref:`QuantumVariables <QuantumVariable>`,
        classical measurement results, or a mixture of both.
        The function may **not** receive quantum arguments because a quantum
        value would need to be copied for each sampling iteration, which is
        prohibited by the no-cloning theorem.
    shots : int or jax.core.Tracer
        The amount of samples to take to compute the expectation value.
    post_processor : callable, optional
        A classical Jax traceable function to apply to the results
        directly after measuring. By default no post processing is applied.

    Raises
    ------
    Exception
        Tried to sample from sampling kernel taking a quantum value

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


        def sampling_kernel(k):
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

            ev_function = expectation_value(sampling_kernel, shots = 50)

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

            ev_function = expectation_value(sampling_kernel, shots = 50)

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
            if len(args) == 1:
                return args[0]
            return args

        post_processor = identity

    # Qache the user function
    @qache
    def user_func(*args):
        return sampling_kernel(*args)

    # This function performs the logic to evaluate the expectation value
    def expectation_value_eval_function(*args, shots=0):

        for arg in args:
            if isinstance(arg, QuantumVariable):
                raise Exception("Tried to sample from state preparation function taking a quantum value")

        # Marker: allows backend_sampler to locate the shot count in the
        # traced Jaxpr without fragile position-based extraction.
        _backend_shots_marker(shots)

        # -----------------------------------------------------------------
        # Typed pytree accumulator strategy (mirrors sampling.py)
        # -----------------------------------------------------------------
        # The sampling kernel may return a single value or a pytree
        # container (tuple, list, dict).  Instead of squeezing everything
        # into one flat accumulator array (which forces a common dtype),
        # we build a tuple of typed running-sum accumulators — one per
        # leaf, each with the leaf's native shape and dtype.  The pytree
        # structure is captured on the first iteration; AuxException
        # restarts the loop with the correctly-shaped accumulators.
        # After the loop each accumulator is divided by *shots* and the
        # pytree is reconstructed via tree_unflatten.

        def _update_acc(acc, decoded_values, return_amount):
            """Add *decoded_values* into *acc* (running sum).

            For pytree containers captures structure/dtypes/shapes on the
            first iteration, raises AuxException if *acc* is still a
            plain 1D array, and adds each leaf into its typed accumulator.
            Scalar returns take the fast path ``acc + jnp.array(value)``.
            """
            struct = tree_structure(decoded_values)
            if struct.num_nodes > 1:  # pytree container
                if not isinstance(decoded_values, (tuple, list, dict)):
                    raise TypeError(
                        f"Unsupported return type {type(decoded_values).__name__!r}. "
                        f"expectation_value() supports tuple, list, and dict "
                        f"containers.  Convert the return value to one of these "
                        f"types."
                    )
                flat_values = tree_leaves(decoded_values)

                if not return_amount:
                    leaf_dtypes = []
                    leaf_shapes = []
                    for v in flat_values:
                        try:
                            leaf_dtypes.append(v.dtype)
                            leaf_shapes.append(v.shape)
                        except AttributeError:
                            leaf_dtypes.append(None)
                            leaf_shapes.append(())
                    return_amount.append((struct, leaf_dtypes, leaf_shapes))

                if not isinstance(acc, tuple):
                    raise AuxException()

                return tuple(a + v for a, v in zip(acc, flat_values))

            # ----------------------------------------------------------
            # Single leaf (scalar or array — not a container).
            # True scalars take the fast path; arrays with shape need a
            # shaped accumulator, captured via AuxException.
            # ----------------------------------------------------------
            if not isinstance(acc, tuple):
                try:
                    leaf_shape = decoded_values.shape
                except AttributeError:
                    leaf_shape = ()
                if leaf_shape == ():
                    return acc + jnp.array(decoded_values)  # true scalar

                # Non-scalar leaf array — capture shape and retry
                if not return_amount:
                    leaf_dtype = getattr(decoded_values, "dtype", None)
                    return_amount.append((struct, [leaf_dtype], [leaf_shape]))
                raise AuxException()

            # Second pass: acc is a 1-tuple of shaped accumulators
            return tuple(a + v for a, v in zip(acc, [decoded_values]))

            # Second pass: acc is a 1-tuple of shaped accumulators
            return tuple(a + v for a, v in zip(acc, [decoded_values]))

        def _make_init_acc(leaf_dtypes, leaf_shapes):
            """Build a tuple of typed zero-arrays, one per leaf.

            Each accumulator has the same shape as its leaf (running sum,
            not per-shot storage).
            """
            return tuple(
                jnp.zeros(shape, dtype=dt) if dt is not None else jnp.zeros(shape)
                for dt, shape in zip(leaf_dtypes, leaf_shapes)
            )

        # We now construct a loop to evaluate the expectation value via adding
        # the decoded and postprocessed measurement result into an accumulator.
        # The following function is the loop body, which is kernelized.
        @quantum_kernel
        def sampling_body_func(i, args):

            # Evaluate the user function
            acc = args[0]
            result_tuple = user_func(*args[1:])

            if not isinstance(result_tuple, tuple):
                result_tuple = (result_tuple,)

            # Build a per-position mask: QuantumVariable -> True, classical -> False.
            is_quantum = [isinstance(x, QuantumVariable) for x in result_tuple]

            # Separate quantum and classical returns.
            qv_tuple = tuple(x for x, is_q in zip(result_tuple, is_quantum) if is_q)
            classical_tuple = tuple(x for x, is_q in zip(result_tuple, is_quantum) if not is_q)

            if qv_tuple:
                # Measure quantum registers only.
                @qache
                def sampling_helper_1(*args):
                    res_list = []
                    for reg in args:
                        res_list.append(measure(reg))
                    return tuple(res_list)

                measurement_ints = sampling_helper_1(*[qv.reg for qv in qv_tuple])

                # Decode quantum, interleave with classical values, apply
                # post-processing.  Classical values are passed as explicit
                # arguments (before measurement ints).  When present the
                # helper is named sampling_helper_2_mixed for detection.
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

                        return post_processor(*full)

                    sampling_helper_2 = jax.jit(sampling_helper_2_mixed)
                else:

                    def sampling_helper_2(*meas_ints):
                        res_list = []
                        for j in range(len(qv_tuple)):
                            res_list.append(qv_tuple[j].jdecoder(meas_ints[j]))
                        return post_processor(*res_list)

                    sampling_helper_2 = jax.jit(sampling_helper_2)

                decoded_values = sampling_helper_2(*classical_tuple, *measurement_ints)

            else:
                # Pure classical — just apply post-processing directly.
                decoded_values = post_processor(*classical_tuple)

            # Update the accumulator (handles scalar / pytree returns).
            if _use_pytree_acc:
                acc = _update_acc(acc, decoded_values, return_amount)
            else:
                # Legacy flat-array path (dict_sampling_eval_function)
                if isinstance(decoded_values, tuple) and len(decoded_values) != 1:
                    return_amount.append(len(decoded_values))
                    if acc.shape[0] == 1:
                        raise AuxException()
                meas_res = jnp.array(decoded_values)
                acc += meas_res

            # Return the updated accumulator for the next loop iteration.
            return (acc, *args[1:])

        # On the first iteration the pytree structure and leaf dtypes/shapes
        # of the return values are captured.  AuxException triggers a retry
        # with a tuple of typed accumulators.  After the loop each
        # accumulator is divided by *shots* and the pytree is reconstructed.

        return_amount = []

        try:
            loop_res = jax.lax.fori_loop(0, shots, sampling_body_func, (jnp.zeros(1), *args))
            return loop_res[0][0] / shots
        except AuxException:
            if _use_pytree_acc:
                struct, leaf_dtypes, leaf_shapes = return_amount[0]
                init_acc = _make_init_acc(leaf_dtypes, leaf_shapes)
                loop_res = jax.lax.fori_loop(
                    0,
                    shots,
                    sampling_body_func,
                    (init_acc, *args),
                )
                acc_tuple = loop_res[0]
                means = tuple(a / shots for a in acc_tuple)
                return tree_unflatten(struct, means)
            else:
                loop_res = jax.lax.fori_loop(0, shots, sampling_body_func, (jnp.zeros(return_amount), *args))
                return loop_res[0] / shots

    if return_dict:
        expectation_value_eval_function.__name__ = "dict_sampling_eval_function"
        _use_pytree_acc = False  # keep legacy flat-array format
    else:
        _use_pytree_acc = True

    def return_function(*args):
        return jax.jit(expectation_value_eval_function)(*args, shots=shots)

    return return_function


class AuxException(Exception):
    pass
