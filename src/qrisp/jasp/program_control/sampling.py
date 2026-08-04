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

from qrisp.jasp.tracing_logic import check_for_tracing_mode, quantum_kernel


@jax.jit
def _backend_shots_marker(val):
    """Identity marker so that ``backend_sampler`` can reliably locate the
    shot count inside a traced ``sampling_eval_function`` Jaxpr."""
    return val


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


def sample(sampling_kernel=None, shots=0, post_processor=None):
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
    sampling_kernel : callable
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
        A classical, Jax traceable function.  For a kernel returning a single
        value the result is a 1D ``jax.Array`` of length ``shots``.  For a
        kernel returning a container (``tuple``, ``list``, ``dict``, or nested
        combinations thereof) each leaf is replaced by a 1D array preserving
        its native dtype — e.g. ``{'a': bool_array, 'b': float_array}``.

    Examples
    --------
    We prepare the state

    .. math::

        \ket{\psi} = \frac{1}{\sqrt{2}} \left(\ket{0}\ket{0}\ket{\text{True}} + \ket{k}\ket{k}\ket{\text{True}})\right)

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

    And subsequently sample from the QuantumFloats:

    ::

        @jaspify
        def main(k):

            sampling_function = sample(sampling_kernel,
                                       shots = 10)

            return sampling_function(k)

        print(main(3))

        # Yields (a tuple of two 1D arrays)
        # (Array([3., 0., 0., 3., 0., 0., 3., 3., 0., 0.], dtype=float64),
        #  Array([3., 0., 0., 3., 0., 0., 3., 3., 0., 0.], dtype=float64))

    To demonstrate the post processing feature, we write a simple post
    processing function:

    ::

        def post_processor(x, y):
            return 2*x + y//2

        @jaspify
        def main(k):

            sampling_function = sample(sampling_kernel,
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
        # Yields e.g. (a tuple of two 1D arrays):
        # (Array([0., 0., 1., ...], dtype=float64),
        #  Array([0., 0., 0., ...], dtype=float64))

    .. note::

        The above example uses ``@jaspify`` (which defaults to
        ``terminal_sampling=False``).  Using ``@jaspify(terminal_sampling=True)``
        with a mixed-returns kernel will raise an error.

    """
    from qrisp.core import QuantumVariable, measure
    from qrisp.jasp import qache

    if isinstance(sampling_kernel, int):
        shots = sampling_kernel
        sampling_kernel = None

    if sampling_kernel is None:
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
        return sampling_kernel(*args)

    # This function evaluates the sampling process
    @jax.jit
    def sampling_eval_function(*args, tracerized_shots=0):

        for arg in args:
            if isinstance(arg, QuantumVariable):
                raise Exception("Tried to sample from state preparation function taking a quantum value")

        # Marker: allows backend_sampler to locate the shot count in the
        # traced Jaxpr without fragile position-based extraction.
        _backend_shots_marker(tracerized_shots)

        # -----------------------------------------------------------------
        # Typed pytree accumulator strategy
        # -----------------------------------------------------------------
        # The user's sampling kernel may return a single value or any
        # JAX-supported pytree container — flat tuple, nested tuple,
        # list, dict, or combinations thereof (e.g.
        # ``{'x': (a, b), 'y': c}``).  We want the final result to mirror
        # that structure, with each leaf replaced by a 1D JAX array of
        # length *shots* preserving the leaf's native dtype.
        #
        # To avoid forcing all return values into a single flat 2D
        # accumulator (which would coerce everything to float64), we
        # build a **tuple of typed 1D accumulators** — one per leaf.
        # The pytree structure is captured on the first iteration via
        # ``tree_structure`` / ``tree_leaves`` and stored in
        # ``return_amount``.  An ``AuxException`` then restarts the loop
        # with the correctly-shaped accumulator tuple.
        #
        # On every iteration each leaf accumulator is updated
        # independently with the corresponding decoded value, so
        # ``bool`` stays ``bool``, ``int`` stays ``int``, etc.
        # After the loop ``tree_unflatten`` rebuilds the original
        # nested shape from the accumulator tuple — no post-hoc
        # slicing or dtype casting needed.
        #
        # The two tiny helpers below encapsulate this logic so that
        # the main loop body stays focused on quantum operations.
        # -----------------------------------------------------------------

        def _update_acc(acc, i, decoded_values, return_amount):
            """Update *acc* at index *i* with *decoded_values*.

            For pytree containers (``tuple``, ``list``, ``dict``) this
            captures the structure and per-leaf dtypes/shapes on the first
            iteration, raises :class:`AuxException` if *acc* is still a
            plain 1D array, and updates a tuple of typed per-leaf
            accumulators.  Each leaf accumulator has shape
            ``(shots, *leaf_shape)`` so that array-valued leaves (e.g. a
            ``(3,)`` array inside a tuple) naturally become
            ``(shots, 3)`` in the result.
            User-defined pytree types are rejected with a clear error.
            Scalar returns take the fast path ``acc.at[i].set(value)``.
            """
            struct = tree_structure(decoded_values)
            if struct.num_nodes > 1:  # pytree container
                if not isinstance(decoded_values, (tuple, list, dict)):
                    raise TypeError(
                        f"Unsupported return type {type(decoded_values).__name__!r}. "
                        f"sample() supports tuple, list, and dict containers. "
                        f"Convert the return value to one of these types."
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

                return tuple(a.at[i].set(v) for a, v in zip(acc, flat_values))

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
                    return acc.at[i].set(decoded_values)  # true scalar

                # Non-scalar leaf array — capture shape and retry
                if not return_amount:
                    leaf_dtype = getattr(decoded_values, "dtype", None)
                    return_amount.append((struct, [leaf_dtype], [leaf_shape]))
                raise AuxException()

            # Second pass: acc is a 1-tuple of shaped accumulators
            return tuple(a.at[i].set(v) for a, v in zip(acc, [decoded_values]))

        def _make_init_acc(shots, leaf_dtypes, leaf_shapes):
            """Build a tuple of typed zero-arrays, one per leaf.

            Each accumulator has shape ``(shots, *leaf_shape)`` so that
            array-valued leaves are stacked along the leading dimension.
            """
            return tuple(
                jnp.zeros((shots,) + shape, dtype=dt) if dt is not None else jnp.zeros((shots,) + shape)
                for dt, shape in zip(leaf_dtypes, leaf_shapes)
            )

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

                decoded_values = result

            # Update the accumulator (handles scalar / tuple / list /
            # nested returns transparently via typed per-leaf arrays).
            acc = _update_acc(acc, i, decoded_values, return_amount)

            return (acc, *args[1:])

        # On the first iteration the pytree structure and leaf dtypes of
        # the return values are captured.  AuxException triggers a retry
        # with a tuple of typed 1D accumulators.  After the loop the
        # accumulator tuple is reconstructed into the original nested
        # shape via tree_unflatten.

        return_amount = []

        try:
            loop_res = jax.lax.fori_loop(0, tracerized_shots, sampling_body_func, (jnp.zeros(shots), *args))
            return loop_res[0]
        except AuxException:
            struct, leaf_dtypes, leaf_shapes = return_amount[0]
            init_acc = _make_init_acc(shots, leaf_dtypes, leaf_shapes)
            loop_res = jax.lax.fori_loop(
                0,
                tracerized_shots,
                sampling_body_func,
                (init_acc, *args),
            )
            return tree_unflatten(struct, loop_res[0])

    from qrisp.jasp import terminal_sampling

    def return_function(*args):

        if check_for_tracing_mode():
            if shots <= 0:
                raise ValueError(f"shots must be a positive integer, got {shots}")
            return sampling_eval_function(*args, tracerized_shots=shots)
        else:
            return terminal_sampling(sampling_kernel, shots)(*args)

    return return_function


class AuxException(Exception):
    pass
