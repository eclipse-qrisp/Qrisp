# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Shared sampling loop of :func:`~qrisp.jasp.sample`, :func:`~qrisp.jasp.expectation_value` and terminal sampling.

All three repeat the same step once per shot and differ only in how the
results of the individual shots are combined (see ``_MODES``):

* ``sample`` stores every shot,
* ``expectation_value`` sums the shots and divides by their number,
* :func:`~qrisp.jasp.terminal_sampling` traces the same loop, but the terminal
  sampling interpreter replaces its result with a dict of outcomes.

The traced Jaxpr has to be readable by the terminal sampling interpreter and
by ``backend_sampler``.  Terminal sampling means that instead of performing
the simulation *shots* times, it is performed once and the shots are then
sampled from the resulting distribution, which is much faster.  To make this
possible, one shot is structured into three steps, each captured in a named
function call:

1. ``user_func`` evaluates the sampling kernel, which prepares the state.
2. ``sampling_helper_1`` measures the returned registers.
3. ``sampling_helper_2`` decodes the measurement results and applies the
   post-processor.  It is named ``sampling_helper_2_mixed`` if the kernel also
   returns classical values, which terminal sampling cannot handle.

The interpreters identify these steps, the loop body ``sampling_body_func``
and the surrounding eval function (named after the mode) by their names, and
take the shot count from the eval function's last argument.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

from qrisp.jasp.tracing_logic import qache, quantum_kernel


class _Mode(NamedTuple):
    """How the results of the individual shots are combined.

    Attributes
    ----------
    eval_name : str
        The name of the traced eval function.  The interpreters dispatch on it.
    init : callable
        ``init(leaves, static_shots)`` returns the initial accumulators, one
        per leaf of the post-processed result.  *leaves* holds an array of the
        leaf's shape and dtype for every leaf.
    update : callable
        ``update(acc, i, leaves)`` adds the leaves of shot *i* to *acc*.
    finalize : callable
        ``finalize(acc, treedef, shots)`` turns the accumulators into the
        result.

    """

    eval_name: str
    init: Callable
    update: Callable
    finalize: Callable


def _init_sums(leaves, static_shots):
    """Initialize one running sum per leaf.

    An expectation value is a mean, so the sums are floating point regardless
    of the leaf's own dtype.  Complex leaves keep theirs, since casting them to
    float would drop the imaginary part.
    """
    return tuple(
        jnp.zeros(leaf.shape, dtype=leaf.dtype if jnp.issubdtype(leaf.dtype, jnp.complexfloating) else None)
        for leaf in leaves
    )


def _add(acc, i, leaves):
    """Add the leaves of one shot to the running sums."""
    return tuple(a + v for a, v in zip(acc, leaves))


_MODES = {
    # sample(): one row per shot, every leaf keeps its native dtype, and
    # array-valued leaves stack along the leading dimension.
    "sample": _Mode(
        eval_name="sampling_eval_function",
        init=lambda leaves, static_shots: tuple(jnp.zeros((static_shots, *leaf.shape), leaf.dtype) for leaf in leaves),
        update=lambda acc, i, leaves: tuple(a.at[i].set(v) for a, v in zip(acc, leaves)),
        finalize=lambda acc, treedef, shots: tree_unflatten(treedef, acc),
    ),
    # expectation_value(): the mean of every leaf.
    "ev": _Mode(
        eval_name="expectation_value_eval_function",
        init=_init_sums,
        update=_add,
        finalize=lambda acc, treedef, shots: tree_unflatten(treedef, [a / shots for a in acc]),
    ),
    # terminal_sampling(): the terminal sampling interpreter replaces the
    # result with a dict of outcomes, so the function only needs a single
    # output that depends on the decoded values.
    "dict": _Mode(
        eval_name="dict_sampling_eval_function",
        init=_init_sums,
        update=_add,
        finalize=lambda acc, treedef, shots: jnp.concatenate([jnp.ravel(a) for a in acc]),
    ),
}


def _sampling_procedure(sampling_kernel, post_processor, mode, static_shots=None):
    """Build the jitted function performing the sampling loop.

    Parameters
    ----------
    sampling_kernel : callable
        The sampling kernel.
    post_processor : callable or None
        Applied to the decoded results of every shot.  ``None`` returns them
        unchanged.
    mode : str
        A key of ``_MODES``: ``"sample"``, ``"ev"`` or ``"dict"``.
    static_shots : int, optional
        The shot count as a static integer.  Only the ``"sample"`` mode needs
        it, to allocate one row per shot.

    Returns
    -------
    callable
        A jitted function ``f(*args, shots)`` evaluating the sampling kernel
        with the classical arguments *args*.  *shots* has to be passed as a
        keyword, so that it is the last argument of the traced function, where
        the interpreters expect it.

    """
    from qrisp.core import QuantumVariable

    mode = _MODES[mode]

    if post_processor is None:
        post_processor = _identity

    @qache
    def user_func(*args):
        return sampling_kernel(*args)

    def eval_function(*args, shots):
        for arg in args:
            if isinstance(arg, QuantumVariable):
                raise Exception("Tried to sample from sampling kernel taking a quantum value")

        # The structure, dtypes and shapes of the post-processed results are
        # only known once the loop body has been traced.  The first pass
        # therefore runs without accumulators: it records them and aborts.
        # The second pass runs with accumulators built from that record.
        captured = []

        @quantum_kernel
        def sampling_body_func(i, carry):
            acc, *kernel_args = carry

            values = _measure_and_decode(user_func(*kernel_args), post_processor)
            leaves, treedef = tree_flatten(values)

            if acc is None:
                _check_container(values, treedef)
                # A leaf may be a raw Python scalar rather than a JAX array --
                # a literal returned by a post_processor, say -- and then
                # carries no dtype or shape of its own.  ``asarray`` supplies
                # both, so an int literal stays integral and a complex one
                # stays complex.
                avals = [jax.ShapeDtypeStruct(jnp.shape(v), jnp.asarray(v).dtype) for v in leaves]
                captured.append((treedef, avals))
                raise _StructureCaptured()

            return (mode.update(acc, i, leaves), *kernel_args)

        try:
            jax.lax.fori_loop(0, shots, sampling_body_func, (None, *args))
        except _StructureCaptured:
            pass

        treedef, avals = captured[0]
        init_acc = mode.init(avals, static_shots)
        acc = jax.lax.fori_loop(0, shots, sampling_body_func, (init_acc, *args))[0]
        return mode.finalize(acc, treedef, shots)

    eval_function.__name__ = mode.eval_name
    return jax.jit(eval_function)


def _measure_and_decode(result, post_processor):
    """Measure and decode the quantum values in a sampling kernel's return value.

    The registers of all QuantumVariables and QuantumArrays in *result* are
    measured in ``sampling_helper_1``.  ``sampling_helper_2`` decodes the
    measurement results, puts the decoded values in place of the quantum
    values and applies *post_processor*.

    Parameters
    ----------
    result : object
        The sampling kernel's return value.
    post_processor : callable
        Applied to the rebuilt return value.

    Returns
    -------
    object
        The post-processed result of one shot.

    """
    from qrisp.core import measure

    layout, qv_tuple, classical_tuple = _split_kernel_returns(result)

    if not qv_tuple:
        return post_processor(*_assemble_kernel_returns(layout, (), classical_tuple))

    @qache
    def sampling_helper_1(*regs):
        return tuple(measure(reg) for reg in regs)

    measurement_ints = sampling_helper_1(*[qv.reg for qv in qv_tuple])

    # Classical values are passed as explicit arguments before the
    # measurement results.
    def decode(*args):
        classical_vals = args[: len(classical_tuple)]
        meas_ints = args[len(classical_tuple) :]
        decoded_q = [qv.jdecoder(meas_int) for qv, meas_int in zip(qv_tuple, meas_ints)]
        return post_processor(*_assemble_kernel_returns(layout, decoded_q, classical_vals))

    # The terminal sampling interpreter replays sampling_helper_2 to decode
    # every outcome.  jaspify rejects sampling_helper_2_mixed under terminal
    # sampling, which cannot sample classical values.
    decode.__name__ = "sampling_helper_2_mixed" if classical_tuple else "sampling_helper_2"
    return jax.jit(decode)(*classical_tuple, *measurement_ints)


def _split_kernel_returns(result):
    """Split a sampling kernel's return value into registers and classical values.

    The return value is flattened as a JAX pytree in which every
    QuantumVariable and QuantumArray is kept as a leaf, so quantum values may
    sit anywhere inside nested ``tuple``, ``list`` and ``dict`` containers.

    Parameters
    ----------
    result : object
        The kernel's return value.

    Returns
    -------
    layout : tuple
        ``(treedef, leaf_kinds)``: the pytree structure of *result* and, per
        leaf, ``"qv"`` for a QuantumVariable, the shape of a QuantumArray, or
        ``"classical"``.  :func:`_assemble_kernel_returns` uses it to rebuild
        the return value after decoding.
    qv_tuple : tuple of QuantumVariable
        The QuantumVariables to measure.  A QuantumArray contributes its
        elements in C order, one QuantumVariable each.
    classical_tuple : tuple
        The classical leaves.

    """
    from qrisp.core import QuantumArray, QuantumVariable

    leaves, treedef = tree_flatten(result, is_leaf=lambda v: isinstance(v, (QuantumVariable, QuantumArray)))

    leaf_kinds = []
    qvs = []
    classical = []
    for leaf in leaves:
        if isinstance(leaf, QuantumVariable):
            leaf_kinds.append("qv")
            qvs.append(leaf)
        elif isinstance(leaf, QuantumArray):
            leaf_kinds.append(leaf.shape)
            flat_qa = leaf.flatten()
            qvs.extend(flat_qa[i] for i in range(flat_qa.size))
        else:
            leaf_kinds.append("classical")
            classical.append(leaf)

    return (treedef, leaf_kinds), tuple(qvs), tuple(classical)


def _assemble_kernel_returns(layout, decoded_q, classical_vals):
    """Rebuild the kernel's return value from decoded and classical leaves.

    The inverse of :func:`_split_kernel_returns`: the decoded elements of a
    QuantumArray are stacked into an array of the QuantumArray's shape.

    Returns
    -------
    list
        The arguments for the post-processor: the elements of a returned
        tuple, or the return value itself otherwise.

    """
    treedef, leaf_kinds = layout
    q_iter = iter(decoded_q)
    c_iter = iter(classical_vals)

    leaves = []
    for kind in leaf_kinds:
        if kind == "qv":
            leaves.append(next(q_iter))
        elif kind == "classical":
            leaves.append(next(c_iter))
        else:
            elements = [next(q_iter) for _ in range(math.prod(kind))]
            leaves.append(jnp.stack(elements).reshape(kind))

    result = tree_unflatten(treedef, leaves)
    return list(result) if isinstance(result, tuple) else [result]


def _check_container(values, treedef):
    """Reject post-processed results in containers other than ``tuple``, ``list`` and ``dict``.

    User-defined pytree types are rejected with a clear error rather than
    being silently flattened.
    """
    if treedef.num_nodes > 1 and not isinstance(values, (tuple, list, dict)):
        raise TypeError(
            f"Unsupported return type {type(values).__name__!r}. "
            "Sampling kernels and post-processors may return tuple, list, and "
            "dict containers. Convert the return value to one of these types."
        )


def _identity(*args):
    """Return a single argument unchanged, several as a tuple."""
    if len(args) == 1:
        return args[0]
    return args


class _StructureCaptured(Exception):
    """Aborts the first trace of the sampling loop once the result structure is recorded."""
