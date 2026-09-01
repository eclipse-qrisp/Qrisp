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

r"""Backend-based sampling decorator for Jasp.

This module provides :func:`backend_sampler` — a decorator that routes
:func:`~qrisp.jasp.sample` and :func:`~qrisp.jasp.expectation_value`
calls through a real quantum backend instead of the Jaspify simulator.

The actual Jaspr interpreters (``_extract_to_qc_args``,
``_body_loop_evaluator``, ``_make_backend_sampling_fn``) live in
:mod:`~qrisp.jasp.interpreter_tools.interpreters.backend_sampling_interpreter`.
This module only contains the outer decorator and the
``pure_callback`` interception layer.

Architecture
============

:func:`backend_sampler` is built from two pieces living in this module
(a third — the Jaspr interpreter — lives in
:mod:`~qrisp.jasp.interpreter_tools.interpreters.backend_sampling_interpreter`):

**Piece 1 — :func:`_make_backend_eqn_evaluator`**
    Intercepts ``sampling_eval_function`` / ``expectation_value_eval_function``
    pjit calls in the outer Jaspr and replaces each with a
    :func:`jax.pure_callback` wrapping the backend-sampling factory
    from the interpreter module.

**Piece 2 — :func:`backend_sampler` / :func:`_make_backend_sampler_wrapper`**
    The decorator that traces the user function with
    :func:`~jax.make_jaxpr`, wires piece 1 into the standard
    Jaspr evaluation loop, and evaluates the Jaspr in pure Python
    (the ``pure_callback`` provides the JIT boundary).

.. rubric:: Usage

.. code-block:: python

    from qrisp import QuantumFloat, h, measure
    from qrisp.jasp import sample, backend_sampler
    from qrisp.default_backend import QrispSimulatorBackend

    backend = QrispSimulatorBackend()

    @backend_sampler(backend=backend)
    def main(k):
        def kernel(k):
            qf = QuantumFloat(4)
            h(qf[0])
            return measure(qf)
        return sample(kernel, shots=100)(k)

    result = main(1)  # JAX array, shape (100,), routed through backend
"""

from jax import ShapeDtypeStruct, jit, pure_callback
from jax.tree_util import tree_flatten

from qrisp.circuit import fast_append
from qrisp.jasp import make_jaxpr
from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    eval_jaxpr,
    extract_invalues,
    insert_call_outvalues,
    insert_outvalues,
)
from qrisp.jasp.interpreter_tools.interpreters.backend_sampling_interpreter import (
    _make_backend_sampling_fn,
)
from qrisp.jasp.interpreter_tools.interpreters.traced_control_flow_interpretation import (
    evaluate_cond_under_trace,
    evaluate_scan_under_trace,
    evaluate_while_loop_under_trace,
)

__all__ = ["backend_sampler"]


# ===========================================================================
# Eqn evaluator that intercepts eval functions with pure_callback
# ===========================================================================


def _make_backend_eqn_evaluator(backend):
    """Return an ``eqn_evaluator`` that swaps the eval functions for ``pure_callback`` calls.

    Intercepts ``sampling_eval_function`` and
    ``expectation_value_eval_function`` pjit calls and wraps each in
    :func:`jax.pure_callback`.  Every other primitive falls through to
    default evaluation.
    """

    def eqn_evaluator(eqn, context_dic, eqn_evaluator=None):
        name = eqn.params.get("name", "")
        prim = eqn.primitive.name

        # ``expectation_value(..., return_dict=True)`` renames its eval
        # function to mark itself for the terminal-sampling interpreter,
        # which returns a dict of outcomes.  There is no equivalent here:
        # results leave this decorator through a jitted
        # :func:`jax.pure_callback`, which has to declare a static output
        # shape and so cannot return a dict.  Reject it rather than fall
        # through -- untouched, the quantum state reaches the jit boundary
        # and XLA fails with an unintelligible aval error.
        if prim in ("jit", "pjit") and name == "dict_sampling_eval_function":
            raise NotImplementedError(
                "backend_sampler does not support "
                "expectation_value(..., return_dict=True): a dict of outcomes "
                "cannot be returned through the jitted pure_callback this "
                "decorator relies on. Use return_dict=False to obtain the "
                "expectation value, sample() to obtain the individual shots, "
                "or terminal_sampling() for the dict form."
            )

        if prim in ("jit", "pjit") and name in (
            "sampling_eval_function",
            "expectation_value_eval_function",
        ):
            invalues = extract_invalues(eqn, context_dic)
            inner_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")

            fn = _make_backend_sampling_fn(inner_jaxpr, name, backend)
            result_shapes = ShapeDtypeStruct(
                eqn.outvars[0].aval.shape,
                eqn.outvars[0].aval.dtype,
            )
            outvals = pure_callback(fn, result_shapes, *invalues)
            insert_outvalues(eqn, context_dic, [outvals])
            return False

        # Everything else: default evaluation.
        return True

    return eqn_evaluator


# ===========================================================================
# Decorator
# ===========================================================================


def backend_sampler(backend):
    r"""Route :func:`~qrisp.jasp.sample` and :func:`~qrisp.jasp.expectation_value` to a backend.

    Calls to these functions are executed on a real backend instead of the
    Jaspify simulator.

    .. warning::

        The decorated function **must** use :func:`~qrisp.jasp.sample`
        or :func:`~qrisp.jasp.expectation_value` to trigger quantum
        execution.  Direct quantum operations (gates, measurements)
        without a surrounding sample/EV call will raise a
        :class:`RuntimeError` pointing you to
        :func:`~qrisp.jasp.jaspify`.

    .. warning::

        Sampling kernels that rely on **real-time feedback** (e.g.
        mid-circuit measurements whose outcomes condition subsequent
        gates) are **not supported**.  ``backend_sampler`` extracts
        and flattens the quantum circuit into a single static circuit
        before execution, so any classical control flow that depends
        on measurement results inside the kernel cannot be captured.
        Use :func:`~qrisp.jasp.jaspify` for such workloads.

    .. note::

        Only the quantum circuit is executed on the backend.
        All **orchestration logic** (the code
        in the decorated function that calls :func:`~qrisp.jasp.sample`
        and :func:`~qrisp.jasp.expectation_value`, passes arguments,
        and combines results) is traced into a Jaspr and compiled via
        :func:`jax.jit`.  This means the non-coherence wrapping logic
        runs at JAX speed, even when orchestrating many sampling calls.

    Parameters
    ----------
    backend : :ref:`BackendInterface`
        The backend to execute on.  See the :ref:`Backend Interface
        <BackendInterface>` documentation for available backends.

    Returns
    -------
    callable
        A decorator wrapping a Jasp-compatible function.

    Raises
    ------
    RuntimeError
        If the decorated function contains quantum operations without
        a surrounding ``sample()`` or ``expectation_value()`` call.
        Use :func:`~qrisp.jasp.jaspify` for single-shot simulation.
    RuntimeError
        If a sampling kernel contains **real-time feedback**
        (mid-circuit measurements whose outcomes — after classical
        post-processing — control subsequent quantum gates).  The
        kernel's quantum circuit must be fully static so it can be
        extracted and executed once.  Use
        :func:`~qrisp.jasp.jaspify` for such workloads.

    Examples
    --------
    Basic sampling through a backend:

    .. code-block:: python

        from qrisp import QuantumFloat, h, measure
        from qrisp.jasp import sample, expectation_value, backend_sampler
        from qrisp.interface import QrispSimulatorBackend

        backend = QrispSimulatorBackend()

        @backend_sampler(backend=backend)
        def main(k):
            def kernel(k):
                qf = QuantumFloat(4)
                h(qf[0])
                return measure(qf)
            return sample(kernel, shots=100)(k)

        result = main(1)
        # result is a JAX array of shape (100,) with backend results

    Using a different backend -- any :class:`~qrisp.interface.Backend` works,
    for instance Qiskit's ``AerSimulator``:

    .. code-block:: python

        from qiskit_aer import AerSimulator
        from qrisp.interface import QiskitBackend

        backend = QiskitBackend(backend=AerSimulator())

        @backend_sampler(backend=backend)
        def main():
            def kernel():
                qv = QuantumFloat(3)
                h(qv)
                return measure(qv)
            return sample(kernel, shots=200)()

        result = main()

    Using :func:`~qrisp.jasp.expectation_value`:

    .. code-block:: python

        @backend_sampler(backend=backend)
        def main():
            def kernel():
                qf = QuantumFloat(4)
                h(qf[0])
                h(qf[1])
                return measure(qf)
            return expectation_value(kernel, shots=500)()

        ev = main()  # scalar or vector JAX array

    Multiple sample / expectation_value calls in the same function:

    .. code-block:: python

        @backend_sampler(backend=backend)
        def main():
            def kernel_a():
                qf = QuantumFloat(3)
                h(qf[0])
                return measure(qf)

            def kernel_b():
                qf = QuantumFloat(3)
                h(qf[1])
                return measure(qf)

            samples_a = sample(kernel_a, shots=100)()
            samples_b = sample(kernel_b, shots=50)()
            return samples_a, samples_b

        a, b = main()
        # Each call is independently routed through the backend.

    """
    return lambda func: _make_backend_sampler_wrapper(func, backend)


# ===========================================================================
# Control-flow handlers — propagate the custom evaluator downwards
# ===========================================================================
#
# ``while``/``cond``/``scan`` are delegated to the shared *under_trace*
# helpers, which re-interpret the sub-Jaxpr(s) with *eqn_evaluator* and replay
# them as real traced JAX primitives.  ``jit``/``pjit`` has no shared helper,
# so it is handled here (as in post_processing_interpreter.py).  Handlers are
# called for their side effect on *context_dic*; the dispatch below reports the
# equation as handled.


def _handle_jit(eqn, context_dic, eqn_evaluator):
    """Re-evaluate a ``jit``/``pjit`` call with *eqn_evaluator*."""
    closed_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
    if closed_jaxpr is None:
        return

    invalues = extract_invalues(eqn, context_dic)
    inner_eval = eval_jaxpr(closed_jaxpr, eqn_evaluator=eqn_evaluator)
    outvals = inner_eval(*(invalues + list(closed_jaxpr.consts)))
    insert_call_outvalues(eqn, context_dic, outvals, len(closed_jaxpr.jaxpr.outvars))


_CONTROL_FLOW_HANDLERS = {
    "jit": _handle_jit,
    "pjit": _handle_jit,
    "while": evaluate_while_loop_under_trace,
    "cond": evaluate_cond_under_trace,
    "scan": evaluate_scan_under_trace,
}


def _make_backend_sampler_wrapper(func, backend):
    """Return a callable that wraps *func* with backend-sampling."""

    def wrapper(*args, **kwargs):
        # ── Trace the decorated function ────────────────────────────
        # Use make_jaxpr (not make_jaspr) — we do NOT want a quantum
        # tracing context for the outer orchestration function.
        try:
            jaspr, out_shape = make_jaxpr(func, return_shape=True)(*args, **kwargs)
        except Exception as e:
            if "quantum tracing context" in str(e):
                raise RuntimeError(
                    "Encountered a quantum operation in "
                    "@backend_sampler without a surrounding "
                    "sample() or expectation_value() call. "
                    "Use @jaspify for single-shot simulation."
                ) from e
            raise

        # ── Build evaluators ────────────────────────────────────────
        be_evaluator = _make_backend_eqn_evaluator(backend)

        # Use a factory to avoid parameter-name shadowing:
        # the inner function captures itself via closure, so nested
        # eval_jaxpr calls always receive the correct evaluator.
        def make_outer_evaluator():
            def eqn_evaluator(eqn, context_dic):
                # Let the backend evaluator try first.
                if be_evaluator(eqn, context_dic, eqn_evaluator) is False:
                    return False

                # -- Propagate the custom evaluator downwards through
                # control-flow and compilation primitives.  Each handler
                # recursively calls eval_jaxpr with *eqn_evaluator* (our
                # custom evaluator), so that sample() / expectation_value()
                # calls nested inside jit, while, cond, or scan are
                # intercepted and replaced with pure_callback.
                handler = _CONTROL_FLOW_HANDLERS.get(eqn.primitive.name)
                if handler is None:
                    return True
                handler(eqn, context_dic, eqn_evaluator)
                return False

            return eqn_evaluator

        eqn_evaluator = make_outer_evaluator()

        # ── Evaluate the Jaspr ──────────────────────────────────────
        # The outer evaluator propagates through jit/pjit/while/cond/
        # scan via the handlers above, replacing sample()/EV calls with
        # pure_callback.  The resulting computation graph contains only
        # classical JAX ops and pure_callback — safe for jit.
        with fast_append(3):
            flat_args = list(tree_flatten(args)[0])
            eval_fn = eval_jaxpr(jaspr, eqn_evaluator=eqn_evaluator)
            res = jit(eval_fn)(*flat_args)

        return res

    wrapper.__name__ = getattr(func, "__name__", "backend_sampler_wrapper")
    return wrapper
