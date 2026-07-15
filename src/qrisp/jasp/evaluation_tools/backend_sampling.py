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
    :func:`~qrisp.jasp.make_jaspr`, wires piece 1 into the standard
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

import jax.numpy as jnp
from jax import ShapeDtypeStruct, pure_callback
from jax.tree_util import tree_flatten, tree_unflatten

from qrisp.circuit import fast_append
from qrisp.core import recursive_qv_search
from qrisp.jasp import make_jaspr
from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
)
from qrisp.jasp.interpreter_tools.interpreters.backend_sampling_interpreter import (
    _make_backend_sampling_fn,
    find_named_jaxpr,
)

__all__ = ["backend_sampler", "find_named_jaxpr"]


# ===========================================================================
# Eqn evaluator that intercepts eval functions with pure_callback
# ===========================================================================

def _make_backend_eqn_evaluator(backend):
    """Return an ``eqn_evaluator`` that replaces the eval functions with
    ``pure_callback`` calls.

    Intercepts ``sampling_eval_function`` and
    ``expectation_value_eval_function`` pjit calls and wraps each in
    :func:`jax.pure_callback`.  Every other primitive falls through to
    default evaluation.
    """
    def eqn_evaluator(eqn, context_dic, eqn_evaluator=None):
        name = eqn.params.get("name", "")
        prim = eqn.primitive.name

        if prim in ("jit", "pjit") and name in (
            "sampling_eval_function",
            "expectation_value_eval_function",
        ):
            invalues = extract_invalues(eqn, context_dic)
            inner_jaxpr = eqn.params.get("jaxpr") or eqn.params.get(
                "call_jaxpr"
            )

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

def backend_sampler(backend=None):
    """Decorator that routes :func:`~qrisp.jasp.sample` and
    :func:`~qrisp.jasp.expectation_value` calls through a real backend.

    The decorator can be used with or without explicit arguments::

        @backend_sampler
        def main(): ...

        @backend_sampler(backend=my_backend)
        def main(): ...

    Parameters
    ----------
    backend : Backend, optional
        The backend to execute on. If ``None``, defaults to
        :class:`~qrisp.default_backend.QrispSimulatorBackend`.

    Returns
    -------
    callable
        A decorator wrapping a Jasp-compatible function.
    """
    # Support both @backend_sampler and @backend_sampler(backend=...)
    if backend is None:
        return lambda func: _make_backend_sampler_wrapper(func, None)
    if callable(backend):
        return _make_backend_sampler_wrapper(backend, None)
    return lambda func: _make_backend_sampler_wrapper(func, backend)


def _make_backend_sampler_wrapper(func, backend):
    """Return a callable that wraps *func* with backend-sampling."""
    def wrapper(*args, **kwargs):
        _backend = backend
        if _backend is None:
            from qrisp.default_backend import QrispSimulatorBackend
            _backend = QrispSimulatorBackend()

        # ── Trace the decorated function ────────────────────────────
        jaspr, out_tree = make_jaspr(func, return_shape=True)(
            *args, **kwargs
        )

        # ── Build evaluators ────────────────────────────────────────
        be_evaluator = _make_backend_eqn_evaluator(_backend)
        _QS = jnp.array(0.0)  # sentinel for QuantumState placeholders

        # Use a factory to avoid parameter-name shadowing:
        # the inner function captures itself via closure, so nested
        # eval_jaxpr calls always receive the correct evaluator.
        def make_outer_evaluator():
            def eqn_evaluator(eqn, context_dic):
                # Let the backend evaluator try first.
                if be_evaluator(eqn, context_dic, eqn_evaluator) is False:
                    return False

                # -- default behaviour (mirrors simulate_jaspr) --
                prim = eqn.primitive.name
                if prim in ("jit", "pjit"):
                    invalues = extract_invalues(eqn, context_dic)
                    sub_jaxpr = eqn.params.get("jaxpr") or eqn.params.get(
                        "call_jaxpr"
                    )
                    outvalues = eval_jaxpr(
                        sub_jaxpr, eqn_evaluator=eqn_evaluator
                    )(*invalues)
                    if not isinstance(outvalues, (list, tuple)):
                        outvalues = [outvalues]
                    insert_outvalues(eqn, context_dic, outvalues)
                    return False
                elif prim == "jasp.create_quantum_kernel":
                    insert_outvalues(eqn, context_dic, _QS)
                    return False
                elif prim == "jasp.consume_quantum_kernel":
                    return False
                elif prim.startswith("jasp."):
                    raise RuntimeError(
                        f"Encountered quantum primitive '{prim}' in "
                        f"@backend_sampler without a surrounding "
                        f"sample() or expectation_value() call. "
                        f"Use @jaspify for single-shot simulation."
                    )
                else:
                    return True

            return eqn_evaluator

        eqn_evaluator = make_outer_evaluator()

        # ── Evaluate the Jaspr ──────────────────────────────────────
        # No @jax.jit — the Jaspr carries AbstractQuantumState values
        # that JAX cannot lower to MLIR.  The pure_callback inside
        # _make_backend_eqn_evaluator already provides the JIT
        # boundary for the eval function.
        with fast_append(3):
            flat_args = list(tree_flatten(args)[0])
            eval_fn = eval_jaxpr(jaspr, eqn_evaluator=eqn_evaluator)
            res = eval_fn(*flat_args, _QS)

        # ── Strip the trailing QuantumState output ───────────────────
        if len(jaspr.jaxpr.outvars) == 2:
            res = res[0]
        else:
            res = res[:-1]

        # ── Reconstruct PyTree ──────────────────────────────────────
        if isinstance(res, tuple):
            res = tree_unflatten(out_tree, res)
        elif res is not None:
            res = tree_unflatten(out_tree, [res])

        # ── Safety check ────────────────────────────────────────────
        if len(recursive_qv_search(res)):
            raise Exception(
                "Tried to backend_sample a function returning a "
                "QuantumVariable.  Use measure() to convert to classical "
                "values before returning."
            )
        return res

    wrapper.__name__ = getattr(func, '__name__', 'backend_sampler_wrapper')
    return wrapper
