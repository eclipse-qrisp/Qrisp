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

r"""Backend-based sampling for Jasp.

This module provides :func:`backend_sampler` — a decorator that routes
:func:`~qrisp.jasp.sample` and :func:`~qrisp.jasp.expectation_value`
calls through a real quantum backend instead of the Jaspify simulator.

Architecture
============

:func:`backend_sampler` is built from three pieces:

**Piece 1 — :func:`_make_backend_sampling_fn`**
    A factory that receives a ``sampling_eval_function`` (or
    ``expectation_value_eval_function``) Jaxpr and returns a plain
    Python function ``fn(*kernel_args, shots) → result``.  It operates
    in two phases:

    **Phase 1 (static)** — extracts the quantum circuit from the
    loop-body Jaspr via :meth:`~Jaspr.to_qc`, executes it **once** on
    the backend, and expands the result into a shuffled array of
    bitstrings (one per shot).

    **Phase 2 (dynamic)** — replays the JAX while-loop inside the eval
    Jaspr via :func:`~eval_jaxpr` with a custom evaluator
    (:func:`_body_loop_evaluator`).  The evaluator intercepts the
    ``sampling_body_func`` pjit and replaces it with a call to
    :meth:`~Jaspr.extract_post_processing` using the **actual**
    loop-carried *i*, *acc*, and kernel-arg values.  This means all
    accumulator typing, indexing, and update logic comes from the
    Jaspr itself — no manual type inspection, EV/sample branching, or
    shape-dependent decoding happens in this module.

**Piece 2 — :func:`_make_backend_eqn_evaluator`**
    An ``eqn_evaluator`` that intercepts the two named JIT calls
    (``sampling_eval_function`` / ``expectation_value_eval_function``)
    in the outer Jaspr and replaces each with a
    :func:`jax.pure_callback` wrapping the factory from piece 1.
    Every other primitive falls through to default evaluation.

**Piece 3 — :class:`_BackendSampler`**
    The decorator that traces the user function with
    :func:`~qrisp.jasp.make_jaspr`, validates it, wires the evaluator
    from piece 2 into the standard Jaspr evaluation loop, and
    evaluates the Jaspr in pure Python (no ``@jax.jit`` — the
    ``pure_callback`` already provides the JIT boundary).

::

    ┌─ outer Jaspr ──────────────────────────────────────────────┐
    │                                                             │
    │  sampling_eval_function / expectation_value_eval_function   │  ← pure_callback
    │  ┌─ inner Jaxpr ─────────────────────────────────────────┐ │
    │  │  while i < shots:                                      │ │
    │  │    create_quantum_kernel                               │ │
    │  │    sampling_body_func(i, acc, *kernel_args, qs)        │ │  ← intercept ③
    │  │    consume_quantum_kernel                              │ │
    │  └────────────────────────────────────────────────────────┘ │
    │  result = acc (or acc / shots for EV)                      │
    └─────────────────────────────────────────────────────────────┘

    Inside the interception ③:
      ┌─ body_jaspr ─────────────────────────────────────────────┐
      │  extract_post_processing(i, acc, *kernel_args)(bitstr)   │
      │  → skips quantum gates, replaces measurements with bits  │
      │  → evaluates classical decoding + acc update natively    │
      │  → returns (acc_out, *kernel_args)                       │
      └──────────────────────────────────────────────────────────┘

Key design properties
---------------------

* **Single backend call per decorated invocation** — the circuit is
  built once, the backend runs once, and only lightweight
  post-processing varies per shot.
* **No manual accumulator logic** — the Jaspr's own while-loop,
  accumulator typing, indexing, and update rules are reused via
  :func:`eval_jaxpr`.  The module does not inspect accumulator
  shapes, check ``is_ev`` booleans, or manually decode tuples.
* **Dynamic *i* and *acc*** — :meth:`~Jaspr.extract_post_processing`
  is called per iteration with the real loop-carried values, so
  operations like ``acc.at[i].set(decoded)`` use the correct index.
* **Minimal interception surface** — only ``sampling_body_func`` is
  intercepted; the while-loop, generic ``jit``/``pjit`` calls, and
  quantum-kernel bookkeeping are all re-evaluated with the custom
  evaluator propagating recursively.

.. rubric:: Usage

.. code-block:: python

    from qrisp import QuantumFloat, h, measure
    from qrisp.jasp import sample, expectation_value, backend_sampler
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

import numpy as np
import jax
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
from qrisp.jasp.jasp_expression.centerclass import Jaspr

__all__ = ["backend_sampler", "find_named_jaxpr"]


# ===========================================================================
# Jaspr traversal utility
# ===========================================================================

def find_named_jaxpr(jaxpr, target_name):
    """Recursively find a ``jit`` / ``pjit`` sub-Jaxpr with the given name.

    Searches through nested jit calls and control-flow bodies
    (``while``, ``cond``, ``scan``) to locate a sub-Jaxpr that was
    annotated with ``name=target_name`` during tracing.

    This is used to extract ``sampling_body_func`` and ``user_func``
    from the sampling Jaspr structure.

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        The Jaxpr to search.
    target_name : str
        The ``name`` parameter to look for on ``jit`` / ``pjit`` equations.

    Returns
    -------
    jax.extend.core.ClosedJaxpr or None
        The matching sub-Jaxpr, or ``None`` if no equation with that name
        is found.
    """
    for eqn in jaxpr.eqns:
        # Check direct jit / pjit calls.
        if eqn.primitive.name in ("jit", "pjit"):
            if eqn.params.get("name") == target_name:
                return eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            # Recurse into nested jit bodies.
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if sub is not None:
                result = find_named_jaxpr(sub.jaxpr, target_name)
                if result is not None:
                    return result
        # Recurse into control-flow bodies (while, cond, scan).
        for key in ("body_jaxpr", "cond_jaxpr", "branches"):
            if key in eqn.params:
                branches = eqn.params[key]
                if not isinstance(branches, (list, tuple)):
                    branches = [branches]
                for branch in branches:
                    result = find_named_jaxpr(branch.jaxpr, target_name)
                    if result is not None:
                        return result
    return None


# ===========================================================================
# Piece 1 — Backend sampling function factory
# ===========================================================================


class _FirstBodyCall(Exception):
    """Raised to stop evaluation after capturing the first
    ``sampling_body_func`` invalues."""


def _extract_to_qc_args(inner_jaxpr, body_jaspr, *invals):
    """Run *inner_jaxpr* until the first ``sampling_body_func`` call,
    capture its invalues, and return them (minus the trailing
    QuantumState) as the *to_qc_args* for *body_jaspr*.

    This lets JAX's own evaluator handle all captured closure
    variables, accumulator initialisation, and loop-index setup —
    no manual position reconstruction needed.
    """
    captured_invalues = []
    _QS = object()  # sentinel for QuantumState placeholders

    def extraction_evaluator(eqn, context_dic):
        prim = eqn.primitive.name
        name = eqn.params.get("name", "")

        if prim in ("jit", "pjit") and name == "sampling_body_func":
            captured_invalues.append(extract_invalues(eqn, context_dic))
            raise _FirstBodyCall()

        # Quantum-kernel bookkeeping — no-ops
        if prim == "jasp.create_quantum_kernel":
            if eqn.outvars:
                context_dic[eqn.outvars[0]] = _QS
            return False

        if prim == "jasp.consume_quantum_kernel":
            return False

        # Let default evaluation handle everything else
        # (while loops, jit, etc.)
        return True

    try:
        eval_jaxpr(inner_jaxpr, eqn_evaluator=extraction_evaluator)(
            *invals
        )
    except _FirstBodyCall:
        pass

    if not captured_invalues:
        raise RuntimeError(
            "sampling_body_func was never called during extraction"
        )

    # invalues = [*captured, i, *kernel_args, acc, qs]
    # to_qc_args = [*captured, i, *kernel_args, acc]  (drop qs)
    return captured_invalues[0][:-1]


def _make_backend_sampling_fn(inner_jaxpr, eval_name, backend):
    """Return ``fn(*kernel_args, shots) → result`` for a given eval Jaxpr.

    **Phase 1 (static)** — extract the flat quantum circuit via
    :meth:`~Jaspr.to_qc`, execute it **once** on the backend, and
    collect all measurement bitstrings.

    **Phase 2 (dynamic)** — replay the JAX while-loop inside
    *inner_jaxpr* with a custom :func:`~eval_jaxpr` evaluator that
    intercepts the ``sampling_body_func`` pjit.  The interception
    calls :meth:`~Jaspr.extract_post_processing` with the **actual**
    loop-carried *i* and *acc* values, then feeds the pre-computed
    measurement bits for that shot.  All accumulator typing, indexing,
    and update logic is handled by the Jaspr itself.
    """
    body_jaxpr = find_named_jaxpr(inner_jaxpr.jaxpr, "sampling_body_func")
    if body_jaxpr is None:
        raise RuntimeError(
            "sampling_body_func not found inside eval function Jaxpr"
        )
    body_jaspr = Jaspr(body_jaxpr)

    # ── Pre-compute invar positions ─────────────────────────────────
    # The accumulator output is always the first outvar (return value
    # of sampling_body_func).  Match its aval against the invars to
    # find the accumulator position (robust against captured vars).
    # The loop index *i* is the first int64 scalar before the acc.
    acc_out_aval = body_jaspr.outvars[0].aval
    acc_pos = None
    for idx, invar in enumerate(body_jaspr.invars):
        aval = invar.aval
        if (type(aval) is type(acc_out_aval)
                and aval.shape == acc_out_aval.shape
                and aval.dtype == acc_out_aval.dtype):
            acc_pos = idx
            break
    if acc_pos is None:
        raise RuntimeError(
            "Could not identify accumulator position in "
            "sampling_body_func invars"
        )

    i_pos = None
    for idx in range(acc_pos):
        aval = body_jaspr.invars[idx].aval
        if hasattr(aval, 'dtype') and aval.dtype == jnp.int64 and aval.shape == ():
            i_pos = idx
            break
    if i_pos is None:
        raise RuntimeError(
            "Could not identify loop-index position in "
            "sampling_body_func invars"
        )

    def backend_sampling_fn(*invals):
        # invals may include JAX-implicitly-prepended captured closure
        # variables.  Split them: the last N are the actual function
        # args (kernel_args + shots), everything before is captured.
        n_expected = len(inner_jaxpr.jaxpr.invars)
        captured = invals[:-n_expected] if len(invals) > n_expected else ()
        actual_args = invals[-n_expected:]
        kernel_args = list(actual_args[:-1])
        shots = int(actual_args[-1])

        # ── Phase 1: static analysis ────────────────────────────────
        # Extract to_qc_args by running inner_jaxpr until the first
        # sampling_body_func call.  This lets JAX's own evaluator
        # handle all captured closure variables, accumulator
        # initialisation, and loop-index setup — no manual position
        # reconstruction needed.  Only pass the actual function args
        # (the last n_expected invals); outer captured vars are
        # consumed by the outer pjit, not by inner_jaxpr.
        to_qc_args = _extract_to_qc_args(
            inner_jaxpr, body_jaspr, *actual_args
        )
        *_, qc = body_jaspr.to_qc(*to_qc_args)

        # Run the backend ONCE.
        raw = backend.run(qc, shots=shots)

        # Expand & shuffle into a list of *shots* measurement arrays.
        meas_results_list = []
        for bitstring, count in raw.items():
            bit_array = jnp.array([c == "1" for c in bitstring], dtype=jnp.bool_)
            meas_results_list.extend([bit_array] * int(count))
        np.random.shuffle(meas_results_list)
        meas_results_array = jnp.stack(meas_results_list) if meas_results_list else jnp.array([], dtype=jnp.bool_)

        # ── Phase 2: dynamic loop evaluation ────────────────────────
        def post_proc(meas_results, *all_non_qs_args):
            """Return ``(*kernel_args, acc_out)`` for one shot.

            *all_non_qs_args* receives the **actual** loop-carried
            values ``(i, *kernel_args, acc)``, so the extracted
            post-processing function uses the real *i* and *acc*
            rather than hard-coded dummies.
            """
            return body_jaspr.extract_post_processing(*all_non_qs_args)(
                meas_results
            )

        loop_eqn_evaluator = _body_loop_evaluator(
            post_proc, meas_results_array, i_pos, acc_pos
        )

        # Evaluate the inner Jaspr (the ``sampling_eval_function`` /
        # ``expectation_value_eval_function`` Jaxpr).  This runs the
        # while-loop and extracts the final result — the Jaspr itself
        # owns all accumulator typing and indexing logic.
        return eval_jaxpr(inner_jaxpr, eqn_evaluator=loop_eqn_evaluator)(
            *invals
        )

    return backend_sampling_fn


def _body_loop_evaluator(post_proc, meas_results, i_pos, acc_pos):
    """Build an ``eqn_evaluator`` for the *inner_jaxpr*
    (``sampling_eval_function`` / ``expectation_value_eval_function``).

    Intercepts:
    * ``sampling_body_func`` pjit — replaced by *post_proc* with the
      pre-computed measurement bits for the current shot.
    * ``jasp.create_quantum_kernel`` / ``jasp.consume_quantum_kernel``
      — treated as no-ops (they produce/consume sentinel objects).
    * ``while`` — re-evaluated with this evaluator propagating
      recursively, so that the loop body sees our interceptions.
    * Generic ``jit``/``pjit`` (not ``sampling_body_func``) —
      propagated recursively for the same reason.
    """

    def eqn_evaluator(eqn, context_dic):
        prim_name = eqn.primitive.name

        # ── Intercept sampling_body_func ────────────────────────────
        if prim_name in ("jit", "pjit") and eqn.params.get("name") == "sampling_body_func":
            invalues = extract_invalues(eqn, context_dic)
            # invalues = [*captured, i, *kernel_args, acc, qs]
            # acc is always at acc_pos, qs at -1, i at i_pos
            iteration = invalues[i_pos]
            qs_val = invalues[-1]

            # post_proc(meas_bits, i, *kernel_args, acc)
            # → (*kernel_args, acc_out)  [QS stripped by extract_post_processing]
            results = post_proc(
                meas_results[iteration], *invalues[:-1]
            )
            if not isinstance(results, tuple):
                results = (results,)

            # Re-attach the QuantumState so the output arity matches
            # the pjit equation's outvars.
            results_with_qs = list(results) + [qs_val]
            insert_outvalues(eqn, context_dic, results_with_qs)
            return False

        # ── Quantum-kernel bookkeeping (no-ops) ─────────────────────
        if prim_name == "jasp.create_quantum_kernel":
            # Produce a sentinel for the QuantumState output.
            if eqn.outvars:
                context_dic[eqn.outvars[0]] = object()
            return False

        if prim_name == "jasp.consume_quantum_kernel":
            # Consumed — nothing to output.
            return False
        
        if prim_name == "while":
            invalues = extract_invalues(eqn, context_dic)
            n_body_consts = eqn.params.get("body_nconsts", 0)
            n_cond_consts = eqn.params.get("cond_nconsts", 0)
            total_consts = n_body_consts + n_cond_consts

            def body_fun(loop_state):
                # loop_state = (cond_consts, body_consts, *carries)
                body_consts = loop_state[n_cond_consts : total_consts]
                carries = loop_state[total_consts:]
                res = eval_jaxpr(
                    eqn.params["body_jaxpr"], eqn_evaluator=eqn_evaluator
                )(*(list(body_consts) + list(carries)))
                if not isinstance(res, tuple):
                    res = (res,)
                return loop_state[:total_consts] + tuple(res)

            def cond_fun(loop_state):
                cond_consts = loop_state[:n_cond_consts]
                carries = loop_state[total_consts:]
                return eval_jaxpr(
                    eqn.params["cond_jaxpr"], eqn_evaluator=eqn_evaluator
                )(*(list(cond_consts) + list(carries)))

            outvalues = jax.lax.while_loop(
                cond_fun, body_fun, tuple(invalues)
            )[total_consts:]
            insert_outvalues(eqn, context_dic, outvalues)
            return False

        # ── Default (all other primitives) ──────────────────────────
        return True

    return eqn_evaluator



# ===========================================================================
# Piece 2 — Eqn evaluator that intercepts eval functions with pure_callback
# ===========================================================================

def _make_backend_eqn_evaluator(backend):
    """Return an ``eqn_evaluator`` that replaces the eval functions with
    ``pure_callback`` calls.

    Every primitive except the two eval-function JIT calls is left to
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
# Piece 3 — Decorator that traces, validates, and JIT-evaluates
# ===========================================================================

def backend_sampler(backend=None):
    """Decorator that routes :func:`~qrisp.jasp.sample` and
    :func:`~qrisp.jasp.expectation_value` calls through a real backend.

    The decorated function **must** use :func:`~qrisp.jasp.sample` or
    :func:`~qrisp.jasp.expectation_value` internally — a
    :class:`RuntimeError` is raised otherwise.

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
    if backend is None:
        return lambda func: _BackendSampler(func, backend=None)
    if callable(backend):
        return _BackendSampler(backend, backend=None)

    def decorator(func):
        return _BackendSampler(func, backend=backend)

    return decorator


class _BackendSampler:
    """Callable that replaces quantum execution with backend calls."""

    def __init__(self, func, backend=None):
        self.func = func
        self.backend = backend

    def __call__(self, *args, **kwargs):
        backend = self.backend
        if backend is None:
            from qrisp.default_backend import QrispSimulatorBackend

            backend = QrispSimulatorBackend()

        # ── Trace the decorated function ────────────────────────────
        jaspr, out_tree = make_jaspr(self.func, return_shape=True)(
            *args, **kwargs
        )

        # ── Validate ────────────────────────────────────────────────
        if (
            find_named_jaxpr(jaspr.jaxpr, "sampling_eval_function") is None
            and find_named_jaxpr(
                jaspr.jaxpr, "expectation_value_eval_function"
            )
            is None
        ):
            raise RuntimeError(
                "@backend_sampler requires the decorated function to use "
                "sample() or expectation_value() internally. "
                "For single-shot simulation, use @jaspify instead."
            )

        # ── Build evaluators ────────────────────────────────────────
        be_evaluator = _make_backend_eqn_evaluator(backend)
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
