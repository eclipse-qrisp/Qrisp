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
calls through a real quantum backend instead of the Qrisp simulator.

How it works
============

Both :func:`~qrisp.jasp.sample` and :func:`~qrisp.jasp.expectation_value`
trace a sampling kernel into a Jaspr with the same internal structure::

    outer Jaspr
      └── sampling_eval_function / expectation_value_eval_function  (pjit)
           └── fori_loop  (while, iterates over shots)
                └── sampling_body_func  (jit, inside quantum_kernel)
                     ├── user_func           ← quantum kernel
                     ├── (inline primitives) ← measurement + decoding
                     └── (inline primitives) ← accumulation

:func:`backend_sampler` uses the same ``eqn_evaluator`` pattern as
:func:`~qrisp.jasp.jaspify` but intercepts at two levels:

1. **Top level** — ``sampling_eval_function`` / ``expectation_value_eval_function``:
   extracts the shot count and kernel arguments.  For
   :func:`~qrisp.jasp.expectation_value` these live at the top level
   because ``sampling_body_func`` only receives a running-sum accumulator,
   not the full sample array.

2. **Loop body** — ``sampling_body_func``:
   on the first iteration extracts the ``user_func`` sub-Jaspr, converts
   it to a :class:`~qrisp.circuit.QuantumCircuit` via
   :meth:`~qrisp.jasp.Jaspr.to_qc`, runs the circuit on the *backend*,
   and caches the post-processed results.  Every subsequent iteration
   looks up the *i*-th cached result and replaces ``user_func``'s output
   with it — JAX handles accumulation naturally.

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

import numpy as np
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

__all__ = ["backend_sampler", "backend_sampling_eqn_evaluator", "find_named_jaxpr"]


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
# Backend sampling eqn_evaluator
# ===========================================================================

def backend_sampling_eqn_evaluator(backend):
    """Factory that returns an ``eqn_evaluator`` for backend-based sampling.

    Plugs into the same Jaspr evaluation infrastructure as
    :func:`~qrisp.jasp.jaspify` (see :func:`simulate_jaspr`).

    **Two-level interception:**

    *   **Top level** — ``sampling_eval_function`` /
        ``expectation_value_eval_function``: extracts shot count and
        kernel arguments into shared state (*ev_state*).

    *   **Loop body** — ``sampling_body_func``: on the first iteration
        extracts *user_func*, calls :meth:`~Jaspr.to_qc` and
        :meth:`~Jaspr.extract_post_processing`, runs the backend, and
        caches processed results.  Every iteration replaces *user_func*'s
        output with the *i*-th cached result.

    **Caching** — results are cached per unique *sampling_body_func*
    Jaxpr, so the backend is called once regardless of loop count.

    Parameters
    ----------
    backend : Backend
        The backend to execute quantum circuits on.

    Returns
    -------
    callable
        An ``eqn_evaluator`` compatible with :func:`eval_jaxpr`.
    """

    # Per-evaluator result cache.
    # Maps id(body_jaxpr) → (results_list, n_returns).
    _result_cache = {}

    # Shared state for the expectation_value path: the top-level
    # expectation_value_eval_function carries shot count and kernel args
    # that are not present in sampling_body_func's own invars.
    _ev_state = {"shots": None, "kernel_args": None}

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _extract_args_sampling_eval(eqn, context_dic):
        """Extract shots + kernel_args from a top-level eval function.

        Returns (shots, kernel_args, invalues).
        """
        invalues = extract_invalues(eqn, context_dic)
        shots = int(invalues[-1])
        kernel_args = []
        for j in range(len(eqn.invars) - 1):
            val = invalues[j]
            if hasattr(val, 'item'):
                val = val.item()
            kernel_args.append(val)
        return shots, kernel_args, invalues

    def _extract_args_body_func(invalues):
        """Extract shots + kernel_args from sampling_body_func (sample() path)."""
        kernel_args = []
        for j in range(2, len(invalues) - 1):
            val = invalues[j]
            if hasattr(val, 'item'):
                val = val.item()
            kernel_args.append(val)
        shots = int(invalues[1].shape[0])   # acc.shape[0] == shots
        return shots, kernel_args

    def _setup_backend_results(user_func_jaxpr, kernel_args, shots):
        """One-time: QC → backend → post-process → flat result list.

        Returns (processed_list, n_returns).
        """
        user_func_jaspr = Jaspr(user_func_jaxpr)

        # QuantumCircuit from user_func → run on backend.
        result_tuple = user_func_jaspr.to_qc(*kernel_args)
        *_, qc = result_tuple
        meas_result = backend.run(qc, shots=shots)

        # Post-processing function from user_func.
        post_proc = user_func_jaspr.extract_post_processing(*kernel_args)

        # Expand {bitstring: count} into a flat, shuffled list.
        n_returns = len(user_func_jaxpr.jaxpr.outvars) - 1  # excluding QS
        processed = []
        for bitstring, count in meas_result.items():
            val = post_proc(bitstring)
            count = int(count)
            if n_returns == 1:
                scalar = float(val) if hasattr(val, 'item') else val
                processed.extend([scalar] * count)
            else:
                tup = tuple(
                    float(v) if hasattr(v, 'item') else v for v in val
                )
                processed.extend([tup] * count)

        np.random.shuffle(processed)
        return processed, n_returns

    def _evaluate_body_replacing_user_func(
        body_jaxpr, invalues, n_returns, user_func_result
    ):
        """Evaluate sampling_body_func with user_func output replaced."""
        def body_eqn_evaluator(inner_eqn, inner_context):
            # Replace user_func output with the pre-computed backend result.
            if (inner_eqn.primitive.name in ("jit", "pjit")
                    and inner_eqn.params.get("name") == "user_func"):
                inner_invalues = extract_invalues(inner_eqn, inner_context)
                qs_in = inner_invalues[-1]
                if n_returns == 1:
                    insert_outvalues(
                        inner_eqn, inner_context, [user_func_result, qs_in]
                    )
                else:
                    insert_outvalues(
                        inner_eqn, inner_context,
                        list(user_func_result) + [qs_in]
                    )
                return False

            # Recursively evaluate nested jit/pjit — don't let them hit
            # exec_eqn which would try XLA compilation on quantum types.
            if inner_eqn.primitive.name in ("jit", "pjit"):
                inner_invalues = extract_invalues(inner_eqn, inner_context)
                sub = (inner_eqn.params.get("jaxpr")
                       or inner_eqn.params.get("call_jaxpr"))
                outvals = eval_jaxpr(
                    sub, eqn_evaluator=body_eqn_evaluator
                )(*inner_invalues)
                if not isinstance(outvals, (list, tuple)):
                    outvals = [outvals]
                insert_outvalues(inner_eqn, inner_context, outvals)
                return False

            return True  # default for all other primitives

        outvalues = eval_jaxpr(
            body_jaxpr, eqn_evaluator=body_eqn_evaluator
        )(*invalues)
        if not isinstance(outvalues, (list, tuple)):
            outvalues = [outvalues]
        return outvalues

    # -----------------------------------------------------------------
    # Main evaluator
    # -----------------------------------------------------------------

    def sampling_eqn_evaluator(eqn, context_dic, eqn_evaluator=None):
        name = eqn.params.get("name", "")

        # ── Top-level sampling / expectation_value calls ──────────────
        if eqn.primitive.name in ("jit", "pjit") and name in (
            "sampling_eval_function", "expectation_value_eval_function",
        ):
            shots, kernel_args, invalues = _extract_args_sampling_eval(
                eqn, context_dic
            )
            _ev_state["shots"] = shots
            _ev_state["kernel_args"] = kernel_args

            # Evaluate inner Jaxpr; sampling_body_func interceptor (below)
            # will use the stored _ev_state if needed.
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            outvalues = eval_jaxpr(sub, eqn_evaluator=eqn_evaluator)(*invalues)
            if not isinstance(outvalues, (list, tuple)):
                outvalues = [outvalues]
            insert_outvalues(eqn, context_dic, outvalues)
            return False

        # ── sampling_body_func ────────────────────────────────────────
        if eqn.primitive.name not in ("jit", "pjit"):
            return True
        if name != "sampling_body_func":
            return True

        invalues = extract_invalues(eqn, context_dic)
        i = int(invalues[0])
        body_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
        cache_key = id(body_jaxpr)

        # Determine shots + kernel_args depending on the path.
        # sample():        invars are (i, acc, *kernel_args, qs)
        # expectation_value: invars are (i, acc_scalar, qs) — use _ev_state
        if len(invalues) > 3:
            shots, kernel_args = _extract_args_body_func(invalues)
        else:
            shots = _ev_state["shots"]
            kernel_args = _ev_state["kernel_args"]

        # ── One-time setup (first iteration only) ─────────────────────
        if cache_key not in _result_cache:
            user_func_jaxpr = find_named_jaxpr(body_jaxpr.jaxpr, "user_func")
            if user_func_jaxpr is None:
                raise RuntimeError(
                    "user_func not found in sampling_body_func"
                )
            _result_cache[cache_key] = _setup_backend_results(
                user_func_jaxpr, kernel_args, shots
            )

        # ── Look up i-th result, evaluate body with user_func replaced ─
        processed_results, n_returns = _result_cache[cache_key]
        user_func_result = processed_results[i]

        outvalues = _evaluate_body_replacing_user_func(
            body_jaxpr, invalues, n_returns, user_func_result
        )
        insert_outvalues(eqn, context_dic, outvalues)
        return False

    return sampling_eqn_evaluator


# ===========================================================================
# Public API: backend_sampler decorator
# ===========================================================================

def backend_sampler(backend=None):
    """Decorator that routes :func:`~qrisp.jasp.sample` and
    :func:`~qrisp.jasp.expectation_value` calls through a real backend.

    Uses the same ``eqn_evaluator`` infrastructure as
    :func:`~qrisp.jasp.jaspify`.  The decorated function **must** use
    :func:`~qrisp.jasp.sample` or :func:`~qrisp.jasp.expectation_value`
    internally — a :class:`RuntimeError` is raised otherwise.

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
        A decorator wrapping a Jasp-compatible function that uses
        :func:`~qrisp.jasp.sample`.

    Raises
    ------
    RuntimeError
        If the decorated function does not use :func:`~qrisp.jasp.sample`
        internally.

    Examples
    --------
    >>> from qrisp import QuantumFloat, h, measure
    >>> from qrisp.jasp import sample, backend_sampler
    >>> from qrisp.default_backend import QrispSimulatorBackend
    >>>
    >>> backend = QrispSimulatorBackend()
    >>>
    >>> @backend_sampler(backend=backend)
    ... def main(k):
    ...     def kernel(k):
    ...         qf = QuantumFloat(4)
    ...         h(qf[0])
    ...         return measure(qf)
    ...     return sample(kernel, shots=100)(k)
    >>>
    >>> result = main(1)
    >>> result.shape
    (100,)
    """
    # @backend_sampler — no arguments, func is the decorated function.
    if backend is None:
        return lambda func: _BackendSampler(func, backend=None)

    # @backend_sampler — bare callable (the function itself).
    if callable(backend):
        return _BackendSampler(backend, backend=None)

    # @backend_sampler(backend=...) — explicit backend.
    def decorator(func):
        return _BackendSampler(func, backend=backend)

    return decorator


class _BackendSampler:
    """Callable decorator that replaces quantum execution with backend calls.

    On each invocation:

    1. Traces the decorated function with :func:`make_jaspr` to obtain a
       :class:`Jaspr`.
    2. Verifies the Jaspr contains a ``sampling_eval_function`` (i.e. the
       function uses :func:`~qrisp.jasp.sample` internally).
    3. Creates a :func:`backend_sampling_eqn_evaluator` and evaluates the
       Jaspr with the same infrastructure that :func:`~qrisp.jasp.jaspify`
       uses, except that ``sampling_body_func`` equations are intercepted
       and routed through the backend.
    """

    def __init__(self, func, backend=None):
        self.func = func
        self.backend = backend

    def __call__(self, *args, **kwargs):
        # ── Resolve backend ────────────────────────────────────────────
        backend = self.backend
        if backend is None:
            from qrisp.default_backend import QrispSimulatorBackend
            backend = QrispSimulatorBackend()

        # ── Trace the decorated function ───────────────────────────────
        # return_shape=True captures the output PyTree structure so we can
        # reconstruct nested return values (tuples, dicts, etc.).
        jaspr, out_tree = make_jaspr(
            self.func, return_shape=True
        )(*args, **kwargs)

        # ── Validate: the function must use sample() or expectation_value() ─
        if (find_named_jaxpr(jaspr.jaxpr, "sampling_eval_function") is None
                and find_named_jaxpr(jaspr.jaxpr, "expectation_value_eval_function") is None):
            raise RuntimeError(
                "@backend_sampler requires the decorated function to use "
                "sample() or expectation_value() internally. "
                "For single-shot simulation, use @jaspify instead."
            )

        # ── Build the custom evaluator ─────────────────────────────────
        backend_evaluator = backend_sampling_eqn_evaluator(backend)

        # ── Prepare Jaspr evaluation arguments ─────────────────────────
        # Every Jaspr carries a QuantumState as its last input and output.
        # We inject a _DummyQS — a lightweight placeholder that satisfies
        # the type checks but is never simulated (our evaluator replaces
        # all quantum execution with backend calls).
        class _DummyQS:
            pass

        sim_args = list(tree_flatten(args)[0]) + [_DummyQS()]

        # ── Top-level equation evaluator ───────────────────────────────
        # Mirrors the pattern from simulate_jaspr() in jaspification.py.
        # The backend_evaluator handles sampling_body_func intercepts;
        # everything else gets the same default behavior as jaspify
        # (recursive jit evaluation, quantum_kernel placeholders).
        def eqn_evaluator(eqn, context_dic):
            # Let the backend evaluator try first.
            handled = backend_evaluator(
                eqn, context_dic, eqn_evaluator=eqn_evaluator
            )
            if handled is False:
                return False  # backend evaluator processed it

            # -- default behavior (matches simulate_jaspr) --
            if eqn.primitive.name in ("jit", "pjit"):
                # Recursively evaluate nested jit calls with the same
                # evaluator so that sampling_body_func is intercepted
                # at any nesting depth.
                invalues = extract_invalues(eqn, context_dic)
                # pjit stores the sub-jaxpr in "call_jaxpr"; jit uses "jaxpr".
                sub_jaxpr = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
                outvalues = eval_jaxpr(
                    sub_jaxpr, eqn_evaluator=eqn_evaluator
                )(*invalues)
                if not isinstance(outvalues, (list, tuple)):
                    outvalues = [outvalues]
                insert_outvalues(eqn, context_dic, outvalues)
                return False
            elif eqn.primitive.name == "jasp.create_quantum_kernel":
                # Quantum kernel boundary — inject dummy state.
                insert_outvalues(eqn, context_dic, _DummyQS())
                return False
            elif eqn.primitive.name == "jasp.consume_quantum_kernel":
                # Quantum kernel boundary — nothing to do.
                return False
            else:
                return True  # use default exec_eqn for other primitives

        # ── Evaluate the Jaspr ─────────────────────────────────────────
        with fast_append(3):
            res = eval_jaxpr(jaspr, eqn_evaluator=eqn_evaluator)(*sim_args)

        # ── Unpack results ─────────────────────────────────────────────
        # The Jaspr's last output is always a QuantumState — discard it.
        if len(jaspr.jaxpr.outvars) == 2:
            res = res[0]
        else:
            res = res[:-1]

        # ── Reconstruct PyTree structure ───────────────────────────────
        # tree_unflatten restores the original return type (tuple, dict,
        # namedtuple, etc.) from the flat list of JAX arrays.
        if isinstance(res, tuple):
            res = tree_unflatten(out_tree, res)
        elif res is not None:
            res = tree_unflatten(out_tree, [res])

        # ── Safety check ───────────────────────────────────────────────
        # The decorated function must not return QuantumVariables — those
        # represent unmeasured quantum states that have no meaning after
        # backend execution.
        if len(recursive_qv_search(res)):
            raise Exception(
                "Tried to backend_sample a function returning a "
                "QuantumVariable.  Use measure() to convert to classical "
                "values before returning."
            )
        return res
