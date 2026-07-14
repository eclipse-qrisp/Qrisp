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
:func:`~qrisp.jasp.sample` calls through a real quantum backend instead of
the Qrisp simulator.

How it works
============

:func:`~qrisp.jasp.sample` traces a sampling kernel into a Jaspr with the
following internal structure (documented in
:mod:`~qrisp.jasp.interpreter_tools.interpreters.terminal_sampling_interpreter`)::

    outer Jaspr
      └── sampling_eval_function  (pjit)
           └── fori_loop  (while, iterates over shots)
                └── sampling_body_func  (jit, inside quantum_kernel)
                     ├── user_func           ← quantum state preparation
                     ├── sampling_helper_1   ← measurement (inlined as primitives)
                     └── sampling_helper_2   ← decoding + post-processing (inlined)

:func:`backend_sampler` uses the same ``eqn_evaluator`` pattern as
:func:`~qrisp.jasp.jaspify` but intercepts at the ``sampling_body_func``
level rather than simulating the loop shot-by-shot:

1. **One-time setup** (first loop iteration):

   * Locate the ``user_func`` sub-Jaspr — this is the user's sampling kernel.
   * Extract a :class:`~qrisp.circuit.QuantumCircuit` from it via
     :meth:`~qrisp.jasp.Jaspr.to_qc`.
   * Run the circuit on the *backend* with the requested shot count.
   * Extract the classical post-processing function via
     :meth:`~qrisp.jasp.Jaspr.extract_post_processing`.
   * Apply post-processing to every backend measurement outcome and cache
     the flat list of results.

2. **Every loop iteration**:

   * Instead of executing ``user_func`` (which would run the quantum
     simulation), replace its output with the *i*-th cached result.
   * Let the existing JAX primitives (``scatter``, ``broadcast_in_dim``,
     etc.) handle accumulation into the target array naturally.

This design delegates array construction, batching, and return-value
shaping to the Jaspr's own JAX logic.  The backend only replaces the
quantum execution — everything else stays the same.

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

    This evaluator is designed to be plugged into the same Jaspr
    evaluation infrastructure that :func:`~qrisp.jasp.jaspify` uses
    (see :func:`simulate_jaspr`).  It intercepts ``sampling_body_func``
    jit calls and replaces the quantum kernel execution with a backend
    run, while leaving the loop and accumulation logic untouched.

    .. rubric:: Interception strategy

    The evaluator targets ``jit`` equations whose ``name`` parameter is
    ``"sampling_body_func"``.  These sit inside the ``fori_loop`` of
    every :func:`~qrisp.jasp.sample` call and represent one shot
    iteration::

        sampling_body_func(i, acc, *kernel_args, qs) → (acc_updated, …, qs)

    .. rubric:: Caching

    Results are cached per unique ``sampling_body_func`` Jaxpr so that
    the backend is only called once even when the sampling loop runs
    hundreds or thousands of iterations.

    Parameters
    ----------
    backend : Backend
        The backend to execute quantum circuits on.

    Returns
    -------
    callable
        An ``eqn_evaluator`` compatible with :func:`eval_jaxpr`.
        Returns ``True`` for equations it doesn't handle (delegating to
        the default evaluator) and ``False`` after processing a
        ``sampling_body_func`` equation.
    """

    # Per-evaluator result cache.
    # Maps id(body_jaxpr) → (results_list, n_returns).
    # One entry per unique sampling_body_func Jaxpr, populated on the first
    # loop iteration and reused for all subsequent iterations.
    _result_cache = {}

    def sampling_eqn_evaluator(eqn, context_dic, eqn_evaluator=None):
        # Only intercept jit calls named "sampling_body_func".
        if eqn.primitive.name != "jit":
            return True
        if eqn.params.get("name", "") != "sampling_body_func":
            return True

        # ── Extract inputs from the context ─────────────────────────────
        # sampling_body_func signature: (i, acc, *kernel_args, qs)
        invalues = extract_invalues(eqn, context_dic)
        i = int(invalues[0])          # loop counter — which shot we're on
        acc = invalues[1]             # accumulator — acc.shape[0] = shots
        qs = invalues[-1]             # QuantumState (unused by backend path)

        body_jaxpr = eqn.params["jaxpr"]   # the sampling_body_func ClosedJaxpr
        cache_key = id(body_jaxpr)

        # ── One-time setup: extract QC, run backend, cache results ──────
        # Only performed on the very first loop iteration (i == 0).
        if cache_key not in _result_cache:

            # --- locate user_func (the user's sampling kernel) ---
            user_func_jaspr = find_named_jaxpr(body_jaxpr.jaxpr, "user_func")
            if user_func_jaspr is None:
                raise RuntimeError(
                    "user_func not found in sampling_body_func"
                )

            # --- extract kernel arguments ---
            # sampling_body_func's invars are (i, acc, *kernel_args, qs).
            # Skip the first two (i, acc) and the last (qs) to get the
            # kernel's own arguments.
            kernel_args = []
            for j in range(2, len(invalues) - 1):
                val = invalues[j]
                if hasattr(val, 'item'):
                    val = val.item()   # JAX tracer → concrete Python
                kernel_args.append(val)

            # --- QuantumCircuit from user_func → run on backend ---
            # to_qc returns (*return_values, qc); we only need the circuit.
            result_tuple = user_func_jaspr.to_qc(*kernel_args)
            *_, qc = result_tuple
            shots = int(acc.shape[0])          # shot count from accumulator
            meas_result = backend.run(qc, shots=shots)

            # --- post-processing from user_func ---
            # This function takes a measurement bitstring and returns the
            # decoded, post-processed value that user_func would have returned.
            post_proc = user_func_jaspr.extract_post_processing(*kernel_args)

            # --- build flat list of post-processed results ---
            # The backend returns {bitstring: count}.  Expand each bitstring
            # *count* times, apply post-processing, and shuffle to match the
            # statistical distribution of independent shots.
            n_returns = len(user_func_jaspr.jaxpr.outvars) - 1  # excluding QS
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
            _result_cache[cache_key] = (processed, n_returns)

        # ── Look up the i-th result ─────────────────────────────────────
        processed_results, n_returns = _result_cache[cache_key]
        user_func_result = processed_results[i]

        # ── Evaluate sampling_body_func with user_func replaced ─────────
        # We evaluate the body Jaspr normally, except that the user_func
        # jit call returns our pre-computed backend result instead of
        # actually running the quantum simulation.  The downstream JAX
        # primitives (scatter, broadcast_in_dim, etc.) handle accumulation
        # into the target array exactly as they would in the original code.
        def body_eqn_evaluator(inner_eqn, inner_context):
            if (inner_eqn.primitive.name == "jit"
                    and inner_eqn.params.get("name") == "user_func"):
                # user_func returns (classical_result…, QuantumState).
                # Provide the classical result(s) and pass through the
                # QuantumState from the input unchanged.
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
                return False  # handled — don't fall through to default
            return True  # delegate everything else to default evaluation

        outvalues = eval_jaxpr(
            body_jaxpr, eqn_evaluator=body_eqn_evaluator
        )(*invalues)
        if not isinstance(outvalues, (list, tuple)):
            outvalues = [outvalues]
        insert_outvalues(eqn, context_dic, outvalues)
        return False  # handled

    return sampling_eqn_evaluator


# ===========================================================================
# Public API: backend_sampler decorator
# ===========================================================================

def backend_sampler(backend=None):
    """Decorator that routes :func:`~qrisp.jasp.sample` calls through a
    real backend instead of the Qrisp simulator.

    Uses the same ``eqn_evaluator`` infrastructure as
    :func:`~qrisp.jasp.jaspify`.  The decorated function **must** use
    :func:`~qrisp.jasp.sample` internally — a :class:`RuntimeError` is
    raised otherwise.

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

        # ── Validate: the function must use sample() internally ─────────
        if find_named_jaxpr(jaspr.jaxpr, "sampling_eval_function") is None:
            raise RuntimeError(
                "@backend_sampler requires the decorated function to use "
                "sample() internally. For single-shot simulation without "
                "sample(), use @jaspify instead."
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
            if eqn.primitive.name == "jit":
                # Recursively evaluate nested jit calls with the same
                # evaluator so that sampling_body_func is intercepted
                # at any nesting depth.
                invalues = extract_invalues(eqn, context_dic)
                outvalues = eval_jaxpr(
                    eqn.params["jaxpr"], eqn_evaluator=eqn_evaluator
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
