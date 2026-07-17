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

r"""Backend-sampling Jaspr interpreter.

This module implements the custom ``eval_jaxpr`` evaluators that replace
quantum execution with pre-computed backend results inside
:func:`~qrisp.jasp.backend_sampler`.  The outer interception (replacing
``sampling_eval_function`` / ``expectation_value_eval_function`` pjit
calls with :func:`jax.pure_callback`) lives in
:mod:`~qrisp.jasp.evaluation_tools.backend_sampling`.

Architecture
============

:func:`backend_sampler` is built from three pieces.  Pieces 1 and 2
(the interpreter layer) live here; Piece 3 (the outer decorator and
``pure_callback`` interception) lives in the evaluation-tools module.

**Piece 1 — :func:`_make_backend_sampling_fn`**
    A factory that receives a ``sampling_eval_function`` (or
    ``expectation_value_eval_function``) Jaxpr and returns a plain
    Python function ``fn(*kernel_args, shots) → result``.  Operates in
    two phases:

    **Phase 1 (static)** — extracts the quantum circuit from the
    loop-body Jaspr via :meth:`~qrisp.jasp.Jaspr.to_qc`, executes it
    **once** on the backend, and expands the result into a shuffled
    array of bitstrings (one per shot).  The arguments for ``to_qc``
    are obtained by running the eval Jaspr with a mini-interpreter
    (:func:`_extract_to_qc_args`) that stops at the first
    ``sampling_body_func`` call — this lets JAX resolve all captured
    closure variables automatically.

    **Phase 2 (dynamic)** — replays the JAX while-loop inside the eval
    Jaspr via :func:`~qrisp.jasp.eval_jaxpr` with a custom evaluator
    (:func:`_body_loop_evaluator`).  The evaluator intercepts the
    ``sampling_body_func`` pjit and replaces it with a call to
    :meth:`~qrisp.jasp.Jaspr.extract_post_processing` using the
    **actual** loop-carried *i*, *acc*, and kernel-arg values.  All
    accumulator typing, indexing, and update logic comes from the
    Jaspr itself.

**Piece 2 — :func:`_body_loop_evaluator`**
    The ``eqn_evaluator`` that runs inside the eval Jaspr.  It
    intercepts ``sampling_body_func`` (replaced by
    :meth:`~qrisp.jasp.Jaspr.extract_post_processing` with pre-computed
    measurement bits), treats ``jasp.create_quantum_kernel`` /
    ``jasp.consume_quantum_kernel`` as no-ops, and propagates itself
    recursively through ``while`` loops and generic ``jit``/``pjit``
    calls.

Example Jaspr structure
=======================

Tracing a simple kernel that applies a Hadamard to a 3-qubit
:class:`~qrisp.QuantumFloat` and measures it with
:func:`~qrisp.jasp.sample` produces the following Jaspr::

    { lambda ; a:QuantumState. let
        b:f64[500] = pjit[
          name=sampling_eval_function          ← intercept ② (outer, in evaluation_tools)
          jaxpr={ lambda ; d:f64[500] e:i64[]. let
              _:i64[] = pjit[
                name=_backend_shots_marker
                jaxpr={ lambda ; e:i64[]. let in (e,) }
              ] e
              ...
              _:i64[] _:i64[] f:f64[500] = while[
                body_jaxpr={ lambda ; g:i64[] h:f64[500] i:QuantumState. let
                    j:QuantumState = jasp.create_quantum_kernel
                    k:QuantumState l:f64[500] = pjit[
                      name=sampling_body_func  ← intercept ③ (this module)
                      jaxpr={ ...
                        pjit[name=user_func] ...
                        pjit[name=sampling_helper_1] ...
                        pjit[name=sampling_helper_2] ...
                      }
                    ] g h j
                    m:QuantumState = jasp.consume_quantum_kernel k
                  in (l, m) }
                ...
              ] ...
            in (f,) }
        ] ...
      in (b,) }

The three interception points ①②③ correspond to the architecture
described above.  Interception ② lives in the evaluation-tools module;
interception ③ (and the supporting ``_extract_to_qc_args``) lives here.

::

    ┌─ outer Jaspr ──────────────────────────────────────────────┐
    │                                                             │
    │  sampling_eval_function / expectation_value_eval_function   │  ← pure_callback (evaluation_tools)
    │  ┌─ inner Jaxpr ─────────────────────────────────────────┐ │
    │  │  while i < shots:                                      │ │
    │  │    create_quantum_kernel                               │ │
    │  │    sampling_body_func(i, acc, *kernel_args, qs)        │ │  ← intercept ③ (this module)
    │  │    consume_quantum_kernel                              │ │
    │  └────────────────────────────────────────────────────────┘ │
    │  result = acc (or acc / shots for EV)                      │
    └─────────────────────────────────────────────────────────────┘

Key design properties
---------------------

* **Single backend call per invocation** — the circuit is built once,
  the backend runs once, and only lightweight post-processing varies
  per shot.
* **No manual accumulator logic** — the Jaspr's own while-loop,
  accumulator typing, indexing, and update rules are reused via
  :func:`eval_jaxpr`.  The module never inspects accumulator shapes
  or manually decodes tuples.
* **Dynamic *i* and *acc*** — :meth:`~qrisp.jasp.Jaspr.extract_post_processing`
  is called per iteration with the real loop-carried values.
* **Minimal interception surface** — only ``sampling_body_func`` is
  intercepted; the while-loop, ``jit``/``pjit`` calls, and
  quantum-kernel bookkeeping all propagate with the custom evaluator.
"""

import numpy as np
import jax
import jax.numpy as jnp

from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
)
from qrisp.jasp.jasp_expression.centerclass import Jaspr


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
        eval_jaxpr(inner_jaxpr, eqn_evaluator=extraction_evaluator)(*invals)
    except _FirstBodyCall:
        pass

    if not captured_invalues:
        raise RuntimeError("sampling_body_func was never called during extraction")

    # invalues = [*captured, i, acc, *kernel_args, qs]
    # to_qc_args = [*captured, i, acc, *kernel_args]  (drop qs)
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
        raise RuntimeError("sampling_body_func not found inside eval function Jaxpr")
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
        if type(aval) is type(acc_out_aval) and aval.shape == acc_out_aval.shape and aval.dtype == acc_out_aval.dtype:
            acc_pos = idx
            break
    if acc_pos is None:
        raise RuntimeError("Could not identify accumulator position in sampling_body_func invars")

    i_pos = None
    for idx in range(acc_pos):
        aval = body_jaspr.invars[idx].aval
        if hasattr(aval, "dtype") and aval.dtype == jnp.int64 and aval.shape == ():
            i_pos = idx
            break
    if i_pos is None:
        raise RuntimeError("Could not identify loop-index position in sampling_body_func invars")

    def backend_sampling_fn(*invals):
        # invals may include JAX-implicitly-prepended captured closure
        # variables.  Split them: the last N are the actual function
        # args (kernel_args + shots), everything before is captured.
        n_expected = len(inner_jaxpr.jaxpr.invars)
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
        to_qc_args = _extract_to_qc_args(inner_jaxpr, body_jaspr, *actual_args)
        try:
            *_, qc = body_jaspr.to_qc(*to_qc_args)
        except Exception as e:
            if "real-time feedback" in str(e):
                raise RuntimeError(
                    "Failed to extract a static QuantumCircuit from "
                    "the sampling kernel because it contains "
                    "real-time feedback (mid-circuit measurements "
                    "whose outcomes control subsequent quantum "
                    "gates).  ``backend_sampler`` requires the "
                    "kernel's quantum circuit to be fully static — "
                    "use :func:`~qrisp.jasp.jaspify` for workloads "
                    "with real-time feedback."
                ) from e
            raise

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
            """Return ``(acc_out, *kernel_args)`` for one shot.

            *all_non_qs_args* receives the **actual** loop-carried
            values ``(i, acc, *kernel_args)``, so the extracted
            post-processing function uses the real *i* and *acc*
            rather than hard-coded dummies.
            """
            return body_jaspr.extract_post_processing(*all_non_qs_args)(meas_results)

        loop_eqn_evaluator = _body_loop_evaluator(post_proc, meas_results_array, i_pos, acc_pos)

        # Evaluate the inner Jaspr (the ``sampling_eval_function`` /
        # ``expectation_value_eval_function`` Jaxpr).  This runs the
        # while-loop and extracts the final result — the Jaspr itself
        # owns all accumulator typing and indexing logic.
        result = jax.jit(eval_jaxpr(inner_jaxpr, eqn_evaluator=loop_eqn_evaluator))(*invals)

        # The result may now be a nested pytree (tuple/list of arrays)
        # from the post-loop restructuring in sampling_eval_function.
        # Flatten to a plain tuple so that pure_callback (which expects
        # a flat sequence when given multiple result shapes) can consume it.
        # (list is registered as a pytree node in sampling.py, so
        # tree_leaves recurses into lists as well as tuples.)
        if isinstance(result, (tuple, list)):
            flat = jax.tree_util.tree_leaves(result)
            if len(flat) == 1:
                return flat[0]
            return tuple(flat)
        return result

    return backend_sampling_fn


# ===========================================================================
# Piece 2 — Eqn evaluator for the inner Jaspr
# ===========================================================================


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
            # invalues = [*captured, i, acc, *kernel_args, qs]
            # acc is at acc_pos, qs at -1, i at i_pos
            iteration = invalues[i_pos]
            qs_val = invalues[-1]

            # post_proc(meas_bits, i, acc, *kernel_args)
            # → (acc_out, *kernel_args)  [QS stripped by extract_post_processing]
            results = post_proc(meas_results[iteration], *invalues[:-1])
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
                body_consts = loop_state[n_cond_consts:total_consts]
                carries = loop_state[total_consts:]
                res = eval_jaxpr(eqn.params["body_jaxpr"], eqn_evaluator=eqn_evaluator)(
                    *(list(body_consts) + list(carries))
                )
                if not isinstance(res, tuple):
                    res = (res,)
                return loop_state[:total_consts] + tuple(res)

            def cond_fun(loop_state):
                cond_consts = loop_state[:n_cond_consts]
                carries = loop_state[total_consts:]
                return eval_jaxpr(eqn.params["cond_jaxpr"], eqn_evaluator=eqn_evaluator)(
                    *(list(cond_consts) + list(carries))
                )

            outvalues = jax.lax.while_loop(cond_fun, body_fun, tuple(invalues))[total_consts:]
            insert_outvalues(eqn, context_dic, outvalues)
            return False

        # ── Default (all other primitives) ──────────────────────────
        return True

    return eqn_evaluator
