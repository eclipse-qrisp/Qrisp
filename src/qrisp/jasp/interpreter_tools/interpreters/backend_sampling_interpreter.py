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
          name=sampling_eval_function          ← intercept ① (outer, in evaluation_tools)
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
                      name=sampling_body_func  ← intercept ② (this module)
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

There are two interception points, ① and ② (note that these do not
line up one-to-one with the three *pieces* above).  Interception ① is
Piece 3: it swaps the eval-function pjit for a :func:`jax.pure_callback`
and lives in the evaluation-tools module.  Interception ② is Pieces 1
and 2: it replaces ``sampling_body_func`` with the pre-computed
post-processing, and lives here along with the supporting
:func:`_extract_to_qc_args`.

::

    ┌─ outer Jaspr ──────────────────────────────────────────────┐
    │                                                             │
    │  sampling_eval_function / expectation_value_eval_function   │  ← intercept ① pure_callback (evaluation_tools)
    │  ┌─ inner Jaxpr ─────────────────────────────────────────┐ │
    │  │  while i < shots:                                      │ │
    │  │    create_quantum_kernel                               │ │
    │  │    sampling_body_func(i, acc, *kernel_args, qs)        │ │  ← intercept ② (this module)
    │  │    consume_quantum_kernel                              │ │
    │  └────────────────────────────────────────────────────────┘ │
    │  result = acc (or acc / shots for EV)                      │
    └─────────────────────────────────────────────────────────────┘

Key design properties
---------------------

* **Single backend call per invocation** — the circuit is built once,
  the backend runs once with the specified number of shots,
  and only lightweight post-processing is applied per shot.
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

import jax
import jax.numpy as jnp
import numpy as np

from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
)
from qrisp.jasp.interpreter_tools.interpreters.traced_control_flow_interpretation import (
    evaluate_while_loop_under_trace,
)
from qrisp.jasp.jasp_expression.centerclass import Jaspr

# ===========================================================================
# Jaspr traversal utility
# ===========================================================================


def _find_named_jaxpr(jaxpr, target_name):
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
                result = _find_named_jaxpr(sub.jaxpr, target_name)
                if result is not None:
                    return result
        # Recurse into control-flow bodies (while, cond, scan).
        for key in ("body_jaxpr", "cond_jaxpr", "branches"):
            if key in eqn.params:
                branches = eqn.params[key]
                if not isinstance(branches, (list, tuple)):
                    branches = [branches]
                for branch in branches:
                    result = _find_named_jaxpr(branch.jaxpr, target_name)
                    if result is not None:
                        return result
    return None


# ===========================================================================
# Piece 1 — Backend sampling function factory
# ===========================================================================


class _FirstBodyCall(Exception):
    """Raised to stop evaluation after capturing the first ``sampling_body_func`` invalues."""


def _extract_to_qc_args(inner_jaxpr, body_jaspr, *invals):
    """Capture the *to_qc_args* for *body_jaspr* from the first ``sampling_body_func`` call.

    Runs *inner_jaxpr* until that call, captures its invalues, and returns
    them minus the trailing QuantumState.

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


def _sampling_body_call(body_jaxpr):
    """Return the ``sampling_body_func`` call inside a loop body, or ``None``."""
    for body_eqn in body_jaxpr.eqns:
        if body_eqn.primitive.name in ("jit", "pjit") and body_eqn.params.get("name") == "sampling_body_func":
            return body_eqn
    return None


def _cond_carry_indices(cond_jaxpr, cond_nconsts):
    """Return the indices of the carries referenced by a loop condition."""
    indices = []
    for cond_eqn in cond_jaxpr.eqns:
        for operand in cond_eqn.invars:
            for pos, invar in enumerate(cond_jaxpr.invars):
                carry = pos - cond_nconsts
                if invar is operand and carry >= 0 and carry not in indices:
                    indices.append(carry)
    return indices


def _find_sampling_loop_index(jaxpr):
    """Return the position of the loop counter in the ``sampling_body_func`` invars.

    The counter is derived structurally rather than inferred from avals.
    The sampling loop's ``cond_jaxpr`` is an ``i < shots`` comparison, which
    narrows the carries down to two: the counter and the shot bound.  Of
    those two, only the counter is threaded into ``sampling_body_func``, so
    intersecting the condition's operands with the body call's arguments
    identifies it uniquely -- without relying on the order of either the
    comparison's operands or the loop's carries.

    This deliberately avoids matching on shapes and dtypes.  JAX prepends
    captured closure values to the loop body's invars and to the body call's
    arguments, so an aval-based search can select a captured value instead of
    the counter -- silently, when that value is an integer scalar, leaving
    every iteration reading the same shot.

    Parameters
    ----------
    jaxpr : jax.extend.core.Jaxpr
        The eval-function Jaxpr (``sampling_eval_function`` or
        ``expectation_value_eval_function``) containing the sampling loop.

    Returns
    -------
    int
        Index of the loop counter within the ``sampling_body_func`` call's
        invars.

    """
    for eqn in jaxpr.eqns:
        if eqn.primitive.name != "while":
            continue

        body_jaxpr = eqn.params["body_jaxpr"].jaxpr
        call_eqn = _sampling_body_call(body_jaxpr)
        if call_eqn is None:
            continue

        body_nconsts = eqn.params["body_nconsts"]
        carries = _cond_carry_indices(eqn.params["cond_jaxpr"].jaxpr, eqn.params["cond_nconsts"])

        hits = []
        for carry in carries:
            body_pos = body_nconsts + carry
            if body_pos >= len(body_jaxpr.invars):
                continue
            body_var = body_jaxpr.invars[body_pos]
            pos = next((i for i, operand in enumerate(call_eqn.invars) if operand is body_var), None)
            if pos is not None:
                hits.append((pos, body_var.aval))

        if len(hits) != 1:
            raise RuntimeError(
                "Could not unambiguously identify the sampling loop index: "
                f"{len(hits)} of the loop condition's operands are passed to "
                "sampling_body_func (expected exactly one)."
            )

        i_pos, aval = hits[0]
        if getattr(aval, "shape", None) != () or not jnp.issubdtype(aval.dtype, jnp.integer):
            raise RuntimeError(f"Expected the sampling loop index to be an integer scalar, found {aval}.")
        return i_pos

    raise RuntimeError("Could not locate the sampling while loop inside the eval function Jaxpr")


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
    body_jaxpr = _find_named_jaxpr(inner_jaxpr.jaxpr, "sampling_body_func")
    if body_jaxpr is None:
        raise RuntimeError("sampling_body_func not found inside eval function Jaxpr")
    body_jaspr = Jaspr(body_jaxpr)

    # ── Locate the loop counter ───────────────────────────────────
    # Derived from the loop's own condition; see _find_sampling_loop_index.
    i_pos = _find_sampling_loop_index(inner_jaxpr.jaxpr)

    def backend_sampling_fn(*invals):
        # invals may include JAX-implicitly-prepended captured closure
        # variables.  Split them: the last N are the actual function
        # args (kernel_args + shots), everything before is captured.
        n_expected = len(inner_jaxpr.jaxpr.invars)
        actual_args = invals[-n_expected:]
        shots = int(actual_args[-1])

        # sample() carries a static shot count, which the decorator already
        # rejected at tracing time.  An expectation_value shot count is a tracer
        # by then -- even when the user passed a plain int -- so it arrives here
        # unvalidated, and here it is finally concrete.  Catch it before it turns
        # into an opaque failure further down (a sampling body that is never
        # reached, or an empty measurement array to index).
        if shots < 1:
            raise ValueError(
                f"backend_sampler requires a positive number of shots, got shots={shots}. "
                "A shot count of 0 selects exact probabilities, which only a simulator can provide "
                "(see terminal_sampling)."
            )

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

        # Expand & shuffle into an array of *shots* measurement rows.
        # Each distinct bitstring is parsed once and the rows are duplicated in
        # bulk with np.repeat.  The backend reports O(distinct outcomes)
        # entries while the expansion is O(shots), so building one small array
        # per shot and stacking them dominates the runtime at high shot counts.
        items = list(raw.items())
        counts = [int(count) for _, count in items]
        if sum(counts):
            unique_bits = np.array([[c == "1" for c in bitstring] for bitstring, _ in items], dtype=np.bool_)
            expanded_bits = np.repeat(unique_bits, counts, axis=0)
            np.random.shuffle(expanded_bits)
            meas_results_array = jnp.asarray(expanded_bits)
        else:
            meas_results_array = jnp.array([], dtype=jnp.bool_)

        # ── Phase 2: dynamic loop evaluation ────────────────────────
        def post_proc(meas_results, *all_non_qs_args):
            """Return ``(acc_out, *kernel_args)`` for one shot.

            *all_non_qs_args* receives the **actual** loop-carried
            values ``(i, acc, *kernel_args)``, so the extracted
            post-processing function uses the real *i* and *acc*
            rather than hard-coded dummies.
            """
            return body_jaspr.extract_post_processing(*all_non_qs_args)(meas_results)

        loop_eqn_evaluator = _body_loop_evaluator(post_proc, meas_results_array, i_pos)

        # Evaluate the inner Jaspr (the ``sampling_eval_function`` /
        # ``expectation_value_eval_function`` Jaxpr).  This runs the
        # while-loop and extracts the final result — the Jaspr itself
        # owns all accumulator typing and indexing logic.
        return jax.jit(eval_jaxpr(inner_jaxpr, eqn_evaluator=loop_eqn_evaluator))(*invals)

    return backend_sampling_fn


# ===========================================================================
# Piece 2 — Eqn evaluator for the inner Jaspr
# ===========================================================================


def _body_loop_evaluator(post_proc, meas_results, i_pos):
    """Build an ``eqn_evaluator`` for the *inner_jaxpr*.

    The *inner_jaxpr* is a ``sampling_eval_function`` or an
    ``expectation_value_eval_function``.

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
            # qs is at -1, the loop counter at i_pos
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
            # Propagate this evaluator through the loop body/condition so the
            # interceptions above still apply inside the loop.
            evaluate_while_loop_under_trace(eqn, context_dic, eqn_evaluator=eqn_evaluator)
            return False

        # ── Default (all other primitives) ──────────────────────────
        return True

    return eqn_evaluator
