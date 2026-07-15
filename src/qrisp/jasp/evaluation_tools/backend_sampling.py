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
    Python function ``fn(*kernel_args, shots) → result``.  Inside,
    ``to_qc`` extracts a :class:`~qrisp.circuit.QuantumCircuit`, the
    backend runs **once** (all shots), and
    :meth:`~Jaspr.extract_post_processing` decodes every shot into the
    same output the original eval function would have produced.

**Piece 2 — :func:`_make_backend_eqn_evaluator`**
    An ``eqn_evaluator`` that intercepts the two named JIT calls
    (``sampling_eval_function`` / ``expectation_value_eval_function``)
    and replaces each with a :func:`jax.pure_callback` wrapping the
    factory from piece 1.  Every other primitive falls through to
    default evaluation — no other equations need special handling.

**Piece 3 — :class:`_BackendSampler`**
    The decorator that traces the user function with
    :func:`~qrisp.jasp.make_jaspr`, validates it, wires the evaluator
    from piece 2 into the standard Jaspr evaluation loop, and JITs the
    whole pipeline with :func:`jax.jit`.

::

    ┌─ outer Jaspr ──────────────────────────────────────────────┐
    │                                                             │
    │  sampling_eval_function / expectation_value_eval_function   │  ← pure_callback
    │  ┌─ inner Jaxpr ─────────────────────────────────────────┐ │
    │  │  while i < shots:                                      │ │
    │  │    create_quantum_kernel                               │ │
    │  │    sampling_body_func(i, acc, *kernel_args, qs)        │ │
    │  │    consume_quantum_kernel                              │ │
    │  └────────────────────────────────────────────────────────┘ │
    │  result = acc (or acc / shots for EV)                      │
    └─────────────────────────────────────────────────────────────┘

Key design properties
---------------------

* **Single backend call per decorated invocation** — the circuit is
  built once, the backend runs once, and only lightweight
  post-processing varies per shot.
* **No kernel-arg extraction fragility** — kernel args are forwarded
  directly from the ``pure_callback`` invals.
* **Everything else JIT'd** — the outer Jaspr evaluation (including
  quantum-kernel bookkeeping and post-loop processing) is JIT-compiled
  by JAX.  Only the backend call lives behind the ``pure_callback``
  boundary.

.. rubric:: Annotated Jaspr example

Tracing a simple kernel that applies a Hadamard to a 3-qubit
:class:`~qrisp.QuantumFloat` and measures it with
:func:`~qrisp.jasp.sample` produces the following Jaspr::

    { lambda ; a:QuantumState. let
        b:f64[500] = pjit[
          name=sampling_eval_function          ← intercept ②
          jaxpr={ lambda ; d:f64[500] e:i64[]. let
              _:i64[] = pjit[
                name=_backend_shots_marker     ← intercept ①
                jaxpr={ lambda ; e:i64[]. let in (e,) }
              ] e
              ...
              _:i64[] _:i64[] f:f64[500] = while[
                body_jaxpr={ lambda ; g:i64[] h:f64[500] i:QuantumState. let
                    j:QuantumState = jasp.create_quantum_kernel
                    k:QuantumState l:f64[500] = pjit[
                      name=sampling_body_func  ← intercept ③
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

def _make_backend_sampling_fn(inner_jaxpr, eval_name, backend):
    """Return ``fn(*kernel_args, shots) → result`` for a given eval Jaxpr.

    Parameters
    ----------
    inner_jaxpr : ClosedJaxpr
        The Jaxpr of ``sampling_eval_function`` or
        ``expectation_value_eval_function``.
    eval_name : str
        Either ``"sampling_eval_function"`` or
        ``"expectation_value_eval_function"``.
    backend : Backend
        The backend to execute circuits on.

    Returns
    -------
    callable
        A function ``fn(*kernel_args, shots)`` that returns the same
        result as the original eval function, but obtained via the
        backend instead of the simulator.
    """
    body_jaxpr = find_named_jaxpr(inner_jaxpr.jaxpr, "sampling_body_func")
    if body_jaxpr is None:
        raise RuntimeError(
            "sampling_body_func not found inside eval function Jaxpr"
        )
    body_jaspr = Jaspr(body_jaxpr)
    is_ev = eval_name == "expectation_value_eval_function"

    def backend_sampling_fn(*invals):
        # invals = (*kernel_args, shots)
        kernel_args = list(invals[:-1])
        shots = int(invals[-1])

        # ── Build dummy args for circuit extraction ─────────────────
        # body_jaspr invars (after @quantum_kernel flattening):
        #   [i, acc, *kernel_args, qs]
        # to_qc / extract_post_processing expect the same minus the
        # trailing QuantumState.
        body_acc_aval = body_jaspr.invars[1].aval
        dummy_acc = jnp.zeros(body_acc_aval.shape, dtype=body_acc_aval.dtype)
        to_qc_args = [0, dummy_acc] + kernel_args
        *_, qc = body_jaspr.to_qc(*to_qc_args)
        raw = backend.run(qc, shots=shots)
        post_proc = body_jaspr.extract_post_processing(*to_qc_args)

        # ── Decode every shot ───────────────────────────────────────
        decoded_list = []
        for bs, cnt in raw.items():
            full_return = post_proc(bs)
            if not isinstance(full_return, tuple):
                full_return = (full_return,)
            # full_return = (acc, *kernel_args)
            acc_out = full_return[0]
            if is_ev:
                # EV accumulator: shape (1,) single, (n_returns,) multi.
                # acc started at zeros, so acc == decoded value.
                if (hasattr(acc_out, "shape")
                        and acc_out.shape == (1,)):
                    val = acc_out[0]          # single → scalar
                else:
                    val = acc_out             # multi  → vector
            else:
                # Sample accumulator: shape (shots,) or (shots, n).
                # acc[i] = decoded; acc[0] is the decoded value.
                if hasattr(acc_out, "shape") and acc_out.shape:
                    val = acc_out[0]          # scalar or row vector
                else:
                    val = acc_out             # already a scalar
            cnt = int(cnt)
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, (tuple, list)):
                decoded_list.extend([tuple(val)] * cnt)
            else:
                decoded_list.extend([float(val)] * cnt)
        np.random.shuffle(decoded_list)

        # ── Assemble final result ───────────────────────────────────
        if is_ev:
            if decoded_list and isinstance(decoded_list[0],
                                           (tuple, list)):
                total = sum(jnp.asarray(x) for x in decoded_list)
            else:
                total = sum(decoded_list)
            return jnp.array(total / shots, dtype=jnp.float64)
        else:
            if decoded_list and isinstance(decoded_list[0],
                                           (tuple, list, np.ndarray)):
                return jnp.stack([jnp.asarray(x) for x in decoded_list])
            return jnp.array(decoded_list, dtype=jnp.float64)

    return backend_sampling_fn


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

        def eqn_evaluator(eqn, context_dic, eqn_evaluator=None):
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

        # ── JIT-evaluate the Jaspr ──────────────────────────────────
        @jax.jit
        def evaluate(*flat_args):
            eval_fn = eval_jaxpr(jaspr, eqn_evaluator=eqn_evaluator)
            return eval_fn(*flat_args, _QS)

        with fast_append(3):
            flat_args = list(tree_flatten(args)[0])
            res = evaluate(*flat_args)

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
