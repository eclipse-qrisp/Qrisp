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

import jax.numpy as jnp
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
from qrisp.jasp.primitives import AbstractQuantumState

__all__ = ["backend_sampler", "backend_sampling_eqn_evaluator", "find_named_jaxpr"]


# ===========================================================================
# Jaspr traversal utility
# ===========================================================================

def find_named_jaxpr(jaxpr, target_name):
    """Recursively find a jit/pjit sub-jaxpr with the given name."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in ("jit", "pjit"):
            if eqn.params.get("name") == target_name:
                return eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if sub is not None:
                result = find_named_jaxpr(sub.jaxpr, target_name)
                if result is not None:
                    return result
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
    """Factory that returns an eqn_evaluator for backend-based sampling.

    This mirrors the ``terminal_sampling_evaluator`` pattern from
    :mod:`~qrisp.jasp.interpreter_tools.interpreters.terminal_sampling_interpreter`:
    it intercepts ``sampling_eval_function`` jit calls and replaces the
    loop simulation with a single backend execution.

    Parameters
    ----------
    backend : Backend
        The backend to execute quantum circuits on.

    Returns
    -------
    callable
        An ``eqn_evaluator`` compatible with :func:`eval_jaxpr`.
    """

    def sampling_eqn_evaluator(eqn, context_dic, eqn_evaluator=None):
        if eqn.primitive.name != "jit":
            return True
        if eqn.params.get("name", "") != "sampling_eval_function":
            return True

        # Step 1: extract invalues (kernel_args + shot_count)
        invalues = extract_invalues(eqn, context_dic)
        shots = int(invalues[-1])

        kernel_args = []
        for i in range(len(eqn.invars) - 1):
            val = invalues[i]
            if hasattr(val, 'item'):
                val = val.item()
            kernel_args.append(val)

        # Step 2: Get loop body Jaspr (sampling_body_func), call to_qc on it
        sampling_body_jaxpr = find_named_jaxpr(
            eqn.params["jaxpr"].jaxpr, "sampling_body_func"
        )
        if sampling_body_jaxpr is None:
            raise RuntimeError("sampling_body_func not found in sampling structure")

        loop_body_jaspr = Jaspr(sampling_body_jaxpr)
        n_non_qst = sum(
            1 for v in loop_body_jaspr.invars
            if not isinstance(v.aval, AbstractQuantumState)
        )
        # Pad with placeholder ints for loop_counter and accumulator
        to_qc_args = [0] * (n_non_qst - len(kernel_args)) + list(kernel_args)
        result_tuple = loop_body_jaspr.to_qc(*to_qc_args)
        *_, qc = result_tuple

        # Step 3: Run on backend
        meas_result = backend.run(qc, shots=shots)

        # Step 4: Get post-processing from user_func (the kernel's Jaspr).
        # user_func has a clean quantum → measure → decode structure.
        # sampling_body_func includes accumulation logic that can't be
        # handled by extract_post_processing.
        user_func_jaxpr = find_named_jaxpr(
            sampling_body_jaxpr.jaxpr, "user_func"
        )
        if user_func_jaxpr is None:
            raise RuntimeError("user_func not found in sampling structure")

        user_func_jaspr = Jaspr(user_func_jaxpr)
        post_proc = user_func_jaspr.extract_post_processing(*kernel_args)

        # Steps 5-6: Apply post-processing → build target array
        n_returns = len(eqn.outvars)
        all_results = []
        for bitstring, count in meas_result.items():
            processed = post_proc(bitstring)
            count = int(count)
            if n_returns == 1:
                val = float(processed) if hasattr(processed, 'item') else processed
                all_results.extend([val] * count)
            else:
                vals = tuple(float(v) if hasattr(v, 'item') else v for v in processed)
                all_results.extend([vals] * count)

        if n_returns == 1:
            result_array = jnp.array(all_results)
        else:
            result_array = jnp.array(all_results)
        arr_np = np.array(result_array)
        np.random.shuffle(arr_np)
        result_array = jnp.array(arr_np)

        # Step 7: Insert into context dict
        insert_outvalues(eqn, context_dic, [result_array])
        return False

    return sampling_eqn_evaluator


# ===========================================================================
# Public API: backend_sampler decorator
# ===========================================================================

def backend_sampler(backend=None):
    """Decorator that routes :func:`~qrisp.jasp.sample` calls through a
    real backend instead of the Qrisp simulator.

    Uses the same ``eqn_evaluator`` infrastructure as
    :func:`~qrisp.jasp.jaspify` but replaces ``sampling_eval_function``
    jit calls with backend execution.  The decorated function **must**
    use :func:`~qrisp.jasp.sample` internally.

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
    Basic usage with :func:`~qrisp.jasp.sample`:

    >>> from qrisp import QuantumFloat, h, measure
    >>> from qrisp.jasp import sample
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
    if backend is None:
        return lambda func: _BackendSampler(func, backend=None)

    if callable(backend):
        return _BackendSampler(backend, backend=None)

    def decorator(func):
        return _BackendSampler(func, backend=backend)

    return decorator


class _BackendSampler:
    """Callable decorator that intercepts sampling via a custom eqn_evaluator."""

    def __init__(self, func, backend=None):
        self.func = func
        self.backend = backend

    def __call__(self, *args, **kwargs):
        backend = self.backend
        if backend is None:
            from qrisp.default_backend import QrispSimulatorBackend
            backend = QrispSimulatorBackend()

        # Lightweight placeholder — satisfies the Jaspr evaluator's requirement
        # that a QuantumState flows through the context dict. Never actually
        # simulated since our evaluator replaces quantum execution with backend
        # calls.
        class _DummyQS:
            pass

        # Trace the function to get the Jaspr
        jaspr, out_tree = make_jaspr(self.func, return_shape=True)(*args, **kwargs)

        # Validate that the function uses sample() internally
        if find_named_jaxpr(jaspr.jaxpr, "sampling_eval_function") is None:
            raise RuntimeError(
                "@backend_sampler requires the decorated function to use "
                "sample() internally. For single-shot simulation without "
                "sample(), use @jaspify instead."
            )

        backend_evaluator = backend_sampling_eqn_evaluator(backend)

        sim_args = list(tree_flatten(args)[0]) + [_DummyQS()]

        def eqn_evaluator(eqn, context_dic):
            handled = backend_evaluator(eqn, context_dic, eqn_evaluator=eqn_evaluator)
            if handled is False:
                return False

            # Default behavior (mirrors simulate_jaspr in jaspification.py)
            if eqn.primitive.name == "jit":
                invalues = extract_invalues(eqn, context_dic)
                outvalues = eval_jaxpr(
                    eqn.params["jaxpr"], eqn_evaluator=eqn_evaluator
                )(*invalues)
                if not isinstance(outvalues, (list, tuple)):
                    outvalues = [outvalues]
                insert_outvalues(eqn, context_dic, outvalues)
                return False
            elif eqn.primitive.name == "jasp.create_quantum_kernel":
                insert_outvalues(eqn, context_dic, _DummyQS())
                return False
            elif eqn.primitive.name == "jasp.consume_quantum_kernel":
                return False
            else:
                return True

        with fast_append(3):
            res = eval_jaxpr(jaspr, eqn_evaluator=eqn_evaluator)(*sim_args)

        if len(jaspr.jaxpr.outvars) == 2:
            res = res[0]
        else:
            res = res[:-1]

        if isinstance(res, tuple):
            res = tree_unflatten(out_tree, res)
        elif res is not None:
            res = tree_unflatten(out_tree, [res])

        if len(recursive_qv_search(res)):
            raise Exception(
                "Tried to backend_sample a function returning a QuantumVariable"
            )
        return res
