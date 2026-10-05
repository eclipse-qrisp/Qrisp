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

"""Interpreter that turns a Jaspr into a classical computation of a resource metric."""

# Instead of performing quantum operations, the profiling interpreter lets a
# metric (a subclass of BaseMetric) update classical metric data for every
# quantum primitive, for example by incrementing a gate counter. The metrics
# themselves live in the *_metric.py modules next to this one, and the
# user-facing decorators in qrisp.jasp.evaluation_tools.profiler.

import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import pure_callback
from jax._src.core import ClosedJaxpr, Jaxpr, JaxprEqn
from jax.tree_util import tree_flatten
from jax.typing import ArrayLike

from qrisp.jasp.interpreter_tools.abstract_interpreter import (
    ContextDict,
    eval_jaxpr,
    extract_invalues,
    insert_call_outvalues,
    insert_outvalues,
)
from qrisp.jasp.interpreter_tools.call_graph_analysis import analyze_call_graph
from qrisp.jasp.interpreter_tools.interpreters.traced_control_flow_interpretation import (
    evaluate_cond_under_trace,
    evaluate_scan_under_trace,
    evaluate_while_loop_under_trace,
)
from qrisp.jasp.primitives import (
    AbstractQubit,
    AbstractQubitArray,
    QuantumPrimitive,
)

if TYPE_CHECKING:
    from qrisp.jasp.jasp_expression import Jaspr


class BaseMetric(ABC):
    """Runtime-enforced base class for profiling metrics.

    Classes inheriting from `BaseMetric` must implement handler methods
    for all quantum primitives, and define an `initial_metric` property
    that specifies the initial value of the metric before any primitives are executed.


    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    """

    def __init__(self, meas_behavior: Callable) -> None:
        """Initialize the BaseMetric."""
        self._meas_behavior: Callable = meas_behavior

    @property
    def meas_behavior(self) -> Callable:
        """Return the measurement behavior function."""
        return self._meas_behavior

    def _validate_measurement_result(self, meas_res: bool | jax.Array) -> None:
        """Validate that measurement result is a boolean.

        Parameters
        ----------
        meas_res : bool or jax.Array
            The value returned by the measurement behavior.

        Raises
        ------
        ValueError
            If ``meas_res`` is not a boolean.

        """
        if isinstance(meas_res, bool):
            return
        if hasattr(meas_res, "dtype") and meas_res.dtype == jax.numpy.bool_:
            return
        raise ValueError(f"Measurement behavior must return a boolean, got {meas_res} of type {type(meas_res)}.")

    def _measurement_body_fun(self, meas_number: ArrayLike, i: ArrayLike, acc: ArrayLike) -> ArrayLike:
        """Sample the i-th qubit of a measured QubitArray and add it to the integer result.

        Parameters
        ----------
        meas_number : ArrayLike
            The number of qubits measured before the QubitArray.
        i : ArrayLike
            The position of the qubit in the QubitArray.
        acc : ArrayLike
            The integer result accumulated from the previous qubits.

        Returns
        -------
        ArrayLike
            ``acc`` with bit ``i`` set to the sampled outcome.

        """
        meas_key = jax.random.key(meas_number + i)
        meas_res = self.meas_behavior(meas_key)
        self._validate_measurement_result(meas_res)
        return acc + jax.numpy.left_shift(1, i) * meas_res

    def _sample_measurement(self, eqn: JaxprEqn, num_measured: ArrayLike, meas_count: ArrayLike) -> ArrayLike:
        """Sample the outcome of a ``jasp.measure`` primitive.

        The k-th qubit measured in the program is sampled with the key
        ``jax.random.key(k)``. Metrics must therefore carry ``meas_count`` in their
        metric data, so that it keeps increasing across loop iterations, branches
        and subroutine calls, and all metrics see the same outcomes for the same program.

        Parameters
        ----------
        eqn : JaxprEqn
            The ``jasp.measure`` equation.
        num_measured : ArrayLike
            The number of measured qubits: the size of the QubitArray, or 1 for a single Qubit.
        meas_count : ArrayLike
            The number of qubits measured before this measurement.

        Returns
        -------
        ArrayLike
            The measurement outcome: an integer for a QubitArray, a boolean for a single Qubit.

        """
        if isinstance(eqn.invars[0].aval, AbstractQubitArray):

            def body_fun(i, acc):
                return self._measurement_body_fun(meas_count, i, acc)

            return jax.lax.fori_loop(0, num_measured, body_fun, jnp.int64(0))

        meas_res = self.meas_behavior(jax.random.key(meas_count))
        self._validate_measurement_result(meas_res)
        return meas_res

    @classmethod
    @abstractmethod
    def from_cache_key(cls, cache_key: tuple) -> "BaseMetric":
        """Reconstruct a metric instance from a hashable cache key.

        Parameters
        ----------
        cache_key : tuple
            The key returned by :meth:`cache_key`.

        Returns
        -------
        BaseMetric
            A metric with the configuration encoded in ``cache_key``.

        """

    @abstractmethod
    def initial_metric(self) -> Sequence:
        """Return the initial value of the metric before any primitives are executed.

        Returns
        -------
        Sequence
            The metric data that represents the initial QuantumState.

        """

    @abstractmethod
    def cache_key(self) -> tuple:
        """Return a hashable representation of this metric's configuration.

        Returns
        -------
        tuple
            A hashable key from which :meth:`from_cache_key` rebuilds the metric.

        """

    ##############################################################
    ### Quantum primitive handlers
    ##############################################################

    @abstractmethod
    def handle_create_qubits(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.create_qubits` primitive.

        The `create_qubits_p` primitive has the following semantics:

        - Invars: (size, QuantumState)

        - Outvars: (QubitArray, QuantumState)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_get_qubit(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.get_qubit` primitive.

        The `get_qubit_p` primitive has the following semantics:

        - Invars: (QubitArray, index)

        - Outvars: (Qubit)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_get_size(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.get_size` primitive.

        The `get_size_p` primitive has the following semantics:

        - Invars: (QubitArray)

        - Outvars: (size)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_fuse(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.fuse` primitive.

        The `fuse_p` primitive has the following semantics:

        - Invars: (Qubit | QubitArray, Qubit | QubitArray)

        - Outvars: (QubitArray)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_slice(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.slice` primitive.

        The `slice_p` primitive has the following semantics:

        - Invars: (QubitArray, start, stop)

        - Outvars: (QubitArray)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_quantum_gate(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.quantum_gate` primitive.

        The `quantum_gate_p` primitive has the following semantics:

        - Invars: (Qubit 0, Qubit 1,  ...  , Param 0, Param 1 ... , QuantumState)

        - Outvars: (QuantumState)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_measure(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.measure` primitive.

        The `measure_p` primitive has the following semantics:

        - Invars: (Qubit | QubitArray, QuantumState)

        - Outvars: (meas_result, QuantumState)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_reset(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.reset` primitive.

        The `reset_p` primitive has the following semantics:

        - Invars: (QubitArray, QuantumState)

        - Outvars: (QuantumState)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    @abstractmethod
    def handle_delete_qubits(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> Any:
        """Handle the `jasp.delete_qubits` primitive.

        The `delete_qubits_p` primitive has the following semantics:

        - Invars: (QubitArray, QuantumState)

        - Outvars: (QuantumState)

        Parameters
        ----------
        invalues : Sequence
            The metric representations of the equation's invars.
        eqn : JaxprEqn
            The equation to evaluate.
        context_dic : ContextDict
            The values of the variables in the current evaluation.

        Returns
        -------
        Any
            The metric representations of the equation's outvars: a single value
            for one outvar, a tuple for several.

        """

    def handle_parity(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> ArrayLike:
        """Handle the `jasp.parity` primitive.

        The `parity_p` primitive has the following semantics:

        - Invars: (meas_result 0, meas_result 1, ...)

        - Outvars: (parity)

        The parity is the XOR of the measurement results, flipped if the ``expectation``
        parameter is 1. It is a classical operation, so all metrics share this implementation.

        Parameters
        ----------
        invalues : Sequence
            The measurement results.
        eqn : JaxprEqn
            The ``jasp.parity`` equation.
        context_dic : ContextDict
            The values of the variables in the current evaluation (unused).

        Returns
        -------
        ArrayLike
            The parity as a boolean.

        """
        del context_dic  # part of the handler interface, not needed here
        expectation = eqn.params.get("expectation", 0)
        result = jnp.bitwise_xor(sum(invalues) % 2, expectation)

        return jnp.array(result, dtype=bool)

    def handle_create_quantum_kernel(self, *_args, **_kwargs):
        """Handle the `jasp.create_quantum_kernel` primitive.

        Parameters
        ----------
        *_args : Any
            The handler arguments (unused).
        **_kwargs : Any
            The handler keyword arguments (unused).

        Raises
        ------
        NotImplementedError
            Always, since quantum kernels cannot be profiled yet.

        """
        raise NotImplementedError("Quantum kernel creation not yet supported in profiling interpreter.")

    def get_handlers(self) -> dict[str, Callable[..., Any]]:
        """Return a mapping from primitive names to handler methods.

        Returns
        -------
        dict[str, Callable[..., Any]]
            The handler of each quantum primitive, keyed by the primitive name.

        """
        return {
            "jasp.create_qubits": self.handle_create_qubits,
            "jasp.get_qubit": self.handle_get_qubit,
            "jasp.get_size": self.handle_get_size,
            "jasp.fuse": self.handle_fuse,
            "jasp.slice": self.handle_slice,
            "jasp.quantum_gate": self.handle_quantum_gate,
            "jasp.measure": self.handle_measure,
            "jasp.reset": self.handle_reset,
            "jasp.delete_qubits": self.handle_delete_qubits,
            "jasp.create_quantum_kernel": self.handle_create_quantum_kernel,
            "jasp.parity": self.handle_parity,
        }


def normalize_slice_bounds(
    size: jax.Array | int, start: jax.Array | int, stop: jax.Array | int
) -> tuple[jax.Array, jax.Array]:
    """Normalize the bounds of a ``jasp.slice`` following Python slicing semantics.

    Negative bounds count from the end of the qubit array, bounds outside the array
    are clamped to it, and a slice whose stop lies before its start is empty.
    Only unit steps exist, since ``DynamicQubitArray`` only supports them.

    Parameters
    ----------
    size : jax.Array or int
        The size of the sliced qubit array.
    start : jax.Array or int
        The start index passed to ``jasp.slice``.
    stop : jax.Array or int
        The stop index passed to ``jasp.slice``.

    Returns
    -------
    start : jax.Array
        The normalized start index, between 0 and ``size``.
    stop : jax.Array
        The normalized stop index, between ``start`` and ``size``.

    """
    start = start + (start < 0) * size
    stop = stop + (stop < 0) * size
    start = jnp.minimum(jnp.maximum(start, 0), size)
    stop = jnp.minimum(jnp.maximum(stop, start), size)
    return start, stop


class SizeBasedMetric(BaseMetric):
    """Base class for metrics that only need the size of each qubit array.

    Such metrics never need to know which qubits an operation acts on, so they
    represent a ``QubitArray`` by its size and a ``Qubit`` by ``None``. This class
    implements the register handlers once for all of them. The constructor is the
    one of :class:`BaseMetric`.

    """

    def handle_get_qubit(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> None:
        """Handle the `jasp.get_qubit` primitive by representing the Qubit as ``None``."""
        return None

    def handle_get_size(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> ArrayLike:
        """Handle the `jasp.get_size` primitive by returning the size representing the QubitArray."""
        return invalues[0]

    def handle_fuse(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> ArrayLike:
        """Handle the `jasp.fuse` primitive.

        A QubitArray operand contributes its size, a single Qubit operand (represented
        by ``None``) contributes one qubit.
        """
        size_1, size_2 = (
            1 if isinstance(invar.aval, AbstractQubit) else value for invar, value in zip(eqn.invars, invalues)
        )
        return size_1 + size_2

    def handle_slice(self, invalues: Sequence, eqn: JaxprEqn, context_dic: ContextDict) -> jax.Array:
        """Handle the `jasp.slice` primitive by returning the size of the slice."""
        size, start, stop = invalues
        start, stop = normalize_slice_bounds(size, start, stop)
        return stop - start


# ---------------------------------------------------------------------------
# Callback helper – decides whether a sub-jaxpr should be wrapped in
# ``jax.pure_callback`` to prevent XLA from inlining it at every call site.
# ---------------------------------------------------------------------------


def _should_use_profiling_callback(
    jaxpr: Jaxpr | ClosedJaxpr, call_graph_stats: dict | None, callback_threshold: int | None
) -> bool:
    """Decide whether *jaxpr* should be called via ``jax.pure_callback``.

    A sub-jaxpr benefits from callback wrapping when it is **reused**
    (``call_count >= 2``) and large enough to matter
    (``call_count * inlined_eqn_count >= callback_threshold``).  Wrapping
    prevents XLA's ``flatten-call-graph`` pass from cloning the HLO at
    every call site.

    Parameters
    ----------
    jaxpr : Jaxpr | ClosedJaxpr
        The sub-jaxpr under consideration.
    call_graph_stats : dict | None
        Output of ``analyze_call_graph``.  ``None`` disables callbacks.
    callback_threshold : int | None
        Minimum value of ``call_count * inlined_eqn_count`` required to
        trigger callback wrapping.  Higher values mean fewer callbacks
        (faster execution, slower compilation).  ``0`` wraps every reused
        sub-jaxpr; ``None`` disables callbacks entirely.

    Returns
    -------
    bool
        Whether to call the sub-jaxpr through ``jax.pure_callback``.

    """
    if call_graph_stats is None:
        return False
    if callback_threshold is None:
        return False
    stats = call_graph_stats.get(id(jaxpr))
    if stats is None:
        return False
    return stats.call_count > 1 and stats.call_count * stats.inlined_eqn_count >= callback_threshold


# Cache for result shape specifications.  Computing result shapes requires
# a full ``make_jaxpr`` trace of the evaluator, so we cache by evaluator
# identity to avoid repeating this work on every call.
_result_shapes_cache: dict[int, Any] = {}


def _get_result_shapes(jaxpr_evaluator, invalues):
    """Compute the output shape specification for ``jax.pure_callback``.

    ``jax.pure_callback`` needs a pytree of ``ShapeDtypeStruct`` that
    describes the shape and dtype of every output *before* the callback
    executes.  We cannot derive this from the raw sub-jaxpr outvars
    because the profiling interpreter replaces quantum abstract types
    (``AbstractQuantumState``, ``AbstractQubitArray``, …) with classical
    metric-state values whose shapes depend on the metric.

    Instead, we trace ``jaxpr_evaluator`` (the profiling-transformed
    evaluator) via ``jax.make_jaxpr(return_shape=True)`` which gives us
    a pytree of ``ShapeDtypeStruct`` matching the *transformed* output
    structure — exactly what ``pure_callback`` expects.

    The result is cached per evaluator identity so the (potentially
    expensive) tracing is performed at most once.

    Parameters
    ----------
    jaxpr_evaluator : Callable
        The profiling-transformed evaluator of the sub-jaxpr.
    invalues : Sequence
        The metric values the sub-jaxpr is called with.

    Returns
    -------
    Any
        A pytree of ``ShapeDtypeStruct`` describing the outputs of ``jaxpr_evaluator``.

    """
    key = id(jaxpr_evaluator)
    if key in _result_shapes_cache:
        return _result_shapes_cache[key]

    _, result_shapes = jax.make_jaxpr(jaxpr_evaluator, return_shape=True)(*invalues)
    _result_shapes_cache[key] = result_shapes
    return result_shapes


# ---------------------------------------------------------------------------
# Compiled-profiler cache.  Keyed by ``(id(jaxpr), metric_cls, cache_key)``.
# Uses ``OrderedDict`` for LRU eviction instead of ``@lru_cache`` because
# ``call_graph_stats`` is an unhashable dict that must be threaded through.
# ---------------------------------------------------------------------------

_PROFILER_CACHE_MAX_SIZE: int = 10_000
_profiler_cache: OrderedDict[tuple, tuple] = OrderedDict()


def get_compiled_profiler(
    jaxpr: Jaxpr,
    metric_cls: type[BaseMetric],
    cache_key: tuple,
    call_graph_stats=None,
    callback_threshold=None,
) -> tuple[Callable, Callable]:
    """Get a compiled profiler for a given Jaxpr and metric configuration.

    Parameters
    ----------
    jaxpr : Jaxpr
        The Jaxpr to be profiled.
    metric_cls : type[BaseMetric]
        A subclass of BaseMetric.
    cache_key : tuple
        Hashable representation of the metric's configuration.
    call_graph_stats : dict | None, optional
        Call graph analysis results for callback optimization.
    callback_threshold : int | None, optional
        Threshold for callback wrapping decisions (passed through to
        nested sub-jaxpr evaluations).

    Returns
    -------
    tuple[Callable, Callable]
        ``(profiler, jaxpr_evaluator)`` – the compiled profiler function and the
        raw evaluator (used for computing result shapes for ``pure_callback``).

    """
    key = (id(jaxpr), metric_cls, cache_key, callback_threshold)
    if key in _profiler_cache:
        _profiler_cache.move_to_end(key)
        return _profiler_cache[key]

    metric = metric_cls.from_cache_key(cache_key)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(metric, call_graph_stats, callback_threshold)
    jaxpr_evaluator = eval_jaxpr(jaxpr, eqn_evaluator=profiling_eqn_evaluator)

    # Always JIT-compile the profiler.  When the caller decides to wrap
    # this in ``pure_callback``, JAX will invoke it with concrete numpy
    # arrays (outside the trace), so the JIT is executed eagerly and
    # the compiled HLO stays local to this callback — preventing XLA's
    # ``flatten-call-graph`` pass from duplicating it at every call site.
    profiler = jax.jit(jaxpr_evaluator)

    # We return both the JIT-compiled profiler (used for execution) and
    # the raw evaluator (needed by ``_get_result_shapes`` to trace output
    # shapes for ``pure_callback``).
    result = (profiler, jaxpr_evaluator)
    _profiler_cache[key] = result
    if len(_profiler_cache) > _PROFILER_CACHE_MAX_SIZE:
        _profiler_cache.popitem(last=False)
    return result


def make_profiling_eqn_evaluator(metric: BaseMetric, call_graph_stats=None, callback_threshold=None) -> Callable:
    """Build a profiling equation evaluator for a given metric.

    Parameters
    ----------
    metric : BaseMetric
        The metric to use for profiling. This should be an instance of a class
        that inherits from `BaseMetric`, which defines the profiling behavior for
        different quantum primitives and operations.

    call_graph_stats : dict[int, JaxprStats] | None, optional
        Call graph analysis results from ``analyze_call_graph``. When provided,
        enables callback-based compilation optimization for reused sub-jaxprs.

    callback_threshold : int | None, optional
        Minimum value of ``call_count * inlined_eqn_count`` required to trigger
        ``jax.pure_callback`` wrapping.  ``None`` (default) disables callbacks
        entirely (fastest execution).  ``0`` wraps every reused sub-jaxpr
        (fastest compilation).

    Returns
    -------
    Callable
        The profiling equation evaluator. This is a function that takes a Jaxpr equation
        and a context dictionary, and evaluates the equation according to the
        profiling logic defined by the metric.

    """
    prim_handlers = metric.get_handlers()

    def profiling_eqn_evaluator(eqn: JaxprEqn, context_dic: ContextDict) -> None | bool:
        """Evaluate ``eqn`` with the metric, returning True to let eval_jaxpr evaluate classical primitives."""
        invalues = extract_invalues(eqn, context_dic)
        prim = eqn.primitive

        if isinstance(prim, QuantumPrimitive):
            prim_handler = prim_handlers.get(prim.name, None)
            if prim_handler is None:
                raise NotImplementedError(
                    f"Don't know how to handle quantum primitive {prim.name} in profiling interpreter."
                )

            outvalues = prim_handler(invalues, eqn, context_dic)
            insert_outvalues(eqn, context_dic, outvalues)

        elif eqn.primitive.name == "cond":
            evaluate_cond_under_trace(eqn, context_dic, eqn_evaluator=profiling_eqn_evaluator)

        elif eqn.primitive.name == "while":
            evaluate_while_loop_under_trace(eqn, context_dic, eqn_evaluator=profiling_eqn_evaluator)

        elif eqn.primitive.name == "scan":
            evaluate_scan_under_trace(eqn, context_dic, eqn_evaluator=profiling_eqn_evaluator)

        elif eqn.primitive.name == "jit":
            # For qached functions, we want to make sure, the compiled function
            # contains only a single implementation per qached function.

            # Within a Jaspr, it is made sure that qached function, which is called
            # multiple times only calls the Jaspr by reference in the pjit primitive
            # this way no "copies" of the implementation appear, thus keeping
            # the size of the intermediate representation limited.

            # We want to carry on this property. For this we use the lru_cache feature
            # on the get_compiled_profiler function. This function returns a
            # jitted function, which will always return the same object if called
            # with the same object. Since identical qached function calls are
            # represented by the same jaxpr, we achieve our goal.

            # When call_graph_stats is available and the sub-jaxpr is reused
            # and large, we wrap the call in jax.pure_callback to prevent
            # XLA's flatten-call-graph pass from duplicating the HLO.
            #
            # Without callback wrapping, XLA inlines every call site's HLO
            # into the parent computation, causing superlinear growth in
            # compilation time and memory for programs with many @qache'd
            # subroutine invocations.
            #
            # The profiler returned by get_compiled_profiler is always
            # JIT-compiled.  When used via pure_callback, JAX calls it
            # with concrete arrays outside the trace, so the JIT-compiled
            # HLO is self-contained and opaque to the parent compilation.

            sub_jaxpr = eqn.params["jaxpr"]
            profiler, jaxpr_evaluator = get_compiled_profiler(
                sub_jaxpr, type(metric), metric.cache_key(), call_graph_stats, callback_threshold
            )

            if _should_use_profiling_callback(sub_jaxpr, call_graph_stats, callback_threshold):
                # Compute output shape spec for pure_callback.  We use the
                # raw (un-jitted) evaluator + make_jaxpr to trace the shapes,
                # since the profiling transform replaces quantum abstract
                # types with classical metric values whose shapes we can't
                # read off the original jaxpr outvars.
                result_shapes = _get_result_shapes(jaxpr_evaluator, invalues)
                outvalues = pure_callback(profiler, result_shapes, *invalues)
            else:
                outvalues = profiler(*invalues)

            insert_call_outvalues(eqn, context_dic, outvalues, len(eqn.outvars))

        else:
            # Classical primitive: let eval_jaxpr evaluate it
            return True

        return None

    return profiling_eqn_evaluator


def build_metric_profiler(jaspr: "Jaspr", metric: BaseMetric, callback_threshold: int | None = None) -> Callable:
    """Build a profiler function for an arbitrary metric over a Jaspr.

    Centralizes the call-graph-analysis + jit + STATIC_TYPES-argument-filtering
    boilerplate shared by get_count_ops_profiler/get_depth_profiler/
    get_num_qubits_profiler -- the only thing that varies between those three is
    which BaseMetric subclass is used and what (if any) auxiliary data (e.g. a
    profiling_dic) they return alongside the profiler.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    metric : BaseMetric
        The metric instance to profile with (e.g. a CountOpsMetric, DepthMetric,
        or NumQubitsMetric).

    callback_threshold : int | None, optional
        Minimum value of ``call_count * inlined_eqn_count`` required to trigger
        ``jax.pure_callback`` wrapping.  ``None`` (default) disables callbacks
        entirely (fastest execution).  ``0`` wraps every reused sub-jaxpr
        (fastest compilation).

    Returns
    -------
    Callable
        A profiler function taking the same *args the Jaspr itself takes.

    """
    _, call_graph_stats = analyze_call_graph(jaspr)
    profiling_eqn_evaluator = make_profiling_eqn_evaluator(metric, call_graph_stats, callback_threshold)
    jitted_evaluator = jax.jit(eval_jaxpr(jaspr, eqn_evaluator=profiling_eqn_evaluator))

    def profiler(*args):
        """Run the jitted evaluator on ``args`` and the initial metric data."""
        # Filter out types that are known to be static (https://github.com/eclipse-qrisp/Qrisp/issues/258)
        # Import here to avoid circular import issues
        from qrisp.operators import FermionicOperator, QubitOperator

        static_types = (str, QubitOperator, FermionicOperator, types.FunctionType)

        initial_metric = metric.initial_metric()

        filtered_args = [x for x in args + (initial_metric,) if type(x) not in static_types]
        return jitted_evaluator(*filtered_args)

    return profiler


def get_cached_jaspr(function: Any, args: tuple[Any, ...], meas_behavior: Any) -> "Jaspr":
    """Return the cached Jaspr trace of ``function`` called with ``args``.

    Centralizes the ``jaspr_dict`` cache-key construction (argument-type signature +
    shape signature + ``hash(meas_behavior)``) and miss-fill via ``make_jaspr``
    shared by the count_ops/depth/num_qubits profiler decorators in
    ``evaluation_tools/profiler.py``. The cache is stored as a ``jaspr_dict``
    attribute on ``function`` itself, so repeated calls with the same function and
    a matching cache key reuse the same trace.

    Parameters
    ----------
    function : Any
        The Jasp-traceable function being profiled. Typed ``Any`` rather than
        ``Callable`` because this function monkey-patches a ``jaspr_dict`` cache
        attribute directly onto it, which the ``Callable`` protocol doesn't declare.

    args : tuple
        The arguments ``function`` is being called with.

    meas_behavior : str | Callable
        The measurement behavior, included in the cache key since it can affect
        the resulting trace.

    Returns
    -------
    Jaspr
        The (possibly cached) Jaspr trace of ``function(*args)``.

    """
    # Import here to avoid circular import issues
    from qrisp.jasp import make_jaspr

    if not hasattr(function, "jaspr_dict"):
        function.jaspr_dict = {}

    signature = tuple(type(arg) for arg in args)
    shape_signature = tuple(arg.shape for arg in tree_flatten(args)[0] if hasattr(arg, "shape"))
    hash_key = (signature, shape_signature, hash(meas_behavior))

    if hash_key not in function.jaspr_dict:
        function.jaspr_dict[hash_key] = make_jaspr(function)(*args)

    return function.jaspr_dict[hash_key]
