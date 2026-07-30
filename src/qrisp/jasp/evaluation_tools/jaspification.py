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

from collections.abc import Callable
from typing import Any, Literal

import jax
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn
from jax.tree_util import tree_flatten, tree_unflatten
from jaxlib.mlir import ir

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.circuit import fast_append
from qrisp.core import recursive_qv_search
from qrisp.jasp.evaluation_tools.buffered_quantum_state import BufferedQuantumState
from qrisp.jasp.interpreter_tools import (
    eval_jaxpr,
    extract_invalues,
    insert_outvalues,
    terminal_sampling_evaluator,
)
from qrisp.jasp.interpreter_tools.abstract_interpreter import ContextDict
from qrisp.jasp.jasp_expression.centerclass import Jaspr, make_jaspr
from qrisp.jasp.primitives import (
    AbstractQuantumState,
    AbstractQubit,
    AbstractQubitArray,
)


def jaspify(func: Callable | bool | None = None, terminal_sampling: bool = False) -> Callable:
    """This simulator is the established Qrisp simulator linked to the Jasp infrastructure.
    Among a variety of simulation tricks, the simulator can leverage state sparsity,
    allowing simulations with up to hundreds of qubits!

    To be called as a decorator of a Jasp-traceable function.

    .. note::

        If you are developing a hybrid algorithm like QAOA or VQE that relies
        heavily on sampling, please activate the ``terminal_sampling`` feature.

    Parameters
    ----------
    func : callable
        The function to simulate.
    terminal_sampling : bool, optional
        Whether to leverage the terminal sampling strategy. Significantly fast
        for all sampling tasks but can yield incorrect results in some situations.
        Check out :ref:`terminal_sampling` form more details. The default is False.

    Returns
    -------
    callable
        A function performing the simulation.

    Examples
    --------
    We simulate a function creating a simple GHZ state:

    ::

        from qrisp import *
        from qrisp.jasp import *

        @jaspify
        def main():

            qf = QuantumFloat(5)

            h(qf[0])

            for i in range(1, 5):
                cx(qf[0], qf[i])

            return measure(qf)

        print(main())
        # Yields either 0 or 31

    To highlight the speed of the terminal sampling feature, we :ref:`sample` from a
    uniform superposition

    ::

        def state_prep():
            qf = QuantumFloat(5)
            h(qf)
            return qf

        @jaspify
        def without_terminal_sampling():
            sampling_func = sample(state_prep, shots = 10000)
            return sampling_func()

        @jaspify(terminal_sampling = True)
        def with_terminal_sampling():
            sampling_func = sample(state_prep, shots = 10000)
            return sampling_func()


    Benchmark the time difference:

    ::

        import time

        t0 = time.time()
        res = without_terminal_sampling()
        print(time.time() - t0)
        # Yields
        # 43.78982925

        t0 = time.time()
        res = with_terminal_sampling()
        print(time.time() - t0)
        # Yields
        # 0.550775527


    """
    if isinstance(func, bool):
        terminal_sampling = func
        func = None

    if func is None:
        return lambda x: jaspify(x, terminal_sampling=terminal_sampling)
    # Narrowed rebinding: pyright doesn't propagate the "func is not None" narrowing
    # above into the return_function closure below, since it captures func by
    # reference. Rebinding to a fresh, explicitly-typed name fixes that.
    checked_func: Callable = func

    def return_function(*args) -> Any:
        # Use return_shape=True to capture the output PyTree structure
        jaspr, out_tree = make_jaspr(checked_func, return_shape=True)(*args)
        jaspr_res = simulate_jaspr(jaspr, *args, terminal_sampling=terminal_sampling)

        # Reconstruct the PyTree structure from flat results
        if isinstance(jaspr_res, tuple):
            jaspr_res = tree_unflatten(out_tree, jaspr_res)
        elif jaspr_res is not None:
            # Single value case - still unflatten to handle any wrapping
            jaspr_res = tree_unflatten(out_tree, [jaspr_res])

        if recursive_qv_search(jaspr_res):
            raise Exception("Tried to jaspify function returning a QuantumVariable")
        return jaspr_res

    return return_function


def stimulate(func: Callable) -> Callable:
    """This function leverages the
    `Stim simulator <https://github.com/quantumlib/Stim?tab=readme-ov-file>`_
    to evaluate a Jasp-traceable function containing only Clifford gates.
    Stim is a popular tool to simulate quantum error correction codes.

    .. note::

        To use this simulator, you need stim installed, which can be achieved via
        ``pip install stim``.

    Parameters
    ----------
    func : callable
        The function to simulate.

    Returns
    -------
    callable
        A function performing the simulation.

    Examples
    --------
    We simulate a function creating a simple GHZ state:

    ::

        from qrisp import *
        from qrisp.jasp import *

        @stimulate
        def main():

            qf = QuantumFloat(5)

            h(qf[0])

            for i in range(1, 5):
                cx(qf[0], qf[i])

            return measure(qf)

        print(main())
        # Yields either 0 or 31

    The ``stimulate`` decorator can also simulate real-time features:

    ::

        @stimulate
        def main():

            qf = QuantumFloat(5)

            h(qf[0])

            cl_bl = measure(qf[0])

            with control(cl_bl):
                for i in range(1, 5):
                    x(qf[i])

            return measure(qf)

        print(main())
        # Yields either 0 or 31

    """

    def return_function(*args) -> Any:
        # Use return_shape=True to capture the output PyTree structure
        jaspr, out_tree = make_jaspr(func, return_shape=True)(*args)
        jaspr_res = simulate_jaspr(jaspr, *args, simulator="stim")

        # Reconstruct the PyTree structure from flat results
        if isinstance(jaspr_res, tuple):
            jaspr_res = tree_unflatten(out_tree, jaspr_res)
        elif jaspr_res is not None:
            # Single value case - still unflatten to handle any wrapping
            jaspr_res = tree_unflatten(out_tree, [jaspr_res])

        if recursive_qv_search(jaspr_res):
            raise Exception("Tried to simulate function returning a QuantumVariable")
        return jaspr_res

    return return_function


def _process_jit_equation(
    eqn: JaxprEqn,
    context_dic: ContextDict,
    eqn_evaluator: Callable,
    terminal_sampling: bool,
) -> bool:
    """Process a "jit" equation within the simulate_jaspr interpreter.

    Subgraphs whose signature is purely classical (no quantum state/qubits
    crossing the boundary) are compiled and executed via jax.jit. Everything
    else is replayed equation-by-equation using the same eqn_evaluator.

    Returns
    -------
    bool
        False once the equation has been fully handled (matching the
        eqn_evaluator protocol used by eval_jaxpr).

    """
    function_name = eqn.params["name"]
    jaxpr = eqn.params["jaxpr"]

    if terminal_sampling:
        translation_dic = {
            "expectation_value_eval_function": "ev",
            "sampling_eval_function": "array",
            "dict_sampling_eval_function": "dict",
        }

        if function_name in translation_dic:
            terminal_sampling_evaluator(translation_dic[function_name])(eqn, context_dic, eqn_evaluator=eqn_evaluator)
            return False

    invalues = extract_invalues(eqn, context_dic)

    # If there are only classical values, we attempt to compile using the jax pipeline.
    # This is required, not just an optimization: quantum primitives only have real
    # side effects in their impl rule, which bind() invokes for concrete/eager values.
    # While jax.jit is tracing (as it does here, via compile_cl_func), bind() instead
    # invokes abstract_eval, which for every quantum primitive does nothing but a
    # shape/type check and returns a fresh AbstractQuantumState() -- no interaction
    # with a BufferedQuantumState at all. So jitting a subgraph that carries a quantum
    # type across its boundary would silently drop every quantum operation inside it
    # instead of raising an error, which is why we must rule this out first.
    for var in jaxpr.jaxpr.invars + jaxpr.jaxpr.outvars:
        if isinstance(
            var.aval,
            (AbstractQuantumState, AbstractQubitArray, AbstractQubit),
        ):
            break
    else:
        compiled_function, is_executable = compile_cl_func(jaxpr.jaxpr, function_name)

        # Functions with purely classical inputs/outputs can still contain
        # kernelized quantum functions. This will raise an NotImplementedError
        # when attempting to compile. Since the compile_cl_func is lru_cached
        # we can store this information to avoid further attempts at compiling
        # such a function.
        if is_executable[0]:
            try:
                outvalues = compiled_function(*(jaxpr.consts + invalues))
                if len(jaxpr.jaxpr.outvars) > 1:
                    insert_outvalues(eqn, context_dic, outvalues)
                else:
                    insert_outvalues(eqn, context_dic, [outvalues])
                return False
            except (TypeError, ir.MLIRError):
                is_executable[0] = False

    # We simulate the inverse Gidney mcx via the non-hybrid version because
    # the hybrid version prevents the simulator from fusing gates, which
    # slows down the simulation
    if eqn.params["name"] == "gidney_mcx_inv_impl":
        # Deferred import: qrisp.alg_primitives can trigger a nested load of
        # qrisp.jasp (via qrisp.core.quantum_array) before qrisp.core itself
        # has finished initializing, so this can't be a top-level import.
        from qrisp.alg_primitives.mcx_algs.circuit_library import gidney_qc

        invalues[-1].append(gidney_qc.inverse().to_gate(), invalues[:-1])
        outvalues = [invalues[-1]]
    else:
        outvalues = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator=eqn_evaluator)(*invalues)
    if not isinstance(outvalues, (list, tuple)):
        outvalues = [outvalues]
    insert_outvalues(eqn, context_dic, outvalues)
    return False


def simulate_jaspr(
    jaxpr: ClosedJaxpr | Jaspr,
    *args,
    terminal_sampling: bool = False,
    simulator: Literal["qrisp", "stim"] = "qrisp",
    return_gate_counts: bool = False,
) -> Any:
    """Simulate a jaspr by replaying it equation-by-equation.

    Purely classical "jit" subgraphs are compiled and executed via jax.jit;
    quantum operations are interpreted directly against a BufferedQuantumState.

    """
    if len(jaxpr.jaxpr.outvars) == 1 and isinstance(jaxpr.jaxpr.outvars[0].aval, AbstractQuantumState):
        return None

    if simulator == "stim" and terminal_sampling:
        raise Exception("Terminal sampling with stim is currently not implemented")

    # An invalid simulator value raises identically, one line below, from
    # BufferedQuantumState.__init__ -- no need to duplicate that check here.
    args = list(tree_flatten(args)[0]) + [BufferedQuantumState(simulator)]

    def eqn_evaluator(eqn: JaxprEqn, context_dic: ContextDict) -> bool:
        if eqn.primitive.name == "jit":
            return _process_jit_equation(eqn, context_dic, eqn_evaluator, terminal_sampling)
        if eqn.primitive.name == "jasp.create_quantum_kernel":
            insert_outvalues(eqn, context_dic, BufferedQuantumState(simulator))
            return False
        if eqn.primitive.name == "jasp.consume_quantum_kernel":
            return False
        return True

    with fast_append(3):
        res = eval_jaxpr(jaxpr, eqn_evaluator=eqn_evaluator)(*(args))

    if return_gate_counts:
        return res[-1].gate_counts

    if isinstance(jaxpr, Jaspr):
        if len(jaxpr.jaxpr.outvars) == 2:
            return res[0]
        return res[:-1]
    return res


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def compile_cl_func(jaxpr: Jaxpr, function_name: str) -> tuple[Callable, list[bool]]:
    """Compile a purely classical sub-jaxpr via jax.jit, caching the result.

    function_name is not used in the body but is part of the lru_cache key,
    keeping cache entries for distinctly-named functions separate.

    Returns
    -------
    tuple
        The jax.jit-compiled function, and a single-element mutable list
        used to record (and share across cache hits) whether that function
        turned out to be actually executable.

    """
    return jax.jit(eval_jaxpr(jaxpr)), [True]
