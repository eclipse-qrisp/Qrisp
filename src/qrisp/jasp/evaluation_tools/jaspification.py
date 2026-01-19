"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

from functools import lru_cache

import jax
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.lib.mlir import ir

from qrisp.jasp.interpreter_tools import extract_invalues, insert_outvalues, eval_jaxpr
from qrisp.jasp.evaluation_tools.buffered_quantum_state import BufferedQuantumState
from qrisp.jasp.primitives import (
    AbstractQuantumCircuit,
    AbstractQubitArray,
    AbstractQubit,
)
from qrisp.core import recursive_qv_search
from qrisp.circuit import fast_append


def jaspify(func=None, terminal_sampling=False):
    """
    This simulator is the established Qrisp simulator linked to the Jasp infrastructure.
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

    from qrisp.jasp import make_jaspr

    treedef_container = []

    def tracing_function(*args):
        res = func(*args)
        flattened_values, tree_def = tree_flatten(res)
        treedef_container.append(tree_def)
        return flattened_values

    def return_function(*args):
        # To prevent "accidental deletion" induced non-determinism we set the
        # garbage collection mode to manual
        if terminal_sampling:
            garbage_collection = "manual"
        else:
            garbage_collection = "auto"
        jaspr = make_jaspr(tracing_function, garbage_collection=garbage_collection)(
            *args
        )
        jaspr_res = simulate_jaspr(jaspr, *args, terminal_sampling=terminal_sampling)
        if isinstance(jaspr_res, tuple):
            jaspr_res = tree_unflatten(treedef_container[0], jaspr_res)
        if len(recursive_qv_search(jaspr_res)):
            raise Exception("Tried to jaspify function returning a QuantumVariable")
        return jaspr_res

    return return_function


def stimulate(func=None):
    """
    This function leverages the
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

    from qrisp.jasp import make_jaspr

    treedef_container = []

    def tracing_function(*args):
        res = func(*args)
        flattened_values, tree_def = tree_flatten(res)
        treedef_container.append(tree_def)
        return flattened_values

    def return_function(*args):
        jaspr = make_jaspr(tracing_function)(*args)
        jaspr_res = simulate_jaspr(jaspr, *args, simulator="stim")
        if isinstance(jaspr_res, tuple):
            jaspr_res = tree_unflatten(treedef_container[0], jaspr_res)
        if len(recursive_qv_search(jaspr_res)):
            raise Exception("Tried to simulate function returning a QuantumVariable")
        return jaspr_res

    return return_function


def simulate_jaspr(
    jaxpr,
    *args,
    terminal_sampling=False,
    simulator="qrisp",
    return_gate_counts=False,
    return_depth=False,
):

    from qrisp.jasp import Jaspr
    from qrisp.alg_primitives.mcx_algs.circuit_library import gidney_qc

    if len(jaxpr.jaxpr.outvars) == 1 and isinstance(
        jaxpr.jaxpr.outvars[0].aval, AbstractQuantumCircuit
    ):
        return None

    if simulator == "stim":
        if terminal_sampling:
            raise Exception("Terminal sampling with stim is currently not implemented")
    elif not simulator == "qrisp":
        raise Exception(f"Don't know simulator {simulator}")

    args = list(tree_flatten(args)[0]) + [BufferedQuantumState(simulator)]

    def eqn_evaluator(eqn, context_dic):

        if eqn.primitive.name == "jit":

            function_name = eqn.params["name"]
            jaxpr = eqn.params["jaxpr"]

            if terminal_sampling:

                translation_dic = {
                    "expectation_value_eval_function": "ev",
                    "sampling_eval_function": "array",
                    "dict_sampling_eval_function": "dict",
                }

                from qrisp.jasp.interpreter_tools import terminal_sampling_evaluator

                if function_name in translation_dic:
                    terminal_sampling_evaluator(translation_dic[function_name])(
                        eqn, context_dic, eqn_evaluator=eqn_evaluator
                    )
                    return

            invalues = extract_invalues(eqn, context_dic)

            # If there are only classical values, we attempt to compile using the jax pipeline
            for var in jaxpr.jaxpr.invars + jaxpr.jaxpr.outvars:
                if isinstance(
                    var.aval,
                    (AbstractQuantumCircuit, AbstractQubitArray, AbstractQubit),
                ):
                    break
            else:

                compiled_function, is_executable = compile_cl_func(
                    jaxpr.jaxpr, function_name
                )

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
            if eqn.params["name"] == "gidney_mcx_inv":
                invalues[-1].append(gidney_qc.inverse().to_gate(), invalues[:-1])
                outvalues = [invalues[-1]]
            else:
                outvalues = eval_jaxpr(
                    eqn.params["jaxpr"], eqn_evaluator=eqn_evaluator
                )(*invalues)
            if not isinstance(outvalues, (list, tuple)):
                outvalues = [outvalues]
            insert_outvalues(eqn, context_dic, outvalues)
        elif eqn.primitive.name == "jasp.create_quantum_kernel":
            insert_outvalues(eqn, context_dic, BufferedQuantumState(simulator))
        elif eqn.primitive.name == "jasp.consume_quantum_kernel":
            pass
        else:
            return True

    with fast_append(3):
        res = eval_jaxpr(jaxpr, eqn_evaluator=eqn_evaluator)(*(args))

    if return_gate_counts:
        return res[-1].gate_counts

    if return_depth:
        raise NotImplementedError("Depth calculation not yet implemented in jaspify")

    if isinstance(jaxpr, Jaspr):
        if len(jaxpr.jaxpr.outvars) == 2:
            return res[0]
        else:
            return res[:-1]
    else:
        return res


@lru_cache(maxsize=int(1e5))
def compile_cl_func(jaxpr, function_name):
    return jax.jit(eval_jaxpr(jaxpr)), [True]
