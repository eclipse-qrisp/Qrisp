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

"""Implements the qache decorator for caching and reusing traced Jasp function jaxprs."""

import functools

import jax

from qrisp.core import recursive_qa_search, recursive_qv_search
from qrisp.jasp.primitives import AbstractQuantumState
from qrisp.jasp.tracing_logic import (
    TracingQuantumSession,
    check_for_tracing_mode,
    get_last_equation,
    tracing_scope,
)


def qache(*func, **kwargs):
    """This decorator allows you to mark a function as "reusable".

    Reusable here means
    that the jasp expression of this function will be cached and reused in the next
    calls (if the function is called with the same signature, i.e. arguments of the
    same abstract type/shape, and, if any arguments are marked static via ``kwargs``,
    the same concrete value for those).

    A qached function therefore has to be traced by the Python interpreter only once
    and after that the function can be called without any Python-interpreter induced
    delay. This can significantly speed up the compilation process.

    Using the ``qache`` decorator not only improves the compilation speed but also
    enables the compiler to speed up transformation processes.

    .. warning::

        Two important rules apply to the ``qache`` decorator to adhere to the
        functional programming paradigm.

        * It is illegal to have a qached function return a QuantumVariable that has been passed as an argument to the function.
        * It is illegal to modify traced attributes of QuantumVariables that have been passed as an argument to the function.

        See the examples section for representatives of these cases.

    Parameters
    ----------
    func : callable
        The function to be qached.
    kwargs : dict, optional
        Keyword arguments that are forwarded to `jax.jit <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>`_.

    Returns
    -------
    qached_function : callable
        A function that will be traced on its first execution and retrieved from
        the cache in any other call.

    Examples
    --------
    We create a simple function that is qached. To simulate an expensive compilation
    task we insert a ``time.sleep`` command.

    ::

        import time
        from qrisp import *
        from qrisp.jasp import qache

        @qache
        def inner_function(qv):
            h(qv[0])
            cx(qv[0], qv[1])
            res_bl = measure(qv[0])

            # Simulate demanding compilation procedure by calling
            time.sleep(1)

            return res_bl

        def main():
            a = QuantumVariable(2)
            b = QuantumFloat(2)

            bl_0 = inner_function(a)
            bl_1 = inner_function(b)
            bl_2 = inner_function(a)
            bl_3 = inner_function(b)

            return bl_0 & bl_1 & bl_2 & bl_3

        # Measure the time required for tracing
        t0 = time.time()
        jaspr = make_jaspr(main)()
        print(time.time() - t0) # 2.0225703716278076

    Even though ``inner_function`` has been called 4 times, we only see a delay of 2 seconds.
    This is because the function has been called with two different quantum types, implying it
    has been traced twice and recalled from the cache twice. We take a look at the :ref:`jaspr`.

    >>> print(jaspr)
    let inner_function = { lambda ; a:QubitArray b:QuantumState. let
        c:Qubit = jasp.get_qubit a 0:i64[]
        d:QuantumState = jasp.quantum_gate[gate=h] c b
        e:Qubit = jasp.get_qubit a 1:i64[]
        f:QuantumState = jasp.quantum_gate[gate=cx] c e d
        g:bool[] h:QuantumState = jasp.measure c f
      in (g, h) } in
    let inner_function1 = { lambda ; i:QubitArray j:i64[] k:QuantumState. let
        l:Qubit = jasp.get_qubit i 0:i64[]
        m:QuantumState = jasp.quantum_gate[gate=h] l k
        n:Qubit = jasp.get_qubit i 1:i64[]
        o:QuantumState = jasp.quantum_gate[gate=cx] l n m
        p:bool[] q:QuantumState = jasp.measure l o
      in (p, q) } in
    { lambda ; r:QuantumState. let
        s:QubitArray t:QuantumState = jasp.create_qubits 2:i64[] r
        u:QubitArray v:QuantumState = jasp.create_qubits 2:i64[] t
        w:bool[] x:QuantumState = jit[name=inner_function jaxpr=inner_function] s v
        y:bool[] z:QuantumState = jit[name=inner_function jaxpr=inner_function1] u 0:i64[]
          x
        ba:bool[] bb:QuantumState = jit[name=inner_function jaxpr=inner_function] s z
        bc:bool[] bd:QuantumState = jit[name=inner_function jaxpr=inner_function1] u
          0:i64[] bb
        be:bool[] = and w y
        bf:bool[] = and be ba
        bg:bool[] = and bf bc
      in (bg, bd) }

    As expected, we see three different function definitions:

    * The first one describes ``inner_function`` called with a :ref:`QuantumVariable`. For this kind
      of signature only the ``QubitArray`` is required.
    * The second one describes ``inner_function`` called with :ref:`QuantumFloat`. Additionally to the
      ``QubitArray``, the ``.exponent`` attribute is also passed to the function, because it is a *traced*
      attribute.
    * The third block is the anonymous top-level function representing ``main``, which calls the
      previously defined functions.

    **Illegal functions**

    We will now demonstrate what type of functions can not be qached.

    ::

        @qache
        def inner_function(qv):
            h(qv[0])
            return qv

        @jaspify
        def main():
            qf_0 = QuantumFloat(2)
            qf_1 = inner_function(qf_0)
            return measure(qf_1)

        main()
        # Yields: Exception: Found parameter QuantumVariable within returned results

    ``inner_function`` returns a :ref:`QuantumVariable` that has been passed as an
    argument and can therefore not be qached.

    The second case of an illegal functions is a function that tries to modify
    a traced attribute of a ``QuantumVariable`` that has been passed as an argument.
    A traced attribute is for instance the ``exponent`` attribute of :ref:`QuantumFloat`.

    ::

        @qache
        def inner_function(qf):
            qf.exponent += 1

        @jaspify
        def main():
            qf = QuantumFloat(2)
            inner_function(qf)

        main()
        # Yields: Exception: Found in-place parameter modification of QuantumVariable

    """
    if kwargs and len(func) == 0:
        return lambda x: qache_helper(x, kwargs)
    elif kwargs and func:
        return qache_helper(func[0], kwargs)
    else:
        return qache_helper(func[0], {})


# temp_list = [False]
def qache_helper(func, jax_kwargs):

    # To achieve the desired behavior we leverage the Jax inbuild caching mechanism.
    # This feature can be used by calling a jitted function in a tracing context.
    # To cache the function we therefore simply need to wrap it with jit and
    # it will be properly cached.

    # if func.__name__ == "jasp_qq_gidney_adder":
    # if temp_list[0]:
    # raise
    # temp_list[0] = True

    # There are however some more things to consider.

    # The Qrisp function doesn't have the AbstractQuantumState object (which is carried by
    # the tracing QuantumSession) in the signature.

    # To make jax properly treat this, we modify the function signature

    # This function performs the input function but also has the AbstractQuantumState
    # in the signature.
    def ammended_function(*args, **kwargs):

        abs_qst = kwargs[10 * "~"]
        del kwargs[10 * "~"]

        # Set the given AbstractQuantumState as the
        # one carried by the tracing QuantumSession
        tr_qs = TracingQuantumSession.get_instance()
        tr_qs.abs_qst = abs_qst

        # We now iterate through the QuantumVariables of the signature to perform two steps:
        # 1. The QuantumVariables from the signature went through a flatten/unflattening process.
        # The unflattening creates a copy of the QuantumVariable object, which is however not
        # registered in any QuantumSession. We therefore need to register them.
        # 2. To prevent the user from performing any in-place modifications of traced QuantumVariable
        # attributes, we collect the tracers to compare them after the function has concluded.
        arg_qvs = recursive_qv_search(args)
        arg_qvs += [qa.qtype for qa in recursive_qa_search(args)]

        flattened_qvs = []
        for qv in arg_qvs:
            tr_qs.register_qv(qv, None)
            flattened_qvs.extend(list(flatten_qv(qv)[0]))

        # Execute the function
        res = func(*args, **kwargs)

        res_qvs = recursive_qv_search(res)

        # It is not legal to return a QuantumVariable that was already given in the parameters.
        if set([hash(qv) for qv in res_qvs]).intersection([hash(qv) for qv in arg_qvs]):
            raise Exception("Found parameter QuantumVariable within returned results")

        res_qvs += [qa.qtype for qa in recursive_qa_search(res)]

        # Check whether there have been in-place modifications of traced attributes of QuantumVariables.
        for qv in arg_qvs:
            flat_qv = list(flatten_qv(qv)[0])
            for i in range(len(flat_qv)):
                if flat_qv[i] is not flattened_qvs.pop(0):
                    raise Exception(f"Found in-place parameter modification of QuantumVariable {qv.name}")

        new_abs_qst = tr_qs.abs_qst
        # Return the result and the result AbstractQuantumState.
        return res, new_abs_qst

    # Modify the name of the ammended function to reflect the input
    ammended_function.__name__ = func.__name__
    # Wrap in jax.jit
    ammended_function = jax.jit(ammended_function, **jax_kwargs)

    from qrisp.jasp.tracing_logic import flatten_qv

    # We now prepare the return function
    @functools.wraps(func)
    def return_function(*args, **kwargs):

        # If we are not in tracing mode, simply execute the function
        if not check_for_tracing_mode():
            return func(*args, **kwargs)

        # Get the AbstractQuantumState for tracing
        tr_qs = TracingQuantumSession.get_instance()

        with tracing_scope(tr_qs, tr_qs.abs_qst):
            # Make sure literals are 32 bit
            args = list(args)
            # for i in range(len(args)):
            #     if isinstance(args[i], bool):
            #         args[i] = jnp.array(args[i], dtype = jnp.bool)
            #     elif isinstance(args[i], int):
            #         args[i] = jnp.array(args[i], dtype = jnp.int64)
            #     elif isinstance(args[i], float):
            #         args[i] = jnp.array(args[i], dtype = jnp.float64)
            #     elif isinstance(args[i], complex):
            #         args[i] = jnp.array(args[i], dtype = jnp.complex)

            # Excecute the function
            ammended_kwargs = dict(kwargs)
            ammended_kwargs[10 * "~"] = tr_qs.abs_qst
            res, abs_qst_new = ammended_function(*args, **ammended_kwargs)

        tr_qs.conclude_tracing()

        # Convert the jaxpr from the traced equation in to a Jaspr
        from qrisp.jasp import Jaspr

        eqn = get_last_equation()

        jaxpr = eqn.params["jaxpr"].jaxpr

        if not isinstance(eqn.invars[-1].aval, AbstractQuantumState):
            for i in range(len(eqn.invars)):
                if isinstance(eqn.invars[i].aval, AbstractQuantumState):
                    eqn.invars[-1], eqn.invars[i] = eqn.invars[i], eqn.invars[-1]
                    break
        if not isinstance(jaxpr.invars[-1].aval, AbstractQuantumState):
            for i in range(len(jaxpr.invars)):
                if isinstance(jaxpr.invars[i].aval, AbstractQuantumState):
                    jaxpr.invars[-1], jaxpr.invars[i] = (
                        jaxpr.invars[i],
                        jaxpr.invars[-1],
                    )
                    break

        eqn.params["jaxpr"] = Jaspr.from_cache(eqn.params["jaxpr"])

        # Update the AbstractQuantumState of the TracingQuantumSession
        tr_qs.abs_qst = abs_qst_new

        # The QuantumVariables from the result went through a flatten/unflattening cycly.
        # The unflattening creates a new QuantumVariable object, that is however not yet
        # registered in any QuantumSession. We register these in the current QuantumSession.
        for qv in recursive_qv_search(res):
            tr_qs.register_qv(qv, None)

        # Return the result.
        return res

    # Marks the function as qached for gate_wrap, see jasp_gate_wrap.
    return_function.qache_jax_kwargs = jax_kwargs

    return return_function


def jasp_gate_wrap(function, args, kwargs, permeability, is_qfree, name):
    """Jasp implementation of :func:`qrisp.gate_wrap`.

    The permeability and qfree-ness specification is stored on the Jaspr of a
    ``jit`` equation. A qached function creates this equation itself. Any other
    function is traced inline inside a :class:`GateWrapEnvironment`, which turns
    its body into such an equation when the environments are flattened.
    """
    jax_kwargs = getattr(function, "qache_jax_kwargs", None)

    if jax_kwargs is None:
        from qrisp.environments import GateWrapEnvironment

        arg_qubits, arg_permeability = [], []
        for i, arg in enumerate(args):
            qubits = _qubit_leaves(arg)
            arg_qubits.extend(qubits)
            arg_permeability.extend([_arg_permeability(permeability, i)] * len(qubits))

        env = GateWrapEnvironment(
            name=function.__name__ if name is None else name,
            arg_qubits=arg_qubits,
            arg_permeability=arg_permeability,
            result_permeability=None if permeability is None else permeability == "full",
            is_qfree=is_qfree,
        )
        with env:
            return function(*args, **kwargs)

    res = function(*args, **kwargs)

    eqn = get_last_equation()
    if name is not None:
        eqn.params["name"] = name

    # Static arguments are not passed to the jit, so they have no invars.
    static_argnums = jax_kwargs.get("static_argnums", ())
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    static_argnums = {i % len(args) for i in static_argnums}
    dynamic_pos = [i for i in range(len(args)) if i not in static_argnums]

    _set_gate_wrap_properties(eqn.params["jaxpr"], dynamic_pos, [args[i] for i in dynamic_pos], permeability, is_qfree)

    return res


def _set_gate_wrap_properties(jaspr, dynamic_pos, dynamic_args, permeability, is_qfree):
    """Store a :func:`qrisp.gate_wrap` permeability/qfree specification on ``jaspr``.

    The specification refers to positional arguments. Each dynamic positional
    argument occupies as many invars as its pytree has leaves; its qubit-type
    invars receive the argument's permeability. Returned qubits are permeable
    only for ``permeability="full"``.
    """
    jaspr.isqfree = is_qfree

    if permeability is None:
        return

    invar_index = 0
    for i, arg in zip(dynamic_pos, dynamic_args):
        num_leaves = len(jax.tree_util.tree_leaves(arg))
        for var in jaspr.invars[invar_index : invar_index + num_leaves]:
            if _is_quantum(var):
                jaspr.permeability[var] = _arg_permeability(permeability, i)
        invar_index += num_leaves

    for var in jaspr.outvars[:-1]:
        if _is_quantum(var):
            jaspr.permeability[var] = permeability == "full"


def _arg_permeability(permeability, i):
    """Permeability of the ``i``-th positional argument under a :func:`qrisp.gate_wrap` specification."""
    if permeability is None:
        return None
    if isinstance(permeability, list):
        return i in permeability
    if permeability not in ("args", "full"):
        raise Exception(f"Don't know permeability option {permeability}")
    return True


def _is_quantum(x):
    """Return True if ``x`` (a Var or tracer) is a qubit or qubit array."""
    from qrisp.jasp.primitives import AbstractQubit, AbstractQubitArray

    return isinstance(getattr(x, "aval", None), (AbstractQubit, AbstractQubitArray))


def _qubit_leaves(arg):
    """Return the qubit and qubit-array tracers in ``arg``.

    Unlike pytree flattening, this does not sort dict keys, which for keys
    like QuantumFloats would trace comparison circuits.
    """
    from qrisp.core import QuantumArray, QuantumVariable
    from qrisp.jasp.tracing_logic import DynamicQubitArray

    if isinstance(arg, QuantumVariable):
        arg = arg.reg
    elif isinstance(arg, QuantumArray):
        arg = arg.qb_array
    if isinstance(arg, DynamicQubitArray):
        arg = arg.tracer

    if _is_quantum(arg):
        return [arg]
    if isinstance(arg, dict):
        arg = [item for key_value in arg.items() for item in key_value]
    if isinstance(arg, (list, tuple)):
        return [qubit for item in arg for qubit in _qubit_leaves(item)]
    return []
