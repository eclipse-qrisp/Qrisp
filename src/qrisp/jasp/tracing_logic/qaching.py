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

import jax
from jax.extend.core import ClosedJaxpr

from qrisp.core import recursive_qv_search, recursive_qa_search

from qrisp.jasp.primitives import AbstractQuantumState
from qrisp.jasp.tracing_logic import TracingQuantumSession, check_for_tracing_mode, get_last_equation


def qache(*func, **kwargs):
    """
    This decorator allows you to mark a function as "reusable". Reusable here means
    that the jasp expression of this function will be cached and reused in the next
    calls (if the function is called with the same signature).

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

    Returns
    -------
    qached_function : callable
        A function that will be traced on it's first execution and retrieved from
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
    let inner_function = { lambda ; a:QuantumState b:QubitArray. let
        c:Qubit = jasp.get_qubit b 0
        d:QuantumState = jasp.h a c
        e:Qubit = jasp.get_qubit b 1
        f:QuantumState = jasp.cx d c e
        g:QuantumState h:bool[] = jasp.measure f c
      in (g, h) } in
    let inner_function1 = { lambda ; i:QuantumState j:QubitArray k:i32[] l:bool[]. let
        m:Qubit = jasp.get_qubit j 0
        n:QuantumState = jasp.h i m
        o:Qubit = jasp.get_qubit j 1
        p:QuantumState = jasp.cx n m o
        q:QuantumState r:bool[] = jasp.measure p m
      in (q, r) } in
    { lambda ; s:QuantumState. let
        t:QuantumState u:QubitArray = jasp.create_qubits s 2
        v:QuantumState w:QubitArray = jasp.create_qubits t 2
        x:QuantumState y:bool[] = pjit[name=inner_function jaxpr=inner_function] v
          u
        z:QuantumState ba:bool[] = pjit[name=inner_function jaxpr=inner_function1] x
          w 0 False
        bb:QuantumState bc:bool[] = pjit[name=inner_function jaxpr=inner_function] z
          u
        bd:QuantumState be:bool[] = pjit[
          name=inner_function
          jaxpr=inner_function1
        ] bb w 0 False
        bf:bool[] = and y ba
        bg:bool[] = and bf bc
        bh:bool[] = and bg be
        bi:QuantumState = jasp.reset bd u
        bj:QuantumState = jasp.delete_qubits bi u
      in (bj, bh) }

    As expected, we see three different function definitions:

    * The first one describes ``inner_function`` called with a :ref:`QuantumVariable`. For this kind of signature only the ``QubitArray`` is required.
    * The second one describes ``inner_function`` called with :ref:`QuantumFloat`. Additionally to the ``QubitArray``, the ``.exponent`` and ``.signed`` attribute are also passed to the function.
    * The third function definition is ``outer_function``, which calls the previously defined functions.

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
        # Yields: Exception: Found in-place parameter modification of QuantumVariable qf

    """

    if len(kwargs) and len(func) == 0:
        return lambda x: qache_helper(x, kwargs)
    elif len(kwargs) and len(func):
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

        abs_qc = kwargs[10*"~"]
        del kwargs[10*"~"]

        # Set the given AbstractQuantumState as the
        # one carried by the tracing QuantumSession
        abs_qs = TracingQuantumSession.get_instance()
        abs_qs.abs_qc = abs_qc

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
            abs_qs.register_qv(qv, None)
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
                if not flat_qv[i] is flattened_qvs.pop(0):
                    raise Exception(
                        f"Found in-place parameter modification of QuantumVariable {qv.name}"
                    )

        new_abs_qc = abs_qs.abs_qc
        # Return the result and the result AbstractQuantumState.
        return res, new_abs_qc

    # Modify the name of the ammended function to reflect the input
    ammended_function.__name__ = func.__name__
    # Wrap in jax.jit
    ammended_function = jax.jit(ammended_function, **jax_kwargs)

    from qrisp.jasp.tracing_logic import flatten_qv

    # We now prepare the return function
    def return_function(*args, **kwargs):

        # If we are not in tracing mode, simply execute the function
        if not check_for_tracing_mode():
            return func(*args, **kwargs)

        # Get the AbstractQuantumState for tracing
        abs_qs = TracingQuantumSession.get_instance()
        abs_qs.start_tracing(abs_qs.abs_qc)

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
        ammended_kwargs[10*"~"] = abs_qs.abs_qc
        try:
            res, abs_qc_new = ammended_function(*args, **ammended_kwargs)
        except Exception as e:
            abs_qs.conclude_tracing()
            raise e

        abs_qs.conclude_tracing()

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
        abs_qs.abs_qc = abs_qc_new

        # The QuantumVariables from the result went through a flatten/unflattening cycly.
        # The unflattening creates a new QuantumVariable object, that is however not yet
        # registered in any QuantumSession. We register these in the current QuantumSession.
        for qv in recursive_qv_search(res):
            abs_qs.register_qv(qv, None)

        # Return the result.
        return res

    return return_function
