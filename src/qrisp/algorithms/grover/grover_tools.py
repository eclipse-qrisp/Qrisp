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

import numpy as np
from qrisp import (
    QuantumArray,
    QuantumVariable,
    QuantumFloat,
    gate_wrap,
    gphase,
    h,
    mcx,
    mcp,
    mcz,
    p,
    x,
    z,
    merge,
    recursive_qs_search,
    conjugate,
    invert,
    control,
    IterationEnvironment,
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.jasp import check_for_tracing_mode, jrange


# Applies the grover diffuser onto the (list of) quantum variable input_object
def diffuser(input_object, phase=np.pi, state_function=None, reflection_indices=None):
    r"""
    Applies the Grover diffuser onto (multiple) QuantumVariables.

    Parameters
    ----------
    input_object : QuantumVariable | QuantumArray | list[QuantumVariable | QuantumArray]
        The (list of) QuantumVariables to apply the Grover diffuser on.
    phase : float or sympy.Symbol, optional
        Specifies the phase shift. The default is $\pi$, i.e. a
        multi-controlled Z gate.
    state_function : function, optional
        A Python function preparing the initial state.
        By default, the function prepares the uniform superposition state.
    refection_indices : list[int], optional
        A list indicating with respect to which variables the reflection is performed.
        By default, the reflection is performed with respect to all variables in ``input_object``.

    Examples
    --------

    We apply the Grover diffuser onto several QuantumChars:

    >>> from qrisp import QuantumChar
    >>> from qrisp.grover import diffuser
    >>> q_ch_list = [QuantumChar(), QuantumChar(), QuantumChar()]
    >>> diffuser(q_ch_list)
    >>> print(q_ch_list[0].qs)

    .. code-block:: none

                  ┌────────────┐
        q_ch_0.0: ┤0           ├
                  │            │
        q_ch_0.1: ┤1           ├
                  │            │
        q_ch_0.2: ┤2           ├
                  │            │
        q_ch_0.3: ┤3           ├
                  │            │
        q_ch_0.4: ┤4           ├
                  │            │
        q_ch_1.0: ┤5           ├
                  │            │
        q_ch_1.1: ┤6           ├
                  │            │
        q_ch_1.2: ┤7  diffuser ├
                  │            │
        q_ch_1.3: ┤8           ├
                  │            │
        q_ch_1.4: ┤9           ├
                  │            │
        q_ch_2.0: ┤10          ├
                  │            │
        q_ch_2.1: ┤11          ├
                  │            │
        q_ch_2.2: ┤12          ├
                  │            │
        q_ch_2.3: ┤13          ├
                  │            │
        q_ch_2.4: ┤14          ├
                  └────────────┘

    """

    if state_function == None:

        if isinstance(input_object, list):

            def state_function(qargs):
                [h(qv) for qv in qargs]

        else:

            def state_function(qargs):
                h(qargs)

    reflection(input_object, state_function, phase=phase, reflection_indices=reflection_indices)

    """
    if isinstance(input_object, QuantumArray):
        input_object = [qv for qv in input_object.flatten()]

    if isinstance(input_object, (list, tuple)) and reflection_indices is None:
        reflection_indices = [i for i in range(len(input_object))]

    if state_function is not None:

        if isinstance(input_object, (list, tuple)):
            def inv_state_function(args):
                with invert():
                    state_function(*args)
        
        else:
            
            def inv_state_function(args):
                    with invert():
                        state_function(args)

    else:
        if isinstance(input_object, list):

            def inv_state_function(args):
                [h(qv) for qv in args]

        else:

            def inv_state_function(args):
                h(args)

    if isinstance(input_object, (list, tuple)):
        with conjugate(inv_state_function)(input_object):
            if check_for_tracing_mode():
                tag_state({input_object[i]: 0 for i in reflection_indices}, phase=phase)
            else:
                tag_state(
                    {
                        input_object[i]: "0" * input_object[i].size
                        for i in reflection_indices
                    },
                    binary_values=True,
                    phase=phase,
                )
        gphase(np.pi, input_object[0][0])
    else:
        with conjugate(inv_state_function)(input_object):
            if check_for_tracing_mode():
                tag_state({input_object: 0}, phase=phase)
            else:
                tag_state(
                    {input_object: input_object.size * "0"},
                    binary_values=True,
                    phase=phase,
                )
        gphase(np.pi, input_object[0])

    return input_object
    """


def tag_state(tag_specificator, binary_values=False, phase=np.pi):
    """
    Applies a phase tag to (multiple) QuantumVariables. The tagged state is specified in
    the dictionary ``tag_specificator``. This dictionary should contain the
    QuantumVariables as keys and the labels of the states which should be tagged as
    values.


    Parameters
    ----------
    tag_specificator : dict
        A dictionary specifying which state should be tagged.
    binary_values : bool, optional
        If set to True, the values in the tag_specificator dict have to be bitstrings
        instead of labels. The default is False.
    phase : float or sympy.Symbol, optional
        Specify the phase shift the tag should perform. The default is pi, i.e. a
        multi-controlled Z gate.


    Examples
    --------

    We construct an oracle that tags the states -3 and 2 on two QuantumFloats ::

        from qrisp.grover import tag_state

        def test_oracle(qf_list):

            tag_dic = {qf_list[0] : -3, qf_list[1] : 2}

            tag_state(tag_dic)

    """

    qv_list = list(tag_specificator.keys())

    if check_for_tracing_mode():

        states = [qv.encoder(tag_specificator[qv]) for qv in qv_list]

        def conjugator(qv_list, temp_qf):
            for i in range(len(qv_list)):
                mcx(qv_list[i], temp_qf[i], method="balauca", ctrl_state=states[i])

        m = len(qv_list)
        temp_qf = QuantumFloat(m)

        if m == 1:
            with conjugate(conjugator)(qv_list, temp_qf):
                with control(phase == np.pi):
                    z(temp_qf)
                with control(phase != np.pi):
                    p(phase, temp_qf)

        else:
            with conjugate(conjugator)(qv_list, temp_qf):
                with control(phase == np.pi):
                    h(temp_qf[-1])

                    mcx(temp_qf[: m - 1], temp_qf[-1], method="balauca", ctrl_state=-1)

                    h(temp_qf[-1])
                with control(phase != np.pi):
                    mcp(phase, temp_qf, method="balauca", ctrl_state=-1)

        temp_qf.delete()

    else:

        states = [tag_specificator[qv] for qv in qv_list]

        if not len(states):
            states = ["1" * qv.size for qv in qv_list]

        bit_string = ""
        from qrisp.misc import bin_rep

        for i in range(len(qv_list)):
            if binary_values:
                bit_string += states[i][::-1]
            else:
                bit_string += bin_rep(qv_list[i].encoder(states[i]), qv_list[i].size)[
                    ::-1
                ]

        qubit_list = sum([list(qv.reg) for qv in qv_list], [])
        state = bit_string

        if state[-1] == "0":
            x(qubit_list[-1])

        # The control state is the state we are looking for without the base qubit
        ctrl_state = state[:-1]
        if phase == np.pi:
            mcz(qubit_list, ctrl_state=ctrl_state + "1")
        else:
            mcp(phase, qubit_list, ctrl_state=ctrl_state + "1")

        # Apply the final x gate
        if state[-1] == "0":
            x(qubit_list[-1])


def grovers_alg(
    qv_list,
    oracle_function,
    kwargs={},
    iterations=0,
    winner_state_amount=None,
    exact=False,
):
    r"""
    Applies Grover's algorithm to a given oracle (in the form of a Python function).

    Parameters
    ----------
    qv_list : QuantumVariable or list[QuantumVariable]
        A (list of) QuantumVariables on which to execute Grover's algorithm.
    oracle_function : function
        A Python function tagging the winner states.
    kwargs : dict, optional
        A dictionary containing keyword arguments for the oracle. The default is {}.
    iterations : int, optional
        The amount of Grover iterations to perfrom.
    winner_state_amount : int, optional
        If not given the amount of iterations, the established formula will be used
        based on the amount of winner states. The default is 1.
    exact : bool, optional
        If set to True, the `exact version <https://arxiv.org/pdf/quant-ph/0106071.pdf>`
        of Grover's algorithm will be evaluated. For this, the correct
        ``winner_state_amount`` has to be supplied and the oracle has to support the
        keyword argument ``phase`` which specifies how much the winner states are
        phaseshifted (in standard Grover this would be $\pi$).

    Raises
    ------

    Exception
        Applied oracle introducing new QuantumVariables without uncomputing/deleting

    Examples
    --------

    We construct an oracle that tags the states -3 and 2 on two QuantumFloats and apply
    Grover's algorithm to it. ::

        from qrisp import QuantumFloat

        #Create list of QuantumFloats
        qf_list = [QuantumFloat(2, signed = True), QuantumFloat(2, signed = True)]

        from qrisp.grover import tag_state, grovers_alg

        def test_oracle(qf_list):

            tag_dic = {qf_list[0] : -3, qf_list[1] : 2}
            tag_state(tag_dic)

        grovers_alg(qf_list, test_oracle)

    >>> from qrisp.misc import multi_measurement
    >>> print(multi_measurement(qf_list))
    {(-3, 2): 0.9966, (0, 0): 0.0001, (0, 1): 0.0001, (0, 2): 0.0001, (0, 3): 0.0001,
    (0, -4): 0.0001, (0, -3): 0.0001, (0, -2): 0.0001, (0, -1): 0.0001, (1, 0): 0.0001,
    (1, 1): 0.0001, (1, 2): 0.0001, (1, 3): 0.0001, (1, -4): 0.0001, (1, -3): 0.0001,
    (1, -2): 0.0001, (1, -1): 0.0001, (2, 0): 0.0001, (2, 1): 0.0001, (2, 2): 0.0001,
    (2, 3): 0.0001, (2, -4): 0.0001, (2, -3): 0.0001, (2, -2): 0.0001, (2, -1): 0.0001,
    (3, 0): 0.0001, (3, 1): 0.0001, (3, 2): 0.0001, (3, 3): 0.0001, (3, -4): 0.0001,
    (3, -3): 0.0001, (3, -2): 0.0001, (3, -1): 0.0001, (-4, 0): 0.0001, (-4, 1): 0.0001,
     (-4, 2): 0.0001, (-4, 3): 0.0001, (-4, -4): 0.0001, (-4, -3): 0.0001,
     (-4, -2): 0.0001, (-4, -1): 0.0001, (-3, 0): 0.0001, (-3, 1): 0.0001,
     (-3, 3): 0.0001, (-3, -4): 0.0001, (-3, -3): 0.0001, (-3, -2): 0.0001,
     (-3, -1): 0.0001, (-2, 0): 0.0001, (-2, 1): 0.0001, (-2, 2): 0.0001,
     (-2, 3): 0.0001, (-2, -4): 0.0001, (-2, -3): 0.0001, (-2, -2): 0.0001,
     (-2, -1): 0.0001, (-1, 0): 0.0001, (-1, 1): 0.0001, (-1, 2): 0.0001,
     (-1, 3): 0.0001, (-1, -4): 0.0001, (-1, -3): 0.0001, (-1, -2): 0.0001,
     (-1, -1): 0.0001}

    **Exact Grovers Algorithm**

    In the next example, we will showcase the ``exact`` functionality. For this we
    create an oracle, which tags all the states of a QuantumVariable, that contain 3
    ones and 2 zeros.

    To count the amount of ones we use quantum phase estimation on the operator

    .. math::

        U = \text{exp}\left(\frac{i 2 \pi}{2^k}
        \sum_{i = 0}^{n-1} ( 1 - \sigma_{z}^i )\right)


    ::

        from qrisp import QPE, p, QuantumVariable, lifted
        from qrisp.grover import grovers_alg, tag_state
        import numpy as np

        def U(qv, prec = None, iter = 1):
            for i in range(qv.size):
                p(iter*2*np.pi/2**prec, qv[i])

        @lifted
        def count_ones(qv):
            prec = int(np.ceil(np.log2(qv.size+1)))
            res = QPE(qv, U, precision = prec, iter_spec = True, kwargs = {"prec" : prec})
            res <<= prec
            return res


    Quick test:

    >>> qv = QuantumVariable(5)
    >>> qv[:] = {"11000" : 1, "11010" : 1, "11110" : 1}
    >>> count_qf = count_ones(qv)
    >>> count_qf.qs.statevector()
    sqrt(3)*(|11000>*|2> + |11010>*|3> + |11110>*|4>)/3

    We now define the oracle ::

        def counting_oracle(qv, phase = np.pi, k = 1):

            count_qf = count_ones(qv)

            tag_state({count_qf : k}, phase = phase)

            count_qf.uncompute()

    And evaluate Grover's algorithm ::

        n = 5
        k = 3
        qv = QuantumVariable(n)

        import math

        grovers_alg(qv, counting_oracle, exact = True, winner_state_amount = math.comb(n,k), kwargs = {"k" : k})  # noqa


    >>> print(qv)
    {'11100': 0.1,
     '11010': 0.1,
     '10110': 0.1,
     '01110': 0.1,
     '11001': 0.1,
     '10101': 0.1,
     '01101': 0.1,
     '10011': 0.1,
     '01011': 0.1,
     '00111': 0.1}

    We see that contrary to regular Grover's algorithm, the states which have not been
    tagged by the oracle have 0 percent measurement probability.

    """

    if exact and winner_state_amount is None:
        raise Exception(
            "Tried to call exact Grover's algorithm without specifying "
            "winner_state_amount"
        )
    elif winner_state_amount is None:
        winner_state_amount = 1

    if check_for_tracing_mode():
        import jax.numpy as jnp
    else:
        import numpy as jnp

    if isinstance(qv_list, list):
        N = 2 ** jnp.sum(jnp.array([qv.size for qv in qv_list]))
    elif isinstance(qv_list, QuantumArray):
        N = 2 ** sum(jnp.array([qv.size for qv in qv_list.flatten()]))
    elif isinstance(qv_list, QuantumVariable):
        N = 2**qv_list.size

    if exact:
        # Implementation for phase calculation for exact grovers alg as in
        # https://arxiv.org/pdf/quant-ph/0106071.pdf
        iterations = 1
        tmp = (
            jnp.sin(jnp.pi / (4 * (iterations - 1) + 6))
            * (N / winner_state_amount) ** 0.5
        )

        def body_fun(state):
            iterations, tmp = state
            return (
                iterations + 1,
                jnp.sin(jnp.pi / (4 * (iterations - 1) + 6))
                * (N / winner_state_amount) ** 0.5,
            )

        def cond_fun(state):
            iterations, tmp = state
            return tmp > 1

        state = (iterations, tmp)

        if check_for_tracing_mode():
            from jax.lax import while_loop

            iterations, tmp = while_loop(cond_fun, body_fun, state)
        else:
            while cond_fun(state):
                state = body_fun(state)
            iterations, tmp = state

        phi = 2 * jnp.arcsin(
            jnp.sin(jnp.pi / (4 * (iterations - 1) + 6))
            * (N / winner_state_amount) ** 0.5
        )

    else:
        if iterations == 0:
            iterations = jnp.pi / 4 * jnp.sqrt(N / winner_state_amount)
            iterations = jnp.int64(jnp.round(iterations))

    if isinstance(qv_list, (list, QuantumArray)):
        [h(qv) for qv in qv_list]
    else:
        h(qv_list)

    if check_for_tracing_mode():

        for i in jrange(iterations):
            if exact:
                oracle_function(qv_list, phase=phi, **kwargs)
                diffuser(qv_list, phase=phi)
            else:
                oracle_function(qv_list, **kwargs)
                diffuser(qv_list)

    else:

        merge(qv_list)
        qs = recursive_qs_search(qv_list)[0]
        qv_amount = len(qs.qv_list)

        with IterationEnvironment(qs, iterations):
            if exact:
                oracle_function(qv_list, phase=phi, **kwargs)
                diffuser(qv_list, phase=phi)
            else:
                oracle_function(qv_list, **kwargs)
                diffuser(qv_list)

        if qv_amount != len(qs.qv_list):
            raise Exception(
                "Applied oracle introducing new QuantumVariables without uncomputing/deleting"
            )


# Workaround to keep the docstring but still gatewrap

temp = diffuser.__doc__

diffuser = gate_wrap(permeability=[], is_qfree=False)(diffuser)

diffuser.__doc__ = temp


temp = tag_state.__doc__

tag_state = gate_wrap(permeability="args", is_qfree=True)(tag_state)

tag_state.__doc__ = temp
