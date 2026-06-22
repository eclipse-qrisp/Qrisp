"""
********************************************************************************
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

from collections.abc import Callable, Sequence
import numpy as np
from typing import Any

from qrisp import (
    QuantumArray,
    QuantumVariable,
    QuantumFloat,
    gate_wrap,
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
    control,
    IterationEnvironment,
)
from qrisp.alg_primitives.reflection import reflection
from qrisp.jasp import check_for_tracing_mode, jrange
from qrisp.typing import FloatLike


# Applies the grover diffuser onto the (sequence of) quantum variable input_object
def diffuser(
    input_object: QuantumVariable
    | QuantumArray
    | Sequence[QuantumVariable | QuantumArray],
    phase: FloatLike = np.pi,
    state_function: Callable | None = None,
    reflection_indices: list[int] | None = None,
):
    r"""
    Applies the Grover diffuser onto (multiple) QuantumVariables.

    Parameters
    ----------
    input_object : QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray]
        The (list of) QuantumVariables to apply the Grover diffuser on.
    phase : FloatLike, optional
        Specifies the phase shift. The default is $\pi$, i.e. a
        multi-controlled Z gate.
    state_function : function, optional
        A Python function preparing the initial state.
        By default, the function prepares the uniform superposition state.
    reflection_indices : list[int], optional
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

    if state_function is None:

        def _state_function(*qargs):
            for arg in qargs:
                h(arg)

        state_function = _state_function

    reflection(
        input_object, state_function, phase=phase, reflection_indices=reflection_indices
    )


def tag_state(
    tag_specificator: dict[QuantumVariable, Any],
    binary_values: bool = False,
    phase: FloatLike = np.pi,
):
    r"""
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
    phase : FloatLike, optional
        Specify the phase shift the tag should perform. The default is $\pi$, i.e. a
        multi-controlled Z gate.

    Examples
    --------

    We construct an oracle that tags the states -3 and 2 on two QuantumFloats

    ::

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
    args: QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray],
    oracle_function: Callable,
    kwargs: dict[str, Any] | None = None,
    iterations: int | None = None,
    winner_state_amount: int | None = None,
    exact: bool = False,
):
    r"""
    Applies Grover's algorithm to a given oracle (in the form of a Python function).

    Parameters
    ----------
    args : QuantumVariable | QuantumArray | Sequence[QuantumVariable | QuantumArray]
        The quantum variable, array, or collection thereof that defines the search space
        for Grover's algorithm.
    oracle_function : Callable
        The quantum oracle function. This callable must accept ``args`` as its argument
        and apply a phase flip to tag the target winner state(s).
        Must uncompute any auxiliary QuantumVariables it uses.
        If ``exact`` is set to True, the oracle function must also support the keyword argument
        ``phase``, which specifies how much the winner states are phase-shifted
        (in standard Grover, this would be $\pi$).
    kwargs : dict, optional
        A dictionary containing keyword arguments to be passed to the oracle. The default is None.
    iterations : int, optional
        The exact amount of Grover iterations to perform.
    winner_state_amount : int, optional
        If ``iterations`` is not specified, the optimal number of iterations is calculated
        using the established mathematical formula based on the number of winner states.
        The default assumption (if omitted) is 1.
    exact : bool, optional
        If set to True, the `exact version <https://arxiv.org/pdf/quant-ph/0106071.pdf>`_
        of Grover's algorithm will be evaluated. For this, the correct
        ``winner_state_amount`` must be supplied, and the oracle must support the
        keyword argument ``phase`` to apply the calculated fractional phase shift.

    Raises
    ------
    ValueError
        If ``exact`` is set to True but ``winner_state_amount`` is not specified.
    ZeroDivisionError
        If ``winner_state_amount`` is 0.

    Examples
    --------

    We construct an oracle that tags the states -3 and 2 on two QuantumFloats and apply
    Grover's algorithm.

    ::

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
    tagged by the oracle have zero percent measurement probability.

    """

    # Necessary to prevent errors in recursive_qs_search when applied to jax arrays.
    if check_for_tracing_mode():
        import jax.numpy as jnp
    else:
        import numpy as jnp

    if kwargs is None:
        kwargs = {}

    if exact and winner_state_amount is None:
        raise ValueError(
            "Exact Grover's algorithm requires 'winner_state_amount' to be specified."
        )
    elif winner_state_amount is None:
        winner_state_amount = 1

    if isinstance(args, Sequence):
        flat_qvs: list[QuantumVariable] = []
        for arg in args:
            if isinstance(arg, QuantumArray):
                flat_qvs.extend(list(arg.flatten()))
            else:
                flat_qvs.append(arg)
        N = 2 ** jnp.sum(jnp.array([qv.size for qv in flat_qvs]))
    elif isinstance(args, QuantumArray):
        N = 2 ** jnp.sum(jnp.array([qv.size for qv in args.flatten()]))
    elif isinstance(args, QuantumVariable):
        N = 2**args.size
    else:
        raise TypeError(f"Unsupported type for args: {type(args)}")

    if exact:
        # Implementation for phase calculation for exact grovers alg as in
        # https://arxiv.org/pdf/quant-ph/0106071.pdf

        theta = jnp.arcsin(jnp.sqrt(winner_state_amount / N))

        iterations = jnp.int64(jnp.ceil(jnp.pi / (4 * theta) - 0.5))

        phi = 2 * jnp.arcsin(
            jnp.sin(jnp.pi / (4 * (iterations - 1) + 6))
            * (N / winner_state_amount) ** 0.5
        )

    elif iterations is None:
        iterations = jnp.pi / 4 * jnp.sqrt(N / winner_state_amount)
        iterations = jnp.int64(jnp.round(iterations))

    if isinstance(args, Sequence):
        for qv in args:
            h(qv)
    else:
        h(args)

    if check_for_tracing_mode():
        for _ in jrange(iterations):
            if exact:
                oracle_function(args, phase=phi, **kwargs)
                diffuser(args, phase=phi)
            else:
                oracle_function(args, **kwargs)
                diffuser(args)

    elif iterations > 0:
        merge(args)
        qs = recursive_qs_search(args)[0]
        # qv_amount = len(qs.qv_list)

        with IterationEnvironment(qs, iterations):
            if exact:
                oracle_function(args, phase=phi, **kwargs)
                diffuser(args, phase=phi)
            else:
                oracle_function(args, **kwargs)
                diffuser(args)

        # NOTE: We could check here whether the oracle introduced new QuantumVariables without uncomputing/deleting them, which would be a common mistake.
        # This check was deactivated, be cause it raises an unjustified error in some cases, e.g., when the oracle acts on a QuantumVariable that is not part of the input `args`. See #586.
        # if qv_amount != len(qs.qv_list):
        #    raise Exception(
        #        "Applied oracle introducing new QuantumVariables without uncomputing/deleting"
        #    )


# Workaround to keep the docstring but still gatewrap

temp = diffuser.__doc__

diffuser = gate_wrap(permeability=[], is_qfree=False)(diffuser)

diffuser.__doc__ = temp


temp = tag_state.__doc__

tag_state = gate_wrap(permeability="args", is_qfree=True)(tag_state)

tag_state.__doc__ = temp
