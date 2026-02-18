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

from qrisp.core import cx, swap
from qrisp.environments import control
from qrisp.jasp import check_for_tracing_mode, jlen, jrange, while_loop

def demux(
    input,
    ctrl_qv,
    output=None,
    ctrl_method=None,
    permit_mismatching_size=False,
    parallelize_qc=False,
):
    """
    This functions allows moving an input value into an iterable output, where the
    position is specified by a ``QuantumFloat``. Demux is short for demultiplexer and
    is a standard component in `classical electrical circuitry
    <https://en.wikipedia.org/wiki/Multiplexer>`_.

    Demux can either move qubit states into a QuantumVariable or ``QuantumVariables``
    into ``QuantumArrays``.

    This function can also be used to "in-place demux" the 0-th entry of an iterable to
    the position specified by ``ctrl_qv``. For more information on this, check the
    second example.

    Parameters
    ----------
    input : Qubit or QuantumVariable
        The input value that is supposed to be moved.
    ctrl_qv : QuantumFloat
        The QuantumFloat specifying to which output the input should be moved.
    output : QuantumVariable or QuantumArray, optional
        The output object, where the input should end up. By default, a new object
        (QuantumVariable or QuantumArray) is created. Note that when this parameter is
        given, it is guaranteed, that the 0-th entry will be moved to the desired
        position, the other entries can also be permuted away from their original
        position.
    ctrl_method : string, optional
        The ``ctrl_method`` string passed to the
        :ref:`control environment <ControlEnvironment>` to generate controlled swaps.
    permit_mismatching_size : bool, optional
        If set to False, an exception will be raised, if the state-space dimension of
        `ctrl_qv`` is differing from the amount of outputs. The default is False.
    parallelize_qc : bool, optional
        If set to True, this option reduces (de)allocates additional qubits to
        reduce the depth. The default is False.

    Raises
    ------
    Exception
        Tried to demux with mismatchingly sized control input.

    Returns
    -------
    output : QuantumVariable or QuantumArray
        The output object with the input signal placed at the index specified by
        ``ctrl_qv``.

    Examples
    --------

    We create a ``QuantumBool`` and demux it into a ``QuantumArray`` ::

        from qrisp import *

        qb = QuantumBool()
        qb.flip()

        index = QuantumFloat(2)

        h(index[1])

        res_array = demux(qb, index)

    >>> print(multi_measurement([index, res_array]))
    {(0, OutcomeArray([1., 0., 0., 0.])): 0.5, (2, OutcomeArray([0., 0., 1., 0.])): 0.5}

    Demux can also be used to move the 0-th entry of a ``QuantumArray`` in-place. ::

        qa = QuantumArray(shape = 4, qtype = qb)

        qa[0].flip()

        demux(qa[0], index, qa)

    >>> print(multi_measurement([index, qa]))
    {(0, OutcomeArray([1., 0., 0., 0.])): 0.5, (2, OutcomeArray([0., 0., 1., 0.])): 0.5}

    For low-level manipulations, demux can move information within ``QuantumVariables``.
    ::

        qf = QuantumVariable(4)

        qf[:] = "1000"

        demux(qf[0], index, qf)

    >>> print(multi_measurement([index, qf]))
    {(0, '1000'): 0.5, (2, '0010'): 0.5}
    """

    from qrisp import QuantumArray, QuantumVariable, Qubit, control, swap

    if output is None:
        if isinstance(input, QuantumVariable):
            output = QuantumArray(input, 2 ** len(ctrl_qv))
        elif isinstance(input, Qubit):
            output = QuantumVariable(2 ** len(ctrl_qv))
        else:
            raise Exception("Don't know how to handle input type " + str(type(input)))
    else:
        if isinstance(output, QuantumArray):
            for qv in output.flatten()[1:]:
                if qv.name == input.name:
                    raise Exception(
                        "Tried to in-place demux QuantumArray entry,"
                        "which is not a 0-th position"
                    )
        elif isinstance(output, QuantumVariable):
            for qb in output.reg[1:]:
                if qb.identifier == input.identifier:
                    raise Exception(
                        "Tried to in-place demux QuantumVariable entry,"
                        "which is not a 0-th position"
                    )

    n = int(np.ceil(np.log2(len(output))))
    N = 2**n

    if len(output) != 2 ** len(ctrl_qv) and not permit_mismatching_size:
        raise Exception("Tried to demux with mismatching sized control input")

    if hash(input) != hash(output[0]):
        swap(input, output[0])

    if not len(ctrl_qv):
        return output

    if len(output) > 2 ** (len(ctrl_qv) - 1):
        with control(ctrl_qv[-1], ctrl_method=ctrl_method):
            swap(output[0], output[N // 2])
    else:
        demux(
            output[0],
            ctrl_qv[:-1],
            output,
            ctrl_method=ctrl_method,
            permit_mismatching_size=permit_mismatching_size,
        )
        return output

    if n > 1:

        if parallelize_qc:
            demux_ancilla = QuantumVariable(len(ctrl_qv) - 1)
            cx(ctrl_qv[:-1], demux_ancilla)
            ctrl_qubits = list(demux_ancilla)
        else:
            ctrl_qubits = ctrl_qv[:-1]

        demux(
            output[0],
            ctrl_qubits,
            # ctrl_qv[:-1],
            output[: N // 2],
            ctrl_method=ctrl_method,
            permit_mismatching_size=permit_mismatching_size,
            parallelize_qc=parallelize_qc,
        )

        demux(
            output[N // 2],
            ctrl_qv[:-1],
            output[N // 2 :],
            ctrl_method=ctrl_method,
            permit_mismatching_size=permit_mismatching_size,
            parallelize_qc=parallelize_qc,
        )
        if parallelize_qc:
            cx(ctrl_qv[:-1], demux_ancilla)
            demux_ancilla.delete()

    return output


def q_indexing(q_array, index):
    from qrisp import invert

    with invert():
        demux(q_array[0], index, q_array, ctrl_method="gray_pt")

    res = q_array[0].duplicate(init=True)

    demux(q_array[0], index, q_array, ctrl_method="gray_pt")

    return res


def q_swap_into(q_array, index, qv):
    from qrisp import invert, swap

    with invert():
        demux(q_array[0], index, q_array, ctrl_method="gray_pt")

    swap(q_array[0], qv)

    demux(q_array[0], index, q_array, ctrl_method="gray_pt")


def cyclic_shift(iterable, shift_amount=1):
    r"""
    Performs a cyclic shift of the values of an iterable with logarithmic depth.
    The shifting amount can be specified.


    Parameters
    ----------
    iterable : list[Qubit] or list[QuantumVariable] or QuantumArray
        The iterable to be shifted.
    shift_amount : integer or QuantumFloat, optional
        The iterable will be shifted by that amount. The default is 1.

    Examples
    --------

    We create a QuantumArray, initiate a sequence of increments and perform a cyclic shift.

    >>> from qrisp import QuantumFloat, QuantumArray, cyclic_shift
    >>> import numpy as np
    >>> qa = QuantumArray(QuantumFloat(3), 8)
    >>> qa[:] = np.arange(8)
    >>> cyclic_shift(qa, shift_amount = 2)
    >>> print(qa)
    {OutcomeArray([6, 7, 0, 1, 2, 3, 4, 5]): 1.0}

    We do something similar to demonstrate the shift by quantum values.
    For this we initiate a :ref:`QuantumFloat` in the superposition of 0, 1 and -3.

    >>> shift_amount = QuantumFloat(3, signed = True)
    >>> shift_amount[:] = {0 : 3**-0.5, 1: 3**-0.5, -3 : 3**-0.5}
    >>> qa = QuantumArray(QuantumFloat(3), 8)
    >>> qa[:] = np.arange(8)
    >>> cyclic_shift(qa, shift_amount)
    >>> print(qa)
    {OutcomeArray([0, 1, 2, 3, 4, 5, 6, 7]): 0.3333, OutcomeArray([7, 0, 1, 2, 3, 4, 5, 6]): 0.3333, OutcomeArray([3, 4, 5, 6, 7, 0, 1, 2]): 0.3333}
    """

    from qrisp import QuantumFloat, control, QuantumBool, cx

    if isinstance(shift_amount, QuantumFloat):

        if shift_amount.mshape[0] < 0:
            raise Exception("Tried to quantum shift by non-integer QuantumFloat")

        if shift_amount.signed:
            with control(shift_amount.sign()):
                cyclic_shift(iterable, -(2 ** (shift_amount.mshape[1])))

        for i in range(*shift_amount.mshape):
            with control(shift_amount.significant(i)):
                cyclic_shift(iterable, 2**i)

        return

    if check_for_tracing_mode():
        _cyclic_shift_jasp(iterable, shift_amount)
    else:
        N = len(iterable)
        n = int(np.floor(np.log2(N)))

        if N == 0 or not shift_amount % N:
            return
        if shift_amount < 0:
            return cyclic_shift(iterable[::-1], -shift_amount)

        if shift_amount != 1:

            perm = np.arange(N)
            perm = (perm - shift_amount) % (N)

            permute_iterable(iterable, perm)
            return

        singular_shift(iterable[: 2**n])
        singular_shift([iterable[0]] + list(iterable[2**n :]), use_saeedi=True)

def _cyclic_shift_jasp(iterable, shift_amount):
    """
    JASP-compatible implementation of cyclic_shift for arbitrary integer
    shift_amount. Decomposes into repeated applications of shift-by-1 using
    singular_shift with DynamicQubitArray slicing/fusion.

    For negative shift_amount, the invert() environment is used to reverse the
    direction (inverse of shift-right = shift-left).

    Parameters
    ----------
    iterable : QuantumVariable or similar
        The iterable to shift (must have a .reg attribute supporting slicing).
    shift_amount : int
        The (classical) shift amount.
    """
    if shift_amount == 0:
        return

    if shift_amount < 0:
        from qrisp import invert
        with invert():
            _cyclic_shift_jasp(iterable, -shift_amount)
        return

    for _ in range(shift_amount):
        _cyclic_shift_one(iterable)


def _cyclic_shift_one(iterable):
    """
    JASP-compatible single cyclic shift (shift_amount=1).
    Decomposes into two singular_shift calls using DynamicQubitArray
    slicing and fusion, mirroring the non-JASP decomposition:
        singular_shift(iterable[:2**n])
        singular_shift([iterable[0]] + iterable[2**n:], use_saeedi=True)
    """
    N = jlen(iterable)

    n = compute_floor_log2(N)
    pow2n = 2**n

    # First singular_shift on iterable[:2**n]
    first_part = iterable.reg[:pow2n]
    singular_shift(first_part)

    # Second singular_shift on [iterable[0]] + iterable[2**n:]
    # Construct via DynamicQubitArray fusion: fuse first qubit with tail
    first_qubit = iterable.reg[:1]
    tail = iterable.reg[pow2n:]
    combined = first_qubit + tail  # DynamicQubitArray.__add__ → fuse_qb_array
    singular_shift(combined, use_saeedi=True)


def compute_floor_log2(N):
    """
    Computes floor(log2(N)) in a JASP-compatible way using while_loop.
    """
    def body_fun(val):
        result, current = val
        current = current // 2
        result += 1
        return result, current

    def cond_fun(val):
        return val[1] > 1

    result, _ = while_loop(cond_fun, body_fun, (0, N))
    return result


def compute_ladder_iterations(N):
    """
    Computes the number of ladder iterations (swap levels) required by
    ``singular_shift`` for an iterable of length ``N``.

    The ladder algorithm swaps at distances 1, 2, 4, …, 2^(k-1).
    To reach every element from position 0 we need 2^(k-1) >= N-1,
    i.e. k = ceil(log2(N)) iterations.

    Args:
        N (int): The length of the iterable.
    Returns:
        int: The number of iterations (ceil(log2(N))).
    """

    power = 1
    iterations = 0

    def body_fun(val):
        power, iterations = val
        return (power * 2, iterations + 1)

    def cond_fun(val):
        return val[0] < N

    power, iterations = while_loop(cond_fun, body_fun, (power, iterations))
    return iterations

def singular_shift(iterable, use_saeedi=False):

    N = jlen(iterable)

    if use_saeedi:
        for i in jrange(N // 2):
            j = (N - i) % N   # equivalent to -i % N

            expr_out = j != i + 1
            expr_in =  i + 1 < N
            with control(expr_out):
                with control(expr_in):
                    swap(iterable[j], iterable[i + 1])

        for i in jrange(N // 2):
            j = (N - i) % N   # equivalent to -i % N
            
            expr_out = j != i + 2
            expr_in = i + 2 < N
            with control(expr_out):
                with control(expr_in):
                    swap(iterable[j], iterable[i + 2])

    else:

        iterations = compute_ladder_iterations(N)
        for j in jrange(iterations):
            step = 2 * 2**j
            max_i = N - 2**j
            for i in jrange(max_i):
                with control((i % step) == 0):
                    left = i
                    right = i + 2**j
                    with control(right < N):
                        swap(iterable[left], iterable[right])


def to_cycles(perm):
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi))  # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles


def permute_iterable(iterable, perm):
    """
    Applies an arbitrary permutation to an iterable with logarithmic depth.

    Parameters
    ----------
    iterable : QuantumArray or List[QuantumVariable] or QuantumVariable or List[Qubit]
        The iterable to perform the permutation on.
    perm : List[integer]
        A list specifying the permutation.

    Examples
    --------

    We create a QuantumArray containing increments and apply a specified permutation.

    >>> from qrisp import QuantumFloat, QuantumArray, permute_iterable
    >>> import numpy as np
    >>> qa = QuantumArray(QuantumFloat(3), 8)
    >>> qa[:] = np.arange(8)
    >>> permute_iterable(qa, perm = [1,0,3,7,5,2,6,4])
    >>> print(qa)
    {OutcomeArray([1, 0, 3, 7, 5, 2, 6, 4]): 1.0}
    >>> print(qa.qs)

    .. code-block:: none

        QuantumCircuit:
        --------------
          qa.0: ────────────X──────────────────────
                            │
          qa.1: ──────X─────┼──────────────────────
                      │     │
          qa.2: ──────┼──X──┼──────────────────────
                ┌───┐ │  │  │
        qa_1.0: ┤ X ├─┼──┼──X──────────────────────
                └───┘ │  │
        qa_1.1: ──────X──┼─────────────────────────
                         │
        qa_1.2: ─────────X─────────────────────────

        qa_2.0: ───────────────────────────X───────
                ┌───┐                      │
        qa_2.1: ┤ X ├──────────────────────┼──X────
                └───┘                      │  │
        qa_2.2: ───────────────────────────┼──┼──X─
                ┌───┐                      │  │  │
        qa_3.0: ┤ X ├──────────X───────────┼──┼──┼─
                ├───┤          │           │  │  │
        qa_3.1: ┤ X ├──────────┼──X────────┼──┼──┼─
                └───┘          │  │        │  │  │
        qa_3.2: ───────────────┼──┼──X─────┼──┼──┼─
                               │  │  │     │  │  │
        qa_4.0: ──────X────────┼──┼──┼─────┼──┼──┼─
                      │        │  │  │     │  │  │
        qa_4.1: ──────┼──X─────┼──┼──┼─────┼──┼──┼─
                ┌───┐ │  │     │  │  │     │  │  │
        qa_4.2: ┤ X ├─┼──┼──X──┼──┼──┼─────┼──┼──┼─
                ├───┤ │  │  │  │  │  │     │  │  │
        qa_5.0: ┤ X ├─X──┼──┼──┼──┼──┼──X──X──┼──┼─
                └───┘    │  │  │  │  │  │     │  │
        qa_5.1: ─────────X──┼──┼──┼──┼──┼──X──X──┼─
                ┌───┐       │  │  │  │  │  │     │
        qa_5.2: ┤ X ├───────X──┼──┼──┼──┼──┼──X──X─
                └───┘          │  │  │  │  │  │
        qa_6.0: ───────────────┼──┼──┼──┼──┼──┼────
                ┌───┐          │  │  │  │  │  │
        qa_6.1: ┤ X ├──────────┼──┼──┼──┼──┼──┼────
                ├───┤          │  │  │  │  │  │
        qa_6.2: ┤ X ├──────────┼──┼──┼──┼──┼──┼────
                ├───┤          │  │  │  │  │  │
        qa_7.0: ┤ X ├──────────X──┼──┼──X──┼──┼────
                ├───┤             │  │     │  │
        qa_7.1: ┤ X ├─────────────X──┼─────X──┼────
                ├───┤                │        │
        qa_7.2: ┤ X ├────────────────X────────X────
                └───┘
        Live QuantumVariables:
        ---------------------
        QuantumFloat qa
        QuantumFloat qa_1
        QuantumFloat qa_2
        QuantumFloat qa_3
        QuantumFloat qa_4
        QuantumFloat qa_5
        QuantumFloat qa_6
        QuantumFloat qa_7

    """

    from sympy.combinatorics import Permutation

    inv_perm = list(Permutation(perm) ** -1)

    cycles = to_cycles(inv_perm)

    for c in cycles:
        cyclic_shift([iterable[i] for i in c], 1)
