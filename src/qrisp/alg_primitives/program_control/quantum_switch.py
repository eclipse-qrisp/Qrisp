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

"""Implements the classical-mode backend for q_switch, dispatching branches by a quantum index."""

import warnings
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from qrisp.alg_primitives import demux
from qrisp.circuit import Qubit
from qrisp.core import QuantumArray, QuantumVariable, cx, mcx, x
from qrisp.environments import (
    conjugate,
    control,
    custom_control,
    custom_inversion,
    invert,
)
from qrisp.jasp import check_for_tracing_mode, jrange, q_cond, q_fori_loop
from qrisp.qtypes import QuantumBool

# A branch_amount may be a traced value: q_switch supports a caller passing one
# that is only known at run time, in which case the tree method routes the
# predicates that depend on it through q_cond.
_BranchAmount = int | jax.core.Tracer
_Branches = list[Callable] | Callable
_Index = QuantumVariable | list[Qubit]
_Range = Callable[..., Iterable[int]]


def _invert_inpl_function(func):
    """Helper function to invert in-place functions."""

    def inverted_func(*args, **kwargs):
        with invert():
            return func(*args, **kwargs)

    return inverted_func


def _normalize_branches(
    index: _Index,
    branches: _Branches,
    branch_amount: _BranchAmount | None,
    method: str,
    inv: bool,
) -> tuple[_Branches, _BranchAmount, bool, _Range]:
    """Resolve the branch representation shared by every compile method.

    Returns the branches (inverted if requested), the number of branches, whether
    ``branches`` is a function rather than a list, and the range helper to iterate
    branch indices with. A function is enumerated over the full index range unless
    the caller caps it with ``branch_amount``; a list carries its own length.

    Raises
    ------
    TypeError
        If ``branches`` is neither a list nor a callable, or if ``branch_amount``
        is combined with a list under the ``"sequential"`` method.

    """
    if is_function_mode := callable(branches):
        if branch_amount is None:
            index_size = len(index) if isinstance(index, list) else index.size
            branch_amount = 2**index_size
        xrange = jrange if check_for_tracing_mode() else range
        if inv:
            branches = _invert_inpl_function(branches)

    elif isinstance(branches, list):
        if branch_amount is None:
            branch_amount = len(branches)
        elif method == "sequential":
            raise TypeError(
                "Argument 'branch_amount' must be None when using the 'sequential' method and a list as a 'branches'"
            )
        if inv:
            branches = [_invert_inpl_function(func) for func in branches]

        xrange = range

    else:
        raise TypeError("Argument 'branches' must be a list or a callable(i, *operands)")

    return branches, branch_amount, is_function_mode, xrange


def _q_switch_sequential(
    index: _Index,
    branches: _Branches,
    operands: tuple[QuantumVariable, ...],
    branch_amount: _BranchAmount,
    is_function_mode: bool,
    xrange: _Range,
    ctrl: Qubit | None,
) -> None:
    """Apply each branch under its own comparison against the index.

    Costs one comparison per branch, so the circuit grows linearly, but it traces
    each branch exactly once.
    """
    control_qbl = QuantumBool()

    for i in xrange(branch_amount):
        with conjugate(mcx)(index, control_qbl, ctrl_state=i):
            with control(control_qbl):
                if ctrl is None:
                    if is_function_mode:
                        branches(i, *operands)
                    else:
                        branches[i](*operands)
                elif is_function_mode:
                    with control(ctrl):
                        branches(i, *operands)
                else:
                    with control(ctrl):
                        branches[i](*operands)

    control_qbl.delete()


def _q_switch_parallel(
    index: _Index,
    branches: _Branches,
    operands: tuple[QuantumVariable, ...],
    branch_amount: _BranchAmount,
    is_function_mode: bool,
    ctrl: Qubit | None,
) -> None:
    """Apply every branch to its own copy of the operand, demultiplexed by the index.

    Exponentially faster than the alternatives at the cost of one operand copy per
    branch. Not available in tracing mode.
    """
    if check_for_tracing_mode():
        raise NotImplementedError("Compile method 'parallel' for switch-case structure not available in tracing mode.")

    if isinstance(index, list):
        raise NotImplementedError(
            "Compile method 'parallel' for switch-case structure not available when 'index' is a list of qubits."
        )

    if len(operands) > 1:
        raise NotImplementedError(
            "Compile method 'parallel' for switch-case structure not available when more then one 'operands' are provided."
        )

    # Idea: Use demux function to move operand and enabling bool into QuantumArray
    # to execute the branches in parallel.

    # This QuantumArray acts as an addressable QRAM via the demux function

    if branch_amount != 2**index.size:
        warnings.warn("Warning: Additional qubit overhead because branch amount is smaller than index QuantumVariable!")

    enable = QuantumArray(qtype=QuantumBool(), shape=(2**index.size,))
    enable[0].flip()

    qa = QuantumArray(qtype=operands[0], shape=((2**index.size,)))

    with conjugate(demux)(operands[0], index, qa, parallelize_qc=True):
        with conjugate(demux)(enable[0], index, enable, parallelize_qc=True):
            for i in range(branch_amount):
                with control(enable[i]):
                    if ctrl is None:
                        if is_function_mode:
                            branches(i, qa[i])
                        else:
                            branches[i](qa[i])
                    elif is_function_mode:
                        with control(ctrl):
                            branches(i, qa[i])
                    else:
                        with control(ctrl):
                            branches[i](qa[i])

    qa.delete()

    enable[0].flip()
    enable.delete()


def _q_switch_tree(
    index: _Index,
    branches: _Branches,
    operands: tuple[QuantumVariable, ...],
    branch_amount: _BranchAmount,
    is_function_mode: bool,
    ctrl: Qubit | None,
) -> None:
    """Apply the branches by walking a balanced binary tree over the index.

    Uses balanced binary trees, https://arxiv.org/pdf/2407.17966v1. An ancilla per
    index qubit tracks the path to the current leaf, and the walk moves between
    adjacent leaves rather than re-deriving each path from scratch, which keeps the
    gate count far below the linear scan of the sequential method.
    """
    # Jasp mode
    #
    # The tree walk is driven by the index width n = index.size, and in Jasp a
    # QuantumVariable's size is always a tracer -- even for a literally sized
    # QuantumFloat(3). The loops over n therefore have to be jrange, and the
    # depths they derive are traced values. Replacing them with a plain range
    # raises TracerIntegerConversionError.
    #
    # Many predicates are nonetheless plain Python values: the depth guards in
    # the statically unrolled parts of the walk, the leaf selection when the
    # index is known, and branch_amount whenever the caller passed an int. For
    # those, q_cond would trace both arms and discard one. x_cond below picks
    # the arm directly instead, which is what the non-traced definition further
    # down already does, and falls back to q_cond for genuinely traced
    # predicates -- including a traced branch_amount.
    if check_for_tracing_mode():
        xrange = jrange
        x_fori_loop = q_fori_loop

        def x_cond(pred, true_fun, false_fun, *operands):
            if isinstance(pred, jax.core.Tracer):
                return q_cond(pred, true_fun, false_fun, *operands)
            return true_fun(*operands) if pred else false_fun(*operands)

        def bitwise_count_diff(a, b):
            return jnp.int32(jnp.bitwise_count(jnp.bitwise_xor(a, b)))

    # Normal mode
    else:
        xrange = range

        def x_fori_loop(lower, upper, body_fun, init_val):
            val = init_val
            for i in range(lower, upper):
                val = body_fun(i, val)
            return val

        def x_cond(pred, true_fun, false_fun, *operands):
            if pred:
                return true_fun(*operands)
            return false_fun(*operands)

        def bitwise_count_diff(a, b):
            return np.int32(np.bitwise_count(np.bitwise_xor(a, b)))

    n = len(index) if isinstance(index, list) else index.size

    # The gate calls below are wrapped so that the lambdas handed to x_cond
    # never close over a name that a surrounding scope rebinds.
    def nor_x(t):
        x(t)

    def nor_cx(c, t):
        cx(c, t)

    def nor_mcx(c, t):
        mcx(c, t)

    def toggle_from_parent(anc, target, parent):
        """Flip ``anc[target]`` conditioned on its parent node ``anc[parent]``.

        ``parent == -1`` means ``target`` is the root of the tree and has no
        parent, so the flip is unconditional -- or conditioned on ``ctrl``
        alone when the whole switch is controlled.
        """
        if ctrl is None:

            def at_root():
                nor_x(anc[target])

        else:

            def at_root():
                nor_cx(ctrl, anc[target])

        x_cond(parent == -1, at_root, lambda: nor_cx(anc[parent], anc[target]))

    def toggle_from_parent_and_index(anc, index_qubit, target, parent):
        """Flip ``anc[target]`` conditioned on ``anc[parent]`` and ``index_qubit``.

        As above, ``parent == -1`` means there is no parent node to condition on.
        """
        if ctrl is None:

            def at_root():
                nor_cx(index_qubit, anc[target])

        else:

            def at_root():
                nor_mcx([index_qubit, ctrl], anc[target])

        x_cond(
            parent == -1,
            at_root,
            lambda: nor_mcx([index_qubit, anc[parent]], anc[target]),
        )

    def bounce(d: int, anc, ca, oper):
        # Cross to the sibling subtree: retarget the node one level up, then
        # descend again on the index qubit for this depth.
        toggle_from_parent(anc, d - 1, d - 2)

        with control(anc[d - 1]):
            x(anc[d])

        toggle_from_parent_and_index(anc, ca[n - 1 - d], d, d - 2)

    def down(d: int, anc, ca, oper):
        # Descend into the zero child: the surrounding x gates flip the index
        # qubit so that the toggle triggers on it being 0 rather than 1.
        x(ca[n - 1 - d])
        toggle_from_parent_and_index(anc, ca[n - 1 - d], d, d - 1)
        x(ca[n - 1 - d])

    def up(d: int, anc, ca, oper):
        # Ascend out of the one child; the inverse of the toggle in down,
        # without the surrounding index flips.
        toggle_from_parent_and_index(anc, ca[n - 1 - d], d, d - 1)

    # Function mode
    if is_function_mode:

        def leaf(d: int, anc, ca, i, oper):
            with control(anc[d]):
                branches(i, *oper)

            # Move from the even leaf to its odd sibling.
            toggle_from_parent(anc, d, d - 1)

            with control(anc[d]):
                branches(i + 1, *oper)

        def last_leaf(d: int, anc, ca, i, oper):
            with control(anc[d]):
                branches(i, *oper)

    # List mode
    elif isinstance(branches, list):
        if len(branches) % 2 != 0:
            # The tree walks leaves in pairs, so an odd list gets one padding
            # branch. It takes *args because branches are invoked with every
            # operand, and it is appended to a copy because `branches` belongs
            # to the caller.
            def identity(*args):
                pass

            branches = [*branches, identity]

        if check_for_tracing_mode():

            def leaf(d: int, anc, ca, i, oper):
                def apply_leaf(A, B):
                    with control(anc[d]):
                        A(*oper)

                    # Move from the even leaf to its odd sibling.
                    toggle_from_parent(anc, d, d - 1)

                    with control(anc[d]):
                        B(*oper)

                for j in range(0, len(branches), 2):
                    x_cond(
                        j == i,
                        apply_leaf,
                        lambda a, b: None,
                        branches[j],
                        branches[j + 1],
                    )

        else:

            def leaf(d: int, anc, ca, i, oper):
                with control(anc[d]):
                    branches[i](*oper)

                # Move from the even leaf to its odd sibling.
                toggle_from_parent(anc, d, d - 1)

                with control(anc[d]):
                    branches[i + 1](*oper)

        def last_leaf(d: int, anc, ca, i, oper):
            def apply(f):
                with control(anc[d]):
                    f(*oper)

            for j in range(0, len(branches)):
                x_cond(j == i, apply, lambda x: None, branches[j])

    else:
        raise TypeError("Argument 'branches' must be a list or a callable(i, *operands)")

    def body_fun(pos, val):
        anc, ca, oper = val

        # Apply leaf
        leaf(n - 1, anc, ca, 2 * pos, oper)

        # Jump to next leaf
        q = bitwise_count_diff(pos, pos + 1)
        for j in xrange(0, q - 1):
            up(n - j - 1, anc, ca, oper)
        bounce(n - q, anc, ca, oper)
        for j in xrange(0, q - 1):
            down(n - (q - 1) + j, anc, ca, oper)

        return anc, ca, oper

    # One ancilla per index qubit, tracking the path to the current leaf.
    anc = QuantumVariable(n)

    # Descend to the first leaf
    for j in xrange(0, n):
        down(j, anc, index, operands)

    # Walk the leaves, jumping from each to the next
    _, _, _ = x_fori_loop(0, -(-branch_amount // 2) - 1, body_fun, (anc, index, operands))

    # Perform the last leaf
    x_cond(
        branch_amount % 2 == 0,
        lambda: leaf(n - 1, anc, index, branch_amount - 2, operands),
        lambda: last_leaf(n - 1, anc, index, branch_amount - 1, operands),
    )

    # Go back from last node
    diff = 2**n - branch_amount
    for j in xrange(0, n):
        up(n - j - 1, anc, index, operands)

        def bf():
            toggle_from_parent(anc, n - j - 1, n - j - 2)

        # The walk stopped short of the full 2**n leaves, so the levels where
        # that shortfall has a set bit need one extra retarget on the way out.
        x_cond((diff >> j) & 1, lambda: bf(), lambda: None)

    anc.delete()


# Switch implementation for quantum index
def _q_switch_q(index, branches, *operands, branch_amount=None, method="auto", inv=False, ctrl=None):
    r"""Executes a switch - case statement distinguishing between given in-place functions.

    Parameters
    ----------
    index : QuantumVariable or list[:ref:`Qubit`]
        An integer value, deciding which function gets executed.
    branches : list[callable] or callable
        List of functions to be executed based on ``index`` or a single function
        that takes the index as first argument.
    *operands : tuple
        The input values for whichever function is applied.
    branch_amount : int, optional
        The amount of branches.
        Only needed if ``branches`` is a function.
        Is automatically inferred from the length of ``branches`` if it is a list.
    method : str, optional
        The method used to implement the quantum switch. Can be ``"auto"``, ``"sequential"``, ``"parallel"``,
        or ``"tree"``. Default is ``"auto"``.
        Method ``"tree"`` uses `balanced binary trees <https://arxiv.org/pdf/2407.17966v1>`_.
        Method ``"parallel"`` is exponentially faster but requires more qubits.

    Examples
    --------
    We write a script that uses a :ref:`QuantumFloat` as index to select
    different operations on another operand :ref:`QuantumFloat`. The index variable is
    put into superposition such that all branches are executed in superposition.

    ::

        from qrisp import *
        from qrisp.jasp import *

        @terminal_sampling
        def main():

            def f0(x): x += 1
            def f1(x): x += 2
            def f2(x): pass
            def f3(x): h(x[1])
            branches = [f0, f1, f2, f3]

            operand = QuantumFloat(4)
            operand[:] = 1
            index = QuantumFloat(2)
            h(index)

            q_switch(index, branches, operand)
            return index, operand

        print(main())
        # {(0.0, 2.0): 0.25000000372529035, (1.0, 3.0): 0.25000000372529035,
        # (2.0, 1.0): 0.25000000372529035, (3.0, 1.0): 0.12499999441206447,
        # (3.0, 3.0): 0.12499999441206447}

    """
    branches, branch_amount, is_function_mode, xrange = _normalize_branches(index, branches, branch_amount, method, inv)

    method = "tree" if method == "auto" else method

    if method == "sequential":
        _q_switch_sequential(index, branches, operands, branch_amount, is_function_mode, xrange, ctrl)
    elif method == "parallel":
        _q_switch_parallel(index, branches, operands, branch_amount, is_function_mode, ctrl)
    elif method == "tree":
        _q_switch_tree(index, branches, operands, branch_amount, is_function_mode, ctrl)
    else:
        raise Exception(f"Don't know compile method {method} for switch-case structure.")


temp = _q_switch_q.__doc__
_q_switch_q = custom_control(custom_inversion(_q_switch_q))
_q_switch_q.__doc__ = temp
