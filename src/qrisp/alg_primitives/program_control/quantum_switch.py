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
from collections.abc import Callable
from typing import Literal, assert_never, get_args

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
_Method = Literal["auto", "sequential", "parallel", "tree"]
# The methods that have an implementation; "auto" is resolved to one of these before dispatch.
_ResolvedMethod = Literal["sequential", "parallel", "tree"]
# The same methods as a runtime tuple, derived from ``_Method`` so that the two cannot drift apart.
_METHODS: tuple[_Method, ...] = get_args(_Method)


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
    method: _Method,
    inv: bool,
) -> tuple[_Branches, _BranchAmount, bool]:
    """Resolve the branch representation shared by every compile method.

    Returns the branches (inverted if requested), the number of branches, and
    whether ``branches`` is a function rather than a list. A function is
    enumerated over the full index range unless the caller caps it with
    ``branch_amount``; a list carries its own length.

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

    else:
        raise TypeError("Argument 'branches' must be a list or a callable(i, *operands)")

    return branches, branch_amount, is_function_mode


def _q_switch_sequential(  # noqa: PLR0913, PLR0917
    index: _Index,
    branches: _Branches,
    operands: tuple[QuantumVariable, ...],
    branch_amount: _BranchAmount,
    is_function_mode: bool,
    ctrl: Qubit | None,
) -> None:
    """Apply each branch under its own comparison against the index.

    Costs one comparison per branch, so the circuit grows linearly, but it traces
    each branch exactly once.
    """
    # In function mode the branch amount defaults to 2**index.size, and in Jasp a
    # QuantumVariable's size is a tracer, so the loop over the branches has to be
    # a jrange -- which also keeps the branch traced exactly once. A list always
    # carries a plain Python length, so range is enough.
    xrange = jrange if is_function_mode and check_for_tracing_mode() else range

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


def _q_switch_parallel(  # noqa: PLR0913, PLR0917
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
            "Compile method 'parallel' for switch-case structure not available "
            "when more then one 'operands' are provided."
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


def _x_cond(pred, true_fun, false_fun, *args):
    """Branch on ``pred``, tracing through q_cond only when it is a traced value.

    In Jasp mode many predicates of the tree walk are nonetheless plain Python
    values: the depth guards in the statically unrolled parts of the walk, the leaf
    selection when the index is known, and ``branch_amount`` whenever the caller
    passed an int. For those, q_cond would trace both arms and discard one, so the
    arm is picked directly instead. A traced ``branch_amount`` still goes through
    q_cond.
    """
    if isinstance(pred, jax.core.Tracer):
        return q_cond(pred, true_fun, false_fun, *args)
    return true_fun(*args) if pred else false_fun(*args)


def _python_fori_loop(lower, upper, body_fun, init_val):
    """Non-traced counterpart of q_fori_loop."""
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def _bitwise_count_diff(a, b):
    """Count the bits in which ``a`` and ``b`` differ."""
    xp = jnp if check_for_tracing_mode() else np
    return xp.int32(xp.bitwise_count(xp.bitwise_xor(a, b)))


def _padding_branch(*args):
    """No-op branch that pads an odd branch list to the even length the tree walk needs.

    It takes ``*args`` because branches are invoked with every operand.
    """


def _call_list_branches(i, branches, stride, apply):
    """Call ``apply(branches[i], ..., branches[i + stride - 1])`` for a possibly traced ``i``.

    A concrete ``i`` indexes the list directly. A traced one cannot, so every
    candidate position ``j`` is emitted under a q_cond on ``j == i``.
    """
    if not isinstance(i, jax.core.Tracer):
        apply(*(branches[i + k] for k in range(stride)))
        return
    for j in range(0, len(branches), stride):
        q_cond(j == i, apply, lambda *_: None, *branches[j : j + stride])


# The arms handed to q_cond below are full functions rather than lambdas: q_cond
# requires both arms to return the same pytree, and a lambda would return the
# gate's own return value while the other arm returns None.
def _toggle_from_parent(anc, target, parent, ctrl):
    """Flip ``anc[target]`` conditioned on its parent node ``anc[parent]``.

    ``parent == -1`` means ``target`` is the root of the tree and has no parent, so
    the flip is unconditional -- or conditioned on ``ctrl`` alone when the whole
    switch is controlled.
    """

    def at_root():
        if ctrl is None:
            x(anc[target])
        else:
            cx(ctrl, anc[target])

    def below_parent():
        cx(anc[parent], anc[target])

    _x_cond(parent == -1, at_root, below_parent)


def _toggle_from_parent_and_index(anc, index_qubit, target, parent, ctrl):
    """Flip ``anc[target]`` conditioned on ``anc[parent]`` and ``index_qubit``.

    As above, ``parent == -1`` means there is no parent node to condition on.
    """

    def at_root():
        if ctrl is None:
            cx(index_qubit, anc[target])
        else:
            mcx([index_qubit, ctrl], anc[target])

    def below_parent():
        mcx([index_qubit, anc[parent]], anc[target])

    _x_cond(parent == -1, at_root, below_parent)


def _bounce(anc, index, n, d, ctrl):
    """Cross to the sibling subtree at depth ``d`` of a tree over ``n`` index qubits.

    Retargets the node one level up, then descends again on the index qubit for
    this depth.
    """
    _toggle_from_parent(anc, d - 1, d - 2, ctrl)
    with control(anc[d - 1]):
        x(anc[d])
    _toggle_from_parent_and_index(anc, index[n - 1 - d], d, d - 2, ctrl)


def _down(anc, index, n, d, ctrl):
    """Descend into the zero child at depth ``d``.

    The surrounding x gates flip the index qubit so that the toggle triggers on it
    being 0 rather than 1.
    """
    x(index[n - 1 - d])
    _toggle_from_parent_and_index(anc, index[n - 1 - d], d, d - 1, ctrl)
    x(index[n - 1 - d])


def _up(anc, index, n, d, ctrl):
    """Ascend out of the one child at depth ``d``; the inverse of the toggle in _down."""
    _toggle_from_parent_and_index(anc, index[n - 1 - d], d, d - 1, ctrl)


def _apply_leaf_pair(anc, d, first, second, operands, ctrl):  # noqa: PLR0913, PLR0917
    """Apply ``first`` at the even leaf below ``anc[d]``, move to its odd sibling and apply ``second``."""
    with control(anc[d]):
        first(*operands)
    _toggle_from_parent(anc, d, d - 1, ctrl)
    with control(anc[d]):
        second(*operands)


def _apply_last_leaf(anc, d, branch, operands):
    """Apply ``branch`` at the leaf below ``anc[d]`` without moving on to a sibling."""
    with control(anc[d]):
        branch(*operands)


def _q_switch_tree(  # noqa: PLR0913, PLR0917
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
    # The tree walk is driven by the index width n = index.size, and in Jasp a
    # QuantumVariable's size is always a tracer -- even for a literally sized
    # QuantumFloat(3). The loops over n therefore have to be jrange, and the
    # depths they derive are traced values. Replacing them with a plain range
    # raises TracerIntegerConversionError.
    if check_for_tracing_mode():
        xrange, x_fori_loop = jrange, q_fori_loop
    else:
        xrange, x_fori_loop = range, _python_fori_loop

    n = len(index) if isinstance(index, list) else index.size

    if is_function_mode:

        def leaf(anc, i, operands):
            _apply_leaf_pair(
                anc,
                n - 1,
                lambda *ops: branches(i, *ops),
                lambda *ops: branches(i + 1, *ops),
                operands,
                ctrl,
            )

        def last_leaf(anc, i, operands):
            _apply_last_leaf(anc, n - 1, lambda *ops: branches(i, *ops), operands)

    else:
        # The tree walks leaves in pairs, so an odd list gets one padding branch,
        # appended to a copy because ``branches`` belongs to the caller.
        if len(branches) % 2 != 0:
            branches = [*branches, _padding_branch]

        # The leaf depth is computed before the candidate q_conds rather than inside
        # each of their arms.
        def leaf(anc, i, operands):
            d = n - 1
            _call_list_branches(
                i,
                branches,
                2,
                lambda first, second: _apply_leaf_pair(anc, d, first, second, operands, ctrl),
            )

        def last_leaf(anc, i, operands):
            d = n - 1
            _call_list_branches(i, branches, 1, lambda branch: _apply_last_leaf(anc, d, branch, operands))

    def body_fun(pos, val):
        anc, index, operands = val

        leaf(anc, 2 * pos, operands)

        # Jump to the next leaf pair: climb to the lowest common ancestor, cross
        # over, and descend again.
        q = _bitwise_count_diff(pos, pos + 1)
        for j in xrange(0, q - 1):
            _up(anc, index, n, n - j - 1, ctrl)
        _bounce(anc, index, n, n - q, ctrl)
        for j in xrange(0, q - 1):
            _down(anc, index, n, n - (q - 1) + j, ctrl)

        return anc, index, operands

    # One ancilla per index qubit, tracking the path to the current leaf.
    anc = QuantumVariable(n)

    # Descend to the first leaf
    for j in xrange(0, n):
        _down(anc, index, n, j, ctrl)

    # Walk the leaves, jumping from each pair to the next
    x_fori_loop(0, -(-branch_amount // 2) - 1, body_fun, (anc, index, operands))

    # Perform the last leaf
    _x_cond(
        branch_amount % 2 == 0,
        lambda: leaf(anc, branch_amount - 2, operands),
        lambda: last_leaf(anc, branch_amount - 1, operands),
    )

    # Go back from the last leaf. The walk stopped short of the full 2**n leaves,
    # so the levels where that shortfall has a set bit need one extra retarget on
    # the way out.
    diff = 2**n - branch_amount
    for j in xrange(0, n):
        _up(anc, index, n, n - j - 1, ctrl)
        _x_cond((diff >> j) & 1, lambda: _toggle_from_parent(anc, n - j - 1, n - j - 2, ctrl), lambda: None)

    anc.delete()


# Switch implementation for quantum index
def _q_switch_q(  # noqa: PLR0913
    index: _Index,
    branches: _Branches,
    *operands: QuantumVariable,
    branch_amount: _BranchAmount | None = None,
    method: _Method = "auto",
    inv: bool = False,
    ctrl: Qubit | None = None,
) -> None:
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
    inv : bool, optional
        Whether to apply the inverse of every branch. Supplied by the
        :func:`custom_inversion <qrisp.custom_inversion>` wrapper. Default is ``False``.
    ctrl : :ref:`Qubit`, optional
        Qubit the whole switch is conditioned on. Supplied by the
        :func:`custom_control <qrisp.custom_control>` wrapper. Default is ``None``.

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
    # Checked before _normalize_branches, which keys on ``method`` itself and would
    # otherwise skip its own checks for a misspelled method.
    if method not in _METHODS:
        raise ValueError(f"Unknown `method`: {method!r}. Possible methods are: {', '.join(map(repr, _METHODS))}.")

    branches, branch_amount, is_function_mode = _normalize_branches(index, branches, branch_amount, method, inv)

    resolved_method: _ResolvedMethod = "tree" if method == "auto" else method

    if resolved_method == "sequential":
        _q_switch_sequential(index, branches, operands, branch_amount, is_function_mode, ctrl)
    elif resolved_method == "parallel":
        _q_switch_parallel(index, branches, operands, branch_amount, is_function_mode, ctrl)
    elif resolved_method == "tree":
        _q_switch_tree(index, branches, operands, branch_amount, is_function_mode, ctrl)
    else:
        assert_never(resolved_method)


_q_switch_q = custom_control(custom_inversion(_q_switch_q))
