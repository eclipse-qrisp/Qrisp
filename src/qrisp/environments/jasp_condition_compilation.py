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

"""Compiles a ConditionEnvironment in Jasp, including the uncomputation of the condition and its intermediates."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from jax.core import DropVar
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from qrisp.jasp import (
    AbstractQubit,
    AbstractQubitArray,
    Jaspr,
    TracingQuantumSession,
    control_jaspr,
    injection_transform,
    make_jaspr,
)
from qrisp.jasp.interpreter_tools import eval_jaxpr, extract_invalues, insert_call_outvalues
from qrisp.jasp.jasp_expression.control_transform import control_eqn
from qrisp.jasp.jasp_expression.environment_collection import dummy_debug_info
from qrisp.jasp.primitives import delete_qubits_p, get_qubit

# How a condition is compiled
# ===========================
#
# The collected body of a ConditionEnvironment starts with a jit equation that
# evaluates the condition C (see ConditionEnvironment.__enter__), followed by a
# get_qubit equation that extracts the truth value of the resulting QuantumBool.
# The remaining equations form the body U. The environment is replaced by a
# single jit equation that executes
#
#     C_fwd ; U controlled on the truth value ; C_inv ; delete temporaries
#
# The temporaries are the qubit arrays that C allocates and doesn't delete: the
# QuantumBool holding the truth value and intermediate results, such as the
# QuantumBools of the comparisons in (qf < 3) | (qf == 7). The functions that C
# calls and that hide temporaries are inlined first, so that these are visible.
# The other functions are kept, since they may carry a custom inverse.
# C_fwd is C, additionally returning the temporaries it doesn't return. C_inv is
# the inverse of C with the temporaries turned into inputs by injection_transform,
# so that it uncomputes the existing qubits instead of allocating new ones.
#
# Why the temporaries end up in |0>: since C maps computational basis states to
# computational basis states (up to a phase), and U acts on the qubits of C (its
# inputs and the temporaries) only as a control or through phases, U never
# changes the values of the qubits C acted on. C_inv therefore finds them as C
# left them and maps them back.
#
# Operations in U that involve the truth value itself are not controlled on it:
# phases on the truth value are applied as they are, and flips of the truth
# value (QuantumBool.flip) invert the condition for the subsequent operations.
# An odd number of flips is undone before C_inv. A block (a function or
# classical control flow) that involves the truth value is not controlled
# either, so all of its operations must involve the truth value, unless its
# permeability is declared with gate_wrap. Measurements and resets can't be
# controlled and are rejected anywhere in U.
#
# The result carries a custom inverse (C_fwd ; U^dagger ; C_inv ; delete) and a
# custom controlled version (C_fwd ; controlled U ; C_inv ; delete), because
# structural inversion would reverse the allocation inside C_fwd, and plain
# control would control C as well. All three are marked as permeable on the
# inputs of C, which they leave unchanged. An enclosing condition on the same
# inputs can therefore contain them.


class ConditionCompilationError(Exception):
    """Raised when a quantum condition can't be compiled in Jasp."""


def compile_condition(body: Jaspr, name: str, description: str) -> Jaspr:
    """Return the Jaspr that executes the collected body of a ConditionEnvironment.

    Parameters
    ----------
    body : Jaspr
        The collected and flattened body of the environment.
    name : str
        The name of the evaluation of the condition in the Jaspr.
    description : str
        Names the condition in error messages.

    Returns
    -------
    Jaspr
        The Jaspr executing C_fwd, the controlled body, C_inv and the deletion
        of the temporaries, with custom inverse and controlled versions.

    """
    condition_eqn, truth_eqn = body.eqns[0], body.eqns[1]
    condition = _inline_functions(condition_eqn.params["jaxpr"])
    analysis = _AllocationAnalysis(condition)
    temporaries = analysis.temporaries
    _check_temporaries(condition, temporaries, description)

    returned = list(condition.outvars[:-1])
    extra = [temporary for temporary in temporaries if not _contains(returned, temporary)]
    forward = _sub_jaspr(condition, condition.invars, condition.eqns, [*returned, *extra, condition.outvars[-1]])
    inverse = _injected_inverse(condition, temporaries)

    # The body, starting with the extraction of the truth value
    results = [Var(aval=var.aval) if isinstance(var, DropVar) else var for var in condition_eqn.outvars]
    use = _sub_jaspr(body, [*body.invars[:-1], *results], body.eqns[1:], body.outvars)

    # Track the qubits that the body receives, including the inputs of the
    # condition and the temporaries it returns, through the body
    inputs = [var for var in condition_eqn.invars[:-1] if isinstance(var, Var) and _is_quantum(var)]
    input_roots = [frozenset([var]) if _is_quantum(var) else frozenset() for var in body.invars[:-1]]
    result_roots = [analysis.roots_of(var) & set(temporaries) for var in returned]
    use_analysis = _AllocationAnalysis(use, input_roots + result_roots + [frozenset()])
    _check_body(use, use_analysis, temporaries, inputs, description)

    truth_qubit = truth_eqn.outvars[0]
    protected = _Protected(use_analysis.roots_of(truth_qubit), frozenset(temporaries), frozenset(inputs))
    controlled_use, flips = _control_on_truth_value(use, use_analysis, truth_qubit, protected, description)
    truth_position = len(body.invars) - 1 + condition_eqn.outvars.index(truth_eqn.invars[0])
    use_jaspr = _undo_flips(controlled_use, flips, truth_position)

    def condition_with(use_jaspr: Jaspr, ctrl: tuple[AbstractQubit, ...] = ()) -> Jaspr:
        """Return the Jaspr of C_fwd ; use_jaspr ; C_inv ; delete temporaries, controlled on ``ctrl``."""

        def conditioned(*args: Any) -> Any:
            """Execute C_fwd, the body, C_inv and delete the temporaries."""
            ctrl_args, args = args[: len(ctrl)], args[len(ctrl) :]
            values = dict(zip(body.constvars, body.consts))
            values.update(zip(body.invars[:-1], args))
            condition_args = [values[var] if isinstance(var, Var) else var.val for var in condition_eqn.invars[:-1]]

            outputs = _as_tuple(forward.embedd(*condition_args, name=name))
            result_values = outputs[: len(returned)]
            temporary_values: dict[Any, Any] = dict(zip(extra, outputs[len(returned) :]))
            for var, value in zip(returned, result_values):
                if _contains(temporaries, var):
                    temporary_values[var] = value

            res = use_jaspr.embedd(*ctrl_args, *args, *result_values, inline=True)
            # injection_transform prepends each injected array, so they arrive in reverse order
            inverse.embedd(*[temporary_values[var] for var in temporaries[::-1]], *condition_args, name=name + "_dg")

            qs = TracingQuantumSession.get_instance()
            for var in temporaries:
                qs.abs_qst = delete_qubits_p.bind(temporary_values[var], qs.abs_qst)
            return res

        return make_jaspr(conditioned)(*ctrl, *[var.aval for var in body.invars[:-1]])

    input_positions = [i for i, var in enumerate(body.invars[:-1]) if _contains(inputs, var)]
    return _complete_condition(condition_with, use_jaspr, input_positions)


def _complete_condition(condition_with: Callable[..., Jaspr], use_jaspr: Jaspr, input_positions: list[int]) -> Jaspr:
    """Return the condition executing ``use_jaspr``, with its inverse and controlled versions.

    All of them are marked as permeable on the inputs of the condition, which
    are the invars at ``input_positions`` (after the control qubit).
    """

    def with_controlled(use_jaspr: Jaspr) -> tuple[Jaspr, Jaspr]:
        """Return the condition executing ``use_jaspr`` and its controlled version."""
        res = condition_with(use_jaspr)
        ctrl = condition_with(control_jaspr(use_jaspr), ctrl=(AbstractQubit(),))
        res.ctrl_jaspr = ctrl
        res.permeability.update({res.invars[i]: True for i in input_positions})
        ctrl.permeability.update({ctrl.invars[0]: True, **{ctrl.invars[i + 1]: True for i in input_positions}})
        return res, ctrl

    res, ctrl = with_controlled(use_jaspr)

    # Some bodies can't be inverted, for instance loops that are not based on
    # jrange. Inverting the condition then fails as for any such function.
    try:
        use_inverse = use_jaspr.inverse()
    except Exception:
        return res

    res_inverse, ctrl_inverse = with_controlled(use_inverse)
    res_inverse.inv_jaspr = res
    res.inv_jaspr = res_inverse
    # Nested conditions are executed through their controlled version
    ctrl_inverse.inv_jaspr = ctrl
    ctrl.inv_jaspr = ctrl_inverse
    return res


def _check_temporaries(condition: Jaspr, temporaries: list[Var], description: str) -> None:
    """Raise an error if the condition creates temporaries that can't be uncomputed."""
    leaks = any(_leaks_allocations(eqn) for eqn in _open(condition).eqns if eqn.primitive.name in _BLOCKS)
    if leaks or not all(_is_injectable(condition, temporary) for temporary in temporaries):
        raise ConditionCompilationError(
            f"{_capitalized(description)} creates a QuantumVariable inside classical control flow (a jrange loop "
            "or a classically controlled environment) and doesn't delete it. Such QuantumVariables can not be "
            "uncomputed after the condition.\n\nCreate the QuantumVariable outside of the classical control "
            "flow, or delete it inside of it."
        )


def _injected_inverse(condition: Jaspr, temporaries: list[Var]) -> Jaspr:
    """Return the inverse of the condition, receiving its temporaries as inputs instead of allocating them.

    ``injection_transform`` prepends each injected array, so the inverse receives
    the temporaries in reverse order, followed by the inputs of the condition.
    """
    injected = _sub_jaspr(condition, condition.invars, condition.eqns, [*temporaries, condition.outvars[-1]])
    for temporary in temporaries:
        injected = injection_transform(injected, temporary)
    return injected.inverse()


def _check_body(
    use: Jaspr, use_analysis: "_AllocationAnalysis", temporaries: list[Var], inputs: list[Var], description: str
) -> None:
    """Raise an error if the body uses the temporaries after the condition, or deletes them or the inputs."""
    if any(use_analysis.roots_of(var) & set(temporaries) for var in use.outvars[:-1]):
        raise ConditionCompilationError(
            f"The truth value of {description} or one of its intermediate results is used after the "
            "condition. They are uncomputed and deleted at the end of the condition, so they are only "
            "available inside of it.\n\nUse them inside of the condition."
        )
    if use_analysis.deleted & set(temporaries):
        raise ConditionCompilationError(
            f"The body of {description} deletes its truth value or one of its intermediate results. "
            "They are uncomputed and deleted at the end of the condition.\n\nRemove the deletion."
        )
    if use_analysis.deleted & set(inputs):
        raise ConditionCompilationError(
            f"The body of {description} deletes one of its arguments. They are needed after the body to "
            "uncompute the condition.\n\nDelete them after the condition."
        )


def _undo_flips(controlled_use: Jaspr, flips: int, truth_position: int) -> Jaspr:
    """Return the controlled body, followed by a flip of the truth value if it was flipped an odd number of times."""

    def use_with_restore(*args: Any) -> Any:
        """Execute the controlled body and undo an odd number of flips of the truth value."""
        res = controlled_use.embedd(*args, inline=True)
        if flips % 2:
            from qrisp import x

            x(get_qubit(args[truth_position], 0))
        return res

    return make_jaspr(use_with_restore)(*[var.aval for var in controlled_use.invars[:-1]])


@dataclass(frozen=True)
class _Protected:
    """The roots of the qubits that the body of a condition may only act on permeably."""

    truth_value: frozenset
    temporaries: frozenset
    inputs: frozenset


def _control_on_truth_value(
    use: Jaspr, use_analysis: "_AllocationAnalysis", truth_qubit: Var, protected: _Protected, description: str
) -> tuple[Jaspr, int]:
    """Control the operations of the body on the truth value.

    Operations that involve the truth value are not controlled. Operations on
    the protected qubits must be permeable on them, except for flips of the truth
    value, which are counted. Measurements and resets are rejected, except for
    those of qubits that a block allocates itself.

    Returns the controlled body and the number of flips.
    """
    # The qubits that exist at the level of the body
    known = frozenset().union(*use_analysis.roots.values())
    new_eqns = []
    flips = 0

    for eqn in use.eqns:
        in_roots = [use_analysis.roots_of(var) for var in eqn.invars]
        if _measures_or_resets(eqn, in_roots, known):
            raise ConditionCompilationError(
                f"The body of {description} performs a measurement or a reset, which can not be controlled on "
                "the condition.\n\nMove it out of the condition."
            )

        if eqn.primitive.name not in ("jasp.quantum_gate", *_BLOCKS):
            new_eqns.append(eqn)
            continue

        if _is_flip(eqn, in_roots, protected.truth_value):
            flips += 1
        else:
            _check_operation(eqn, in_roots, protected, description)
        uses_truth_value = any(roots & protected.truth_value for roots in in_roots)
        new_eqns.append(eqn if uses_truth_value else control_eqn(eqn, truth_qubit))

    controlled = Jaspr(
        constvars=list(use.constvars),
        invars=list(use.invars),
        outvars=list(use.outvars),
        eqns=new_eqns,
        consts=list(use.consts),
        debug_info=dummy_debug_info,
    )
    return controlled, flips


def _is_flip(eqn: JaxprEqn, in_roots: list[frozenset], truth_roots: frozenset) -> bool:
    """Return True if ``eqn`` is an x gate on the truth value."""
    is_x = eqn.primitive.name == "jasp.quantum_gate" and eqn.params["gate"].name == "x"
    return is_x and bool(in_roots[0]) and in_roots[0] <= truth_roots


def _check_operation(eqn: JaxprEqn, in_roots: list[frozenset], protected: _Protected, description: str) -> None:
    """Raise an error if an operation of the body can't be executed inside of the condition."""
    if any(roots & protected.temporaries for roots in in_roots) and not _acts_permeably(
        eqn, in_roots, protected.temporaries
    ):
        raise ConditionCompilationError(
            f"The body of {description} applies an operation to its truth value or to one of its "
            "intermediate results that is not supported. Inside of a condition, they can be used as a "
            "control and for phases, and the truth value can be flipped with QuantumBool.flip() to invert "
            "the condition.\n\nFor other operations, compute a separate QuantumBool."
        )
    if any(roots & protected.inputs for roots in in_roots) and not _acts_permeably(eqn, in_roots, protected.inputs):
        raise ConditionCompilationError(
            f"The body of {description} changes one of its arguments. They must stay unchanged, since the "
            "condition is uncomputed from them afterwards. Inside of a condition, its arguments can be used as "
            "a control and for phases. This also excludes functions that change them only temporarily, such "
            "as the comparisons <, >, <= and >=.\n\nChange the arguments outside of the condition, and enter "
            "such comparisons as nested conditions, for instance with qf < 3:."
        )
    if any(roots & protected.truth_value for roots in in_roots) and not _acts_only_on(
        eqn, in_roots, protected.truth_value
    ):
        raise ConditionCompilationError(
            f"The body of {description} contains a function or classical control flow that uses the truth value "
            "together with operations that don't involve it. Such blocks are not controlled on the condition, "
            "so these operations would be executed even if the condition is false.\n\nMove these operations "
            "out of the block. If the block is a function that uses the truth value as a control for them, "
            "declare its permeability with gate_wrap on top of qache."
        )


def _acts_permeably(eqn: JaxprEqn, in_roots: list[frozenset], targets: frozenset) -> bool:
    """Return True if every gate in ``eqn`` that acts on ``targets`` is permeable on them.

    Blocks are analyzed recursively, unless they are a function whose arguments
    acting on ``targets`` are declared permeable with ``gate_wrap``. Other
    operations on the targets, such as measurements and deletions, are not
    permeable.
    """
    if eqn.primitive.name == "jasp.quantum_gate":
        gate = eqn.params["gate"]
        return all(gate.permeability.get(i) is True for i in range(gate.num_qubits) if in_roots[i] & targets)

    if eqn.primitive.name in _BLOCKS:
        return _is_declared(eqn, in_roots, targets) or all(
            _acts_permeably(sub_eqn, sub_roots, targets)
            for sub_eqn, sub_roots in _sub_equations(eqn, in_roots)
            if any(roots & targets for roots in sub_roots)
        )

    # Qubit array operations like get_qubit don't act on the qubits
    return not _acts_on_state(eqn) or not any(roots & targets for roots in in_roots)


def _acts_only_on(eqn: JaxprEqn, in_roots: list[frozenset], targets: frozenset) -> bool:
    """Return True if every gate in ``eqn`` certainly acts on a qubit that stems from ``targets``.

    Blocks are analyzed recursively, unless they are a function whose arguments
    acting on ``targets`` are declared permeable with ``gate_wrap``, which counts
    as a single gate.
    """
    if eqn.primitive.name == "jasp.quantum_gate" or _is_declared(eqn, in_roots, targets):
        return any(roots and roots <= targets for roots in in_roots)

    if eqn.primitive.name in _BLOCKS:
        return all(_acts_only_on(sub_eqn, sub_roots, targets) for sub_eqn, sub_roots in _sub_equations(eqn, in_roots))

    return True


def _is_declared(eqn: JaxprEqn, in_roots: list[frozenset], targets: frozenset) -> bool:
    """Return True if ``eqn`` calls a function that declares its arguments acting on ``targets`` permeable."""
    if eqn.primitive.name != "jit":
        return False
    jaspr = eqn.params["jaxpr"]
    declared = getattr(jaspr, "permeability", {})
    involved = [var for var, roots in zip(_open(jaspr).invars, in_roots) if roots & targets]
    return bool(involved) and all(declared.get(var) is True for var in involved)


def _block_analyses(eqn: JaxprEqn, in_roots: list[frozenset]) -> list[tuple[Any, "_AllocationAnalysis"]]:
    """Return the jaxprs of a jit, cond or while equation with the analyses of their roots."""
    name = eqn.primitive.name
    if name == "jit":
        return [(eqn.params["jaxpr"], _AllocationAnalysis(eqn.params["jaxpr"], in_roots))]
    if name == "cond":
        return [(branch, _AllocationAnalysis(branch, in_roots[1:])) for branch in eqn.params["branches"]]
    return [(eqn.params["body_jaxpr"], _while_analysis(eqn, in_roots))]


def _sub_equations(eqn: JaxprEqn, in_roots: list[frozenset]) -> Iterator[tuple[JaxprEqn, list[frozenset]]]:
    """Yield the equations of a block with the roots of their invars."""
    for jaxpr, analysis in _block_analyses(eqn, in_roots):
        for sub_eqn in _open(jaxpr).eqns:
            yield sub_eqn, [analysis.roots_of(var) for var in sub_eqn.invars]


class _Fresh:
    """Type of the _FRESH marker."""


# Marks qubit arrays that a block returns without having received them
_FRESH = _Fresh()


class _AllocationAnalysis:
    """Follow the qubit arrays of a Jaspr back to the inputs or allocations they stem from.

    Every qubit-type variable is mapped to its *roots*: the invars or
    allocations it stems from, through ``get_qubit``, ``slice`` and ``fuse``. An
    allocation is a ``create_qubits`` equation or a qubit array that a block
    (``jit``, ``cond`` or ``while``) returns without having received it.

    Parameters
    ----------
    jaxpr : Jaxpr | ClosedJaxpr
        The Jaspr to analyze.
    in_roots : list[frozenset] | None, optional
        The roots of the invars. By default, every qubit-type invar is its own root.

    Attributes
    ----------
    roots : dict[Var, frozenset]
        The roots of every qubit-type variable.
    allocations : list[Var]
        The allocations, in the order of the equations.
    deleted : set
        The roots that are deleted.
    temporaries : list[Var]
        The allocations that are not deleted.
    out_roots : list[frozenset]
        The roots of the outvars.

    """

    def __init__(self, jaxpr: Jaxpr | ClosedJaxpr, in_roots: list[frozenset] | None = None) -> None:
        """Analyze ``jaxpr``."""
        jaxpr = _open(jaxpr)
        if in_roots is None:
            in_roots = [frozenset([var]) if _is_quantum(var) else frozenset() for var in jaxpr.invars]
        self.roots: dict[Var, frozenset] = {var: roots for var, roots in zip(jaxpr.invars, in_roots) if roots}
        self.allocations: list[Var] = []
        self.deleted: set = set()

        for eqn in jaxpr.eqns:
            out_roots, deleted = _eqn_roots(eqn, [self.roots_of(var) for var in eqn.invars])
            self.deleted |= deleted
            for var, roots in zip(eqn.outvars, out_roots):
                var_roots = roots
                if _FRESH in roots:
                    var_roots = (roots - {_FRESH}) | {var}
                    self.allocations.append(var)
                if var_roots:
                    self.roots[var] = var_roots

        self.temporaries: list[Var] = [var for var in self.allocations if var not in self.deleted]
        self.out_roots: list[frozenset] = [self.roots_of(var) for var in jaxpr.outvars]

    def roots_of(self, var: Var | Literal) -> frozenset:
        """Return the roots of ``var``."""
        if isinstance(var, Literal):
            return frozenset()
        return self.roots.get(var, frozenset())


def _eqn_roots(eqn: JaxprEqn, in_roots: list[frozenset]) -> tuple[list[frozenset], set]:
    """Return the roots of the outvars of ``eqn`` and the roots it deletes.

    Qubit arrays that the equation allocates are marked with ``_FRESH``.
    """
    name = eqn.primitive.name

    if name == "jasp.create_qubits":
        return [frozenset([_FRESH]), frozenset()], set()

    if name == "jasp.delete_qubits":
        return [frozenset()] * len(eqn.outvars), set(in_roots[0])

    if name in _BLOCKS:
        outer = frozenset().union(*in_roots)
        out_roots = [frozenset() for _ in eqn.outvars]
        deleted = set()
        for _, analysis in _block_analyses(eqn, in_roots):
            out_roots = [
                roots | {root if root in outer else _FRESH for root in analysis_roots}
                for roots, analysis_roots in zip(out_roots, analysis.out_roots)
            ]
            deleted |= analysis.deleted & outer
        return out_roots, deleted

    # Classical equations and qubit array operations (get_qubit, slice, fuse, ...)
    quantum_in = frozenset().union(*[roots for var, roots in zip(eqn.invars, in_roots) if _is_quantum(var)])
    return [quantum_in if _is_quantum(var) else frozenset() for var in eqn.outvars], set()


def _while_analysis(eqn: JaxprEqn, in_roots: list[frozenset]) -> _AllocationAnalysis:
    """Return the analysis of a while loop body, with the carried roots of all iterations."""
    cond_nconsts, body_nconsts = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
    consts = in_roots[cond_nconsts : cond_nconsts + body_nconsts]
    carry = in_roots[cond_nconsts + body_nconsts :]
    # Iterate until the carried roots are stable, since iterations may permute them
    analysis = _AllocationAnalysis(eqn.params["body_jaxpr"], consts + carry)
    for _ in range(len(carry)):
        new_carry = [a | {root for root in b if root is not _FRESH} for a, b in zip(carry, analysis.out_roots)]
        if new_carry == carry:
            break
        carry = new_carry
        analysis = _AllocationAnalysis(eqn.params["body_jaxpr"], consts + carry)
    return analysis


def _leaks_allocations(eqn: JaxprEqn) -> bool:
    """Return True if a block allocates qubits that it neither deletes nor returns, also inside of nested blocks."""
    for body in _block_bodies(eqn):
        analysis = _AllocationAnalysis(body)
        returned = frozenset().union(*analysis.out_roots)
        if any(temporary not in returned for temporary in analysis.temporaries):
            return True
        if any(_leaks_allocations(sub_eqn) for sub_eqn in _open(body).eqns if sub_eqn.primitive.name in _BLOCKS):
            return True
    return False


# Primitives of equations that contain jaxprs
_BLOCKS = ("jit", "cond", "while")


def _block_bodies(eqn: JaxprEqn) -> list:
    """Return the jaxprs of a jit, cond or while equation."""
    name = eqn.primitive.name
    if name == "jit":
        return [eqn.params["jaxpr"]]
    if name == "cond":
        return list(eqn.params["branches"])
    return [eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]]


def _measures_or_resets(eqn: JaxprEqn, in_roots: list[frozenset], targets: frozenset) -> bool:
    """Return True if ``eqn`` measures or resets a qubit that stems from ``targets``, also inside of blocks.

    Measurements and resets of qubits that a block allocates itself, for
    instance in measurement-based uncomputation, don't stem from ``targets``.
    """
    if eqn.primitive.name in ("jasp.measure", "jasp.reset"):
        return any(roots & targets for roots in in_roots)
    return eqn.primitive.name in _BLOCKS and any(
        _measures_or_resets(sub_eqn, sub_roots, targets) for sub_eqn, sub_roots in _sub_equations(eqn, in_roots)
    )


def _inline_functions(jaspr: Jaspr) -> Jaspr:
    """Return ``jaspr`` with the jit equations inlined that hide allocations, also nested ones.

    Other jit equations are kept, since they may carry a custom inverse. Jit
    equations inside of classical control flow are not inlined.
    """

    def eqn_evaluator(eqn: JaxprEqn, context_dic: Any) -> bool | None:
        """Evaluate jit equations that hide allocations by inlining their equations, and all others as they are."""
        if eqn.primitive.name != "jit" or not _leaks_allocations(eqn):
            return True
        res = eval_jaxpr(eqn.params["jaxpr"], eqn_evaluator=eqn_evaluator)(*extract_invalues(eqn, context_dic))
        insert_call_outvalues(eqn, context_dic, res, len(eqn.outvars))
        return None

    def inlined(*args: Any) -> Any:
        """Execute ``jaspr`` with the jit equations inlined that hide allocations."""
        qs = TracingQuantumSession.get_instance()
        res = _as_tuple(eval_jaxpr(jaspr, eqn_evaluator=eqn_evaluator)(*args, qs.abs_qst))
        qs.abs_qst = res[-1]
        return res[:-1]

    return make_jaspr(inlined)(*[var.aval for var in jaspr.invars[:-1]])


def _is_injectable(jaxpr: Jaxpr | ClosedJaxpr, var: Var | Literal) -> bool:
    """Return True if ``var`` is allocated by a ``create_qubits`` equation, possibly inside (nested) jit equations."""
    for eqn in _open(jaxpr).eqns:
        if _contains(eqn.outvars, var):
            if eqn.primitive.name == "jasp.create_qubits":
                return True
            if eqn.primitive.name == "jit":
                sub_jaxpr = _open(eqn.params["jaxpr"])
                index = next(i for i, outvar in enumerate(eqn.outvars) if outvar is var)
                return _is_injectable(sub_jaxpr, sub_jaxpr.outvars[index])
            return False
    return False


def _sub_jaspr(base: Jaspr, invars: list, eqns: list[JaxprEqn], outvars: list) -> Jaspr:
    """Return a Jaspr with the given signature and equations, sharing the constants of ``base``."""
    return Jaspr(
        constvars=list(base.constvars),
        invars=list(invars),
        outvars=list(outvars),
        eqns=list(eqns),
        consts=list(base.consts),
        debug_info=dummy_debug_info,
    )


def _open(jaxpr: Jaxpr | ClosedJaxpr) -> Jaxpr:
    """Return the Jaxpr of a ClosedJaxpr (including Jasprs)."""
    return jaxpr.jaxpr if isinstance(jaxpr, ClosedJaxpr) else jaxpr


def _acts_on_state(eqn: JaxprEqn) -> bool:
    """Return True if ``eqn`` receives the quantum state, i.e. acts on qubits."""
    from qrisp.jasp import AbstractQuantumState

    return any(isinstance(getattr(var, "aval", None), AbstractQuantumState) for var in eqn.invars)


def _is_quantum(var: Any) -> bool:
    """Return True if ``var`` is a qubit or qubit array."""
    return isinstance(getattr(var, "aval", None), (AbstractQubit, AbstractQubitArray))


def _contains(variables: list, var: Any) -> bool:
    """Return True if ``var`` is one of ``variables`` (by identity, since Literals can't be compared)."""
    return any(variable is var for variable in variables)


def _as_tuple(res: Any) -> tuple:
    """Return the results of Jaspr.embedd as a tuple."""
    if res is None:
        return ()
    return tuple(res) if isinstance(res, (tuple, list)) else (res,)


def describe_condition(function: Callable) -> str:
    """Name the condition in error messages."""
    name = getattr(function, "__name__", "<lambda>")
    return "the quantum condition" if name == "<lambda>" else f"the quantum condition {name}"


def _capitalized(text: str) -> str:
    """Return ``text`` with its first letter in upper case."""
    return text[:1].upper() + text[1:]
