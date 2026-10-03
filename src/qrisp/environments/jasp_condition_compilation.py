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

from collections.abc import Callable
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
# QuantumBools of the comparisons in (qf < 3) | (qf == 7). C_fwd is C,
# additionally returning the temporaries it doesn't return. C_inv is the
# inverse of C with the temporaries turned into inputs by injection_transform,
# so that it uncomputes the existing qubits instead of allocating new ones.
#
# Why the temporaries end up in |0>: since C maps computational basis states to
# computational basis states (up to a phase), and U acts on the qubits of C only
# as a control or through phases, U never changes the values of the qubits C
# acted on. C_inv therefore finds them as C left them and maps them back.
#
# Operations in U that involve the truth value itself are not controlled on it:
# phases on the truth value are applied as they are, and flips of the truth
# value (QuantumBool.flip) invert the condition for the subsequent operations.
# An odd number of flips is undone before C_inv.
#
# The result carries a custom inverse (C_fwd ; U^dagger ; C_inv ; delete) and a
# custom controlled version (C_fwd ; controlled U ; C_inv ; delete), because
# structural inversion would reverse the allocation inside C_fwd, and plain
# control would control C as well.


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
    condition = condition_eqn.params["jaxpr"]
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

    # Track the temporaries returned by the condition through the body
    result_roots = [analysis.roots_of(var) & set(temporaries) for var in returned]
    use_analysis = _AllocationAnalysis(use, [frozenset()] * (len(body.invars) - 1) + result_roots + [frozenset()])
    _check_body(use, use_analysis, temporaries, description)

    controlled_use, flips = _control_on_truth_value(use, use_analysis, truth_eqn.outvars[0], temporaries, description)
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
            temporary_values = dict(zip(extra, outputs[len(returned) :]))
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

    res = condition_with(use_jaspr)
    res.ctrl_jaspr = condition_with(control_jaspr(use_jaspr), ctrl=(AbstractQubit(),))

    # A body with measurements can't be inverted. Inverting the condition then
    # fails as for any other function with measurements.
    try:
        use_inverse = use_jaspr.inverse()
    except Exception:
        return res

    res_inverse = condition_with(use_inverse)
    res_inverse.ctrl_jaspr = condition_with(control_jaspr(use_inverse), ctrl=(AbstractQubit(),))
    res_inverse.inv_jaspr = res
    res.inv_jaspr = res_inverse
    return res


def _check_temporaries(condition: Jaspr, temporaries: list[Var], description: str) -> None:
    """Raise an error if the condition creates temporaries that can't be uncomputed."""
    leaks = any(_leaks_allocations(eqn) for eqn in _open(condition).eqns if eqn.primitive.name in ("cond", "while"))
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


def _check_body(use: Jaspr, use_analysis: "_AllocationAnalysis", temporaries: list[Var], description: str) -> None:
    """Raise an error if the body uses the temporaries after the condition or deletes them."""
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


def _control_on_truth_value(
    use: Jaspr, use_analysis: "_AllocationAnalysis", truth_qubit: Var, temporaries: list[Var], description: str
) -> tuple[Jaspr, int]:
    """Control the operations of the body on the truth value.

    Operations that involve the truth value are not controlled. Operations that
    involve the temporaries must act on them permeably (as a control or through
    phases), except for flips of the truth value, which are counted.

    Returns the controlled body and the number of flips.
    """
    truth_roots = use_analysis.roots_of(truth_qubit)
    temporary_roots = frozenset(temporaries)
    new_eqns = []
    flips = 0

    for eqn in use.eqns:
        name = eqn.primitive.name

        if name == "jasp.measure":
            raise ConditionCompilationError(
                f"The body of {description} performs a measurement, which can not be controlled on the "
                "condition.\n\nMove the measurement out of the condition."
            )

        if name not in ("jasp.quantum_gate", "jit", "cond", "while"):
            new_eqns.append(eqn)
            continue

        in_roots = [use_analysis.roots_of(var) for var in eqn.invars]
        if not any(roots & temporary_roots for roots in in_roots):
            new_eqns.append(control_eqn(eqn, truth_qubit))
            continue

        uses_truth_value = any(roots & truth_roots for roots in in_roots)
        if uses_truth_value and name == "jasp.quantum_gate" and eqn.params["gate"].name == "x":
            flips += 1
        elif not _acts_permeably(eqn, in_roots, temporary_roots):
            raise ConditionCompilationError(
                f"The body of {description} applies an operation to its truth value or to one of its "
                "intermediate results that is not supported. Inside of a condition, they can be used as a "
                "control and for phases, and the truth value can be flipped with QuantumBool.flip() to invert "
                "the condition.\n\nFor other operations, compute a separate QuantumBool."
            )
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


def _acts_permeably(eqn: JaxprEqn, in_roots: list[frozenset], targets: frozenset) -> bool:
    """Return True if every gate in ``eqn`` that acts on ``targets`` is permeable on them.

    Blocks are analyzed recursively, unless they are a function whose arguments
    acting on ``targets`` are declared permeable with ``gate_wrap``. Measurements
    and deletions of the targets are not permeable.
    """
    name = eqn.primitive.name

    if name == "jasp.quantum_gate":
        gate = eqn.params["gate"]
        return all(gate.permeability.get(i) is True for i in range(gate.num_qubits) if in_roots[i] & targets)

    if name in ("jasp.measure", "jasp.delete_qubits"):
        return not any(roots & targets for roots in in_roots)

    if name == "jit":
        jaspr = eqn.params["jaxpr"]
        declared = getattr(jaspr, "permeability", {})
        is_declared = all(
            declared.get(var) is True for var, roots in zip(_open(jaspr).invars, in_roots) if roots & targets
        )
        bodies = [] if is_declared else [(jaspr, in_roots)]
    elif name == "cond":
        bodies = [(branch, in_roots[1:]) for branch in eqn.params["branches"]]
    elif name == "while":
        cond_nconsts = eqn.params["cond_nconsts"]
        bodies = [(eqn.params["body_jaxpr"], in_roots[cond_nconsts:])]
    elif not _acts_on_state(eqn):
        # Qubit array operations like get_qubit don't act on the qubits
        return True
    else:
        return not any(roots & targets for roots in in_roots)

    for jaxpr, roots in bodies:
        analysis = _AllocationAnalysis(jaxpr, roots)
        for sub_eqn in _open(jaxpr).eqns:
            sub_roots = [analysis.roots_of(var) for var in sub_eqn.invars]
            if any(r & targets for r in sub_roots) and not _acts_permeably(sub_eqn, sub_roots, targets):
                return False
    return True


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

    if name in ("jit", "cond", "while"):
        outer = frozenset().union(*in_roots)
        if name == "jit":
            analyses = [_AllocationAnalysis(eqn.params["jaxpr"], in_roots)]
        elif name == "cond":
            analyses = [_AllocationAnalysis(branch, in_roots[1:]) for branch in eqn.params["branches"]]
        else:
            analyses = [_while_analysis(eqn, in_roots)]
        out_roots = [frozenset() for _ in eqn.outvars]
        deleted = set()
        for analysis in analyses:
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
    """Return True if a cond or while equation allocates qubits that it neither deletes nor returns."""
    bodies = eqn.params["branches"] if eqn.primitive.name == "cond" else [eqn.params["body_jaxpr"]]
    for body in bodies:
        analysis = _AllocationAnalysis(body)
        returned = frozenset().union(*analysis.out_roots)
        if any(temporary not in returned for temporary in analysis.temporaries):
            return True
    return False


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
