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

"""Implements the jasp_uncompute decorator, which uncomputes the temporary QuantumVariables of a function in Jasp."""

import functools

import jax
from jax.extend.core import ClosedJaxpr, Literal, Var

from qrisp.environments import QuantumEnvironment
from qrisp.jasp import (
    AbstractQuantumState,
    AbstractQubit,
    AbstractQubitArray,
    Jaspr,
    TracingQuantumSession,
    check_for_tracing_mode,
    control_jaspr,
    eval_jaxpr,
    extract_invalues,
    get_last_equation,
    injection_transform,
    insert_outvalues,
    make_jaspr,
)
from qrisp.jasp.jasp_expression.environment_collection import dummy_debug_info
from qrisp.jasp.primitives import delete_qubits_p


def jasp_uncompute(function):
    r"""Decorator that uncomputes the temporary QuantumVariables of a function in Jasp.

    A temporary is a :ref:`QuantumVariable` that the decorated function allocates
    and neither returns nor deletes, typically the :ref:`QuantumBool` results of
    comparisons and logical operations. After the function has used them,
    ``jasp_uncompute`` reverts their computation, so that they return to
    :math:`\ket{0}`, and deletes them.

    Outside of Jasp mode, the decorator behaves like
    :func:`auto_uncompute <qrisp.auto_uncompute>`.

    .. rubric:: When to use it

    ``jasp_uncompute`` is meant for functions with a *compute, then use*
    structure, such as Grover oracles: first all temporaries are computed, then
    they are used. The function body is split at the last operation that
    modifies a temporary. Everything up to that point is the *computation* :math:`C`,
    everything after it is the *use* :math:`U`. The function is then executed as
    :math:`C^\dagger U C`. If it is called in a controlled context, only :math:`U`
    is controlled.

    For this to be correct, the function must satisfy the following conditions.
    Conditions 1, 3 and 4 are checked and violations raise an error.
    Condition 2 can not be checked automatically and is the responsibility of
    the user, as for :ref:`ConjugationEnvironment`.

    #. **Compute, then use.** All operations that compute temporaries come
       before all operations that use them. In particular, the computation must
       neither use a temporary without modifying another temporary (for
       instance by applying a phase to it) nor modify a QuantumVariable that
       the function returns.
    #. **Clean computation.** The computation is :ref:`qfree <uncomputation>`:
       it maps computational basis states to computational basis states (up to
       a phase), as Qrisp's comparisons, logical operations and arithmetic do.
       Ancillas it allocates and deletes must be back in :math:`\ket{0}` before
       they are deleted.
    #. **Permeable use.** The use only acts diagonally (in the computational
       basis) on the temporaries and on every qubit the computation reads. It
       may for instance apply phases or controlled operations depending on the
       temporaries, but it must not change their value or the value of the
       inputs they were computed from.
    #. **No measurements.** Neither the computation nor the use measures a
       temporary, and the computation contains no measurement at all.

    Additionally, temporaries must be created directly in the function body or
    by a gate-wrapped or qached function, not inside classical control flow
    such as :func:`jrange <qrisp.jasp.jrange>` loops or classically controlled
    environments. Do not delete temporaries yourself, as a deleted
    QuantumVariable is not uncomputed.

    Decorated functions can be inverted, controlled and used inside other
    decorated functions.

    Parameters
    ----------
    function : callable
        The function whose temporaries should be uncomputed.

    Returns
    -------
    callable
        The decorated function.

    Raises
    ------
    JaspUncomputationError
        If the function violates condition 1, 3 or 4, or creates temporaries
        inside classical control flow. The error is raised when the Jasp program
        is compiled, for instance by :func:`jaspify <qrisp.jasp.jaspify>`.

    Examples
    --------
    We create a Grover oracle that tags the states of a :ref:`QuantumFloat`
    that are smaller than 3 or equal to 7. The comparisons create three
    temporary :ref:`QuantumBools <QuantumBool>`, which are uncomputed after the
    phase tag.

    ::

        from qrisp import QuantumFloat, jasp_uncompute, z
        from qrisp.grover import grovers_alg
        from qrisp.jasp import terminal_sampling

        @jasp_uncompute
        def oracle(qf):
            flag1 = qf < 3
            flag2 = qf == 7
            flag_all = flag1 | flag2
            z(flag_all)

        @terminal_sampling
        def main():
            qf = QuantumFloat(4)
            grovers_alg(qf, oracle, iterations=1)
            return qf

    Four of the 16 states are tagged, so a single Grover iteration finds them
    with certainty.

    >>> main()
    {0.0: 0.25, 1.0: 0.25, 2.0: 0.25, 7.0: 0.25}

    Without ``jasp_uncompute``, the temporaries would remain entangled with
    ``qf``, and the tagged states would only be found with a probability of
    about 70%.

    If the oracle is called in a controlled context, only the phase tag is
    controlled, while the comparisons and their uncomputation are not.

    ::

        from qrisp import QuantumBool, control

        def controlled_oracle(qf, ctrl):
            with control(ctrl):
                oracle(qf)

    """
    from qrisp.permeability import auto_uncompute

    auto_uncomputed_function = auto_uncompute(function)

    @functools.wraps(function)
    def uncomputed_function(*args, **kwargs):
        if not check_for_tracing_mode():
            return auto_uncomputed_function(*args, **kwargs)

        with JaspUncomputationEnvironment(function.__name__):
            return function(*args, **kwargs)

    return uncomputed_function


class JaspUncomputationEnvironment(QuantumEnvironment):
    """Environment behind :func:`jasp_uncompute`. Its body is uncomputed when the environments are flattened."""

    def __init__(self, function_name):
        """Create the environment for the decorated function ``function_name``."""
        QuantumEnvironment.__init__(self)
        # Not called name, since the environment instance is also the JAX primitive.
        self.function_name = function_name

    # How the uncomputation works
    # ===========================
    #
    # The environment body is the traced function as a Jaspr B. We look at the
    # top-level equations of B that act on the quantum state ("quantum
    # equations"): gates, qubit allocations/deletions, measurements and blocks
    # (jit, cond, while).
    #
    # 1. Temporaries. Qubit arrays are tracked by their *root*: the invar or
    #    the allocation they stem from (through get_qubit, slicing and fusing).
    #    An allocation is a top-level create_qubits or a qubit array that a
    #    block returns without having received it. The temporaries are the
    #    allocations that are neither deleted nor among the outvars of B.
    #
    # 2. Effects. For every quantum equation we determine, per root, whether it
    #    acts on that root permeably (commutes with Z on all its qubits),
    #    writes it, measures it, deletes it, or whether this is unknown:
    #    - a gate by the permeability of its Operation,
    #    - a block by recursing into its body, overridden by the permeability
    #      stored on its Jaspr (set by gate_wrap, see Jaspr.inherit_permeability).
    #      This override matters: an MCX implementation applies X gates to its
    #      controls, which makes it look like it writes them, although the block
    #      as a whole is permeable on its controls.
    #
    # 3. Split. Let L be the last quantum equation that allocates a temporary or
    #    writes one. The equations up to and including L form the computation
    #    C, the remaining ones the use U. If the effect of equation L on the
    #    temporaries is unknown, we can't tell whether it belongs to C or U and
    #    raise an error instead of guessing.
    #
    # 4. Checks. C contains no measurement, every equation of C that touches a
    #    temporary also writes one (otherwise it is a use, which C^dagger would
    #    cancel), C doesn't write a qubit array that outlives the function, U is
    #    permeable on every root that C touches and nothing measures a
    #    temporary.
    #
    # 5. Emission. B is replaced by a single jit equation K that executes
    #
    #        C_fwd ; U ; C_inv ; delete temporaries
    #
    #    C_fwd is C including the allocation of the temporaries. C_inv is the
    #    inverse of C with the temporaries (and allocations that U still needs)
    #    turned into inputs by injection_transform, so that it acts on the
    #    existing qubits instead of allocating new ones. K carries a custom
    #    inverse (C_fwd ; U^dagger ; C_inv ; delete) and a custom controlled
    #    version (C_fwd ; controlled U ; C_inv ; delete), because structural
    #    inversion of K would reverse the allocation inside C_fwd, and plain
    #    control would control C as well.
    #
    # Why this is correct
    # -------------------
    #
    # Setup. We split the qubits into two groups:
    #
    # - D: the qubits C acts on, i.e. the inputs that C reads (e.g. the
    #   QuantumFloat of an oracle) and the temporaries. The ancillas that C
    #   allocates and deletes internally are not part of D, since C returns
    #   them to |0> before deleting them.
    # - R: all other qubits, e.g. a target qubit that only U writes, ancillas
    #   of U, or the control qubit if the function is called in a controlled
    #   context.
    #
    # The state space is H_D (x) H_R. We write |d>, |e> for computational
    # basis states (bitstrings) of D and |psi> for an arbitrary state of R.
    # C acts as the identity on R.
    #
    # The computation. Because C is qfree, it maps every basis state |d> of D
    # whose temporaries are in |0> to a single basis state, up to a phase:
    #
    #     C |d> = e^{i phi(d)} |pi(d)>
    #
    # pi(d) contains the computed values of the temporaries. C doesn't need to
    # leave its inputs unchanged; pi may also permute them.
    #
    # The use. U is permeable on every qubit q of D, i.e. it commutes with Z_q.
    # Hence it also commutes with the projectors (1 + Z_q)/2 = |0><0|_q and
    # (1 - Z_q)/2 = |1><1|_q, and with their products, which are the
    # projectors |e><e| onto the basis states of D. Inserting
    # 1 = sum_e |e><e| on both sides of U, all terms |e><e| U |e'><e'| with
    # e != e' vanish, so U is block diagonal in the basis of D:
    #
    #     U = sum_e |e><e| (x) W_e
    #
    # where W_e = <e|U|e> is a unitary on R: the operation U applies to R if
    # D is in the state |e>. For example, for U = z(flag), W_e is the phase
    # (-1)^flag, and for U = cx(flag, target), W_e is X on the target if
    # flag = 1 and the identity otherwise. U never changes the bitstring on D.
    #
    # Putting it together:
    #
    #     C^dagger U C |d>|psi> = C^dagger e^{i phi(d)} |pi(d)> W_{pi(d)}|psi>
    #                           = |d> W_{pi(d)}|psi>
    #
    # The temporaries are back in |0> and can be deleted, the inputs are
    # unchanged, the phase phi cancels, and R received exactly the operation U
    # applies for the computed temporaries. By linearity, this also holds for
    # superpositions over d and for states that are entangled between D and R.
    #
    # The same argument applies to K's custom inverse with U^dagger and to the
    # controlled version with the controlled U: both are permeable on D as
    # well, with blocks W_e^dagger and controlled W_e. This is also why the
    # split must be exact: an equation of U that ended up in C would be
    # cancelled by C^dagger, and an equation of C that ended up in U would
    # leave the temporaries dirty.

    def jcompile(self, eqn, context_dic):
        """Replace the collected body by its uncomputed version."""
        args = extract_invalues(eqn, context_dic)
        body = eqn.params["jaspr"].flatten_environments()

        uncomputed = _uncompute(body, self.function_name)
        if uncomputed is None:
            res = eval_jaxpr(body)(*args)
        else:
            res = jax.jit(uncomputed.eval)(*args)
            jit_eqn = get_last_equation()
            jit_eqn.params["jaxpr"] = uncomputed
            jit_eqn.params["name"] = self.function_name

        if not isinstance(res, tuple):
            res = (res,)

        insert_outvalues(eqn, context_dic, res)


class JaspUncomputationError(Exception):
    """Raised when :func:`jasp_uncompute` can not uncompute a function."""


# Effects of an equation on a root, ordered by severity
PERMEABLE, UNKNOWN, WRITE, MEASURE, DELETE = range(5)

# Marks qubit arrays that a block returns without having received them
FRESH = object()


def _uncompute(body, name):
    """Return the uncomputed version of ``body``, or None if it has no temporaries."""
    analysis = _TopLevelAnalysis(body)

    for eqn in body.eqns:
        if eqn.primitive.name in ("cond", "while") and _creates_temporaries(eqn):
            raise JaspUncomputationError(
                f"The function {name} creates a temporary QuantumVariable inside classical control flow "
                "(a jrange loop or a classically controlled environment), which jasp_uncompute can not uncompute."
                "\n\nCreate the QuantumVariable outside of the classical control flow."
            )

    temporaries = analysis.temporaries
    if not temporaries:
        return None

    split = analysis.split_index(name)
    analysis.check(split, name)

    crossing, state_after_compute = _crossing_variables(body, split)

    # Allocations that C_inv must receive instead of performing them
    injected = temporaries + [
        root for root in analysis.allocations(split) if root in crossing and root not in temporaries
    ]
    if not all(_is_injectable(body, root) for root in injected):
        raise JaspUncomputationError(
            f"The function {name} creates a QuantumVariable inside classical control flow "
            "(a jrange loop or a classically controlled environment) before all temporaries have been "
            "computed, which jasp_uncompute can not uncompute."
            "\n\nCreate the QuantumVariable outside of the classical control flow."
        )

    compute_eqns = body.eqns[: split + 1]
    compute_outvars = crossing + [root for root in injected if root not in crossing]
    c_fwd = _sub_jaspr(body, body.invars, compute_eqns, [*compute_outvars, state_after_compute])

    c_inj = _sub_jaspr(body, body.invars, compute_eqns, [*injected, state_after_compute])
    for root in injected:
        c_inj = injection_transform(c_inj, root)
    c_inv = c_inj.inverse()

    use = _sub_jaspr(body, [*body.invars[:-1], *crossing, state_after_compute], body.eqns[split + 1 :], body.outvars)

    def conjugate_with(use_jaspr, ctrl=()):
        def conjugated(*args):
            ctrl_args, args = args[: len(ctrl)], args[len(ctrl) :]
            values = dict(zip(compute_outvars, _as_tuple(c_fwd.embedd(*args, name=f"{name}_compute"))))
            res = use_jaspr.embedd(*ctrl_args, *args, *[values[var] for var in crossing], inline=True)
            # injection_transform prepends each injected array, so they arrive in reverse order
            c_inv.embedd(*[values[root] for root in injected[::-1]], *args, name=f"{name}_uncompute")
            qs = TracingQuantumSession.get_instance()
            for root in temporaries:
                qs.abs_qst = delete_qubits_p.bind(values[root], qs.abs_qst)
            return res

        return make_jaspr(conjugated)(*ctrl, *[var.aval for var in body.invars[:-1]])

    res = conjugate_with(use)
    res.ctrl_jaspr = conjugate_with(control_jaspr(use), ctrl=(AbstractQubit(),))

    # U can only be inverted without measurements. Inverting the result then
    # fails as for any other function with measurements.
    if not analysis.use_measures(split):
        inverse = conjugate_with(use.inverse())
        inverse.ctrl_jaspr = conjugate_with(control_jaspr(use.inverse()), ctrl=(AbstractQubit(),))
        inverse.inv_jaspr = res
        res.inv_jaspr = inverse

    # C^dagger U C is permeable on a root if U is, see the comment above jcompile
    for var, root in zip(res.invars, body.invars):
        if _is_quantum(var):
            res.permeability[var] = analysis.use_permeability(split, root)

    return res


def _crossing_variables(body, split):
    """Return the variables that C defines and U or the caller needs, and the quantum state after C."""
    used_after = {var for eqn in body.eqns[split + 1 :] for var in eqn.invars if isinstance(var, Var)}
    used_after.update(var for var in body.outvars if isinstance(var, Var))

    crossing = []
    state_after_compute = body.invars[-1]
    for eqn in body.eqns[: split + 1]:
        for var in eqn.outvars:
            if isinstance(var.aval, AbstractQuantumState):
                state_after_compute = var
            elif var in used_after and var not in crossing:
                crossing.append(var)
    return crossing, state_after_compute


class _TopLevelAnalysis:
    """Roots and effects of the top-level quantum equations of a Jaspr."""

    def __init__(self, body):
        self.body = body
        self.roots = {var: frozenset([var]) for var in body.invars if _is_quantum(var)}
        self.inputs = set(self.roots)
        # Allocation root -> index of the allocating equation
        self.origin = {}
        # Index of every quantum equation -> its effects
        self.effects = {}
        self.deleted = set()

        for i, eqn in enumerate(body.eqns):
            effects, out_roots = _eqn_effects(eqn, self.roots_of)
            for var, roots in zip(eqn.outvars, out_roots):
                var_roots = _resolve_fresh(roots, var)
                if var in var_roots:
                    self.origin[var] = i
                if var_roots:
                    self.roots[var] = var_roots
            if _acts_on_state(eqn):
                self.effects[i] = effects
                self.deleted.update(root for root, kind in effects.items() if kind == DELETE)

        returned = set().union(*[self.roots_of(var) for var in body.outvars])
        self.temporaries = [root for root in self.origin if root not in self.deleted and root not in returned]

    def roots_of(self, var):
        if isinstance(var, Literal):
            return frozenset()
        return self.roots.get(var, frozenset())

    def allocations(self, before):
        return [root for root, i in self.origin.items() if i <= before]

    def split_index(self, name):
        temporaries = set(self.temporaries)
        split = None
        for i, effects in self.effects.items():
            allocates = any(self.origin.get(root) == i for root in temporaries)
            if allocates or any(effects.get(root, PERMEABLE) in (UNKNOWN, WRITE) for root in temporaries):
                split = i

        effects = self.effects[split]
        allocates = any(self.origin.get(root) == split for root in temporaries)
        if not allocates and not any(effects.get(root) == WRITE for root in temporaries):
            raise JaspUncomputationError(
                f"The function {name} applies {_describe(self.body.eqns[split])} to a temporary QuantumVariable, "
                "and jasp_uncompute can not determine whether it computes or uses the temporary."
                "\n\nIf it only uses the temporary, wrap it with gate_wrap and specify its permeability."
            )
        return split

    def check(self, split, name):
        temporaries = set(self.temporaries)
        allocated = set()
        compute_roots = set()

        for i, effects in self.effects.items():
            eqn = self.body.eqns[i]
            touched = set(effects) & temporaries

            if any(effects[root] == MEASURE for root in touched):
                raise JaspUncomputationError(
                    f"The function {name} measures a temporary QuantumVariable, which jasp_uncompute can not "
                    "uncompute.\n\nMeasure the temporary outside of the uncomputed function, or don't use "
                    "jasp_uncompute for it."
                )

            if i > split:
                for root in compute_roots - self.deleted:
                    if effects.get(root, PERMEABLE) != PERMEABLE:
                        raise JaspUncomputationError(
                            f"After computing its temporary QuantumVariables, the function {name} applies "
                            f"{_describe(eqn)}, which modifies qubits that the temporaries were computed from. "
                            "jasp_uncompute can then not restore the temporaries.\n\nOnly use the temporaries, "
                            "for example in phase gates or as controls, after they have been computed."
                        )
                continue

            if eqn.primitive.name == "jasp.measure":
                raise JaspUncomputationError(
                    f"The function {name} performs a measurement while computing its temporary QuantumVariables, "
                    "which can not be uncomputed.\n\nMove the measurement after the temporaries have been used."
                )

            allocates = {root for root in temporaries if self.origin.get(root) == i}
            writes = {root for root in touched if effects[root] in (UNKNOWN, WRITE)}
            if touched & allocated and not writes and not allocates:
                raise JaspUncomputationError(
                    f"The function {name} uses a temporary QuantumVariable in {_describe(eqn)} before all "
                    "temporaries have been computed. jasp_uncompute would revert this use together with the "
                    "computation.\n\nCompute all temporaries first, then use them."
                )
            allocated |= allocates

            for root, kind in effects.items():
                if root in temporaries or kind == DELETE:
                    continue
                if root not in self.inputs and root not in self.deleted and kind in (UNKNOWN, WRITE):
                    raise JaspUncomputationError(
                        f"The function {name} modifies a QuantumVariable it returns before all temporaries "
                        "have been computed. jasp_uncompute would revert this together with the computation."
                        "\n\nCompute the returned QuantumVariables after the temporaries."
                    )
                compute_roots.add(root)

    def use_measures(self, split):
        return any(
            self.body.eqns[i].primitive.name == "jasp.measure" or MEASURE in effects.values()
            for i, effects in self.effects.items()
            if i > split
        )

    def use_permeability(self, split, root):
        kinds = [effects.get(root, PERMEABLE) for i, effects in self.effects.items() if i > split]
        kind = max(kinds, default=PERMEABLE)
        if kind == PERMEABLE:
            return True
        if kind == WRITE:
            return False
        return None


def _eqn_effects(eqn, roots_of):
    """Return the effects of ``eqn`` (root -> kind) and the roots of its outvars."""
    in_roots = [roots_of(var) for var in eqn.invars]
    handler = _EFFECT_HANDLERS.get(eqn.primitive.name)
    if handler is not None:
        return handler(eqn, in_roots)

    # Classical equations and qubit array operations (get_qubit, slice, fuse, ...)
    quantum_in = frozenset().union(*[roots for var, roots in zip(eqn.invars, in_roots) if _is_quantum(var)])
    effects = {root: UNKNOWN for root in quantum_in} if _acts_on_state(eqn) else {}
    return effects, [quantum_in if _is_quantum(var) else frozenset() for var in eqn.outvars]


def _gate_effects(eqn, in_roots):
    gate = eqn.params["gate"]
    effects = {}
    for i in range(gate.num_qubits):
        permeable = gate.permeability.get(i)
        kind = UNKNOWN if permeable is None else PERMEABLE if permeable else WRITE
        _merge(effects, dict.fromkeys(in_roots[i], kind))
    return effects, _no_roots(eqn)


def _jit_effects(eqn, in_roots):
    sub_jaspr = eqn.params["jaxpr"]
    effects, out_roots = _analyze(sub_jaspr, in_roots)
    # The permeability specification of the whole block takes precedence
    specified = {}
    for var, roots in zip(_open(sub_jaspr).invars, in_roots):
        permeable = getattr(sub_jaspr, "permeability", {}).get(var)
        if permeable is not None:
            _merge(specified, dict.fromkeys(roots, PERMEABLE if permeable else WRITE))
    for root, kind in specified.items():
        if effects.get(root, PERMEABLE) <= WRITE:
            effects[root] = kind
    return effects, out_roots


def _cond_effects(eqn, in_roots):
    effects, out_roots = {}, _no_roots(eqn)
    for branch in eqn.params["branches"]:
        branch_effects, branch_roots = _analyze(branch, in_roots[1:])
        _merge(effects, branch_effects)
        out_roots = [a | b for a, b in zip(out_roots, branch_roots)]
    return effects, out_roots


def _while_effects(eqn, in_roots):
    cond_nconsts, body_nconsts = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
    consts = in_roots[cond_nconsts : cond_nconsts + body_nconsts]
    carry = in_roots[cond_nconsts + body_nconsts :]
    # Iterate until the carried roots are stable, since iterations may permute them
    for _ in range(len(carry) + 1):
        effects, out_roots = _analyze(eqn.params["body_jaxpr"], consts + carry)
        new_carry = [a | b for a, b in zip(carry, out_roots)]
        if new_carry == carry:
            break
        carry = new_carry
    return effects, carry


_EFFECT_HANDLERS = {
    "jasp.quantum_gate": _gate_effects,
    "jasp.create_qubits": lambda eqn, in_roots: ({}, [frozenset([eqn.outvars[0]]), frozenset()]),
    "jasp.delete_qubits": lambda eqn, in_roots: (dict.fromkeys(in_roots[0], DELETE), _no_roots(eqn)),
    "jasp.measure": lambda eqn, in_roots: (dict.fromkeys(in_roots[0], MEASURE), _no_roots(eqn)),
    "jit": _jit_effects,
    "cond": _cond_effects,
    "while": _while_effects,
}


def _no_roots(eqn):
    return [frozenset()] * len(eqn.outvars)


def _resolve_fresh(roots, var):
    """Replace the FRESH marker by ``var``, the qubit array that a block newly returns."""
    return (roots - {FRESH}) | {var} if FRESH in roots else roots


def _analyze(jaxpr, in_roots):
    """Return the effects of a sub-jaxpr on the given roots of its invars, and the roots of its outvars."""
    jaxpr = _open(jaxpr)
    roots = {var: r for var, r in zip(jaxpr.invars, in_roots) if r}
    outer = frozenset().union(*in_roots)

    def roots_of(var):
        if isinstance(var, Literal):
            return frozenset()
        return roots.get(var, frozenset())

    effects = {}
    for eqn in jaxpr.eqns:
        eqn_effects, out_roots = _eqn_effects(eqn, roots_of)
        _merge(effects, eqn_effects)
        for var, roots_from_eqn in zip(eqn.outvars, out_roots):
            var_roots = _resolve_fresh(roots_from_eqn, var)
            if var_roots:
                roots[var] = var_roots

    effects = {root: kind for root, kind in effects.items() if root in outer}
    out_roots = [frozenset(root if root in outer else FRESH for root in roots_of(var)) for var in jaxpr.outvars]
    return effects, out_roots


def _merge(effects, new_effects):
    for root, kind in new_effects.items():
        effects[root] = max(effects.get(root, PERMEABLE), kind)


def _creates_temporaries(eqn):
    """Return True if a cond or while equation allocates qubits that it neither deletes nor returns."""
    if eqn.primitive.name == "cond":
        bodies = eqn.params["branches"]
    else:
        bodies = [eqn.params["body_jaxpr"]]
    return any(_TopLevelAnalysis(_open(body)).temporaries for body in bodies)


def _is_injectable(jaxpr, var):
    """Return True if ``var`` is allocated by a create_qubits equation, possibly inside (nested) jit equations."""
    for eqn in _open(jaxpr).eqns:
        if var in eqn.outvars:
            if eqn.primitive.name == "jasp.create_qubits":
                return True
            if eqn.primitive.name == "jit":
                sub_jaxpr = _open(eqn.params["jaxpr"])
                return _is_injectable(sub_jaxpr, sub_jaxpr.outvars[eqn.outvars.index(var)])
            return False
    return False


def _sub_jaspr(body, invars, eqns, outvars):
    return Jaspr(
        constvars=list(body.constvars),
        invars=list(invars),
        outvars=list(outvars),
        eqns=list(eqns),
        consts=list(body.consts),
        debug_info=dummy_debug_info,
    )


def _open(jaxpr):
    """Return the Jaxpr of a ClosedJaxpr (including Jasprs)."""
    return jaxpr.jaxpr if isinstance(jaxpr, ClosedJaxpr) else jaxpr


def _acts_on_state(eqn):
    return any(isinstance(getattr(var, "aval", None), AbstractQuantumState) for var in eqn.invars)


def _is_quantum(var):
    return isinstance(getattr(var, "aval", None), (AbstractQubit, AbstractQubitArray))


def _as_tuple(res):
    if res is None:
        return ()
    return tuple(res) if isinstance(res, (tuple, list)) else (res,)


def _describe(eqn):
    if eqn.primitive.name == "jasp.quantum_gate":
        return f"the gate {eqn.params['gate'].name}"
    if eqn.primitive.name == "jit":
        return f"the function {eqn.params['name']}"
    if eqn.primitive.name == "cond":
        return "a classically controlled operation"
    if eqn.primitive.name == "while":
        # Gates applied to whole QuantumVariables are traced as loops
        body_eqns = _open(eqn.params["body_jaxpr"]).eqns
        gate_names = {e.params["gate"].name for e in body_eqns if e.primitive.name == "jasp.quantum_gate"}
        if len(gate_names) == 1:
            return f"the gate {gate_names.pop()}"
        return "a loop"
    return f"an operation ({eqn.primitive.name})"
