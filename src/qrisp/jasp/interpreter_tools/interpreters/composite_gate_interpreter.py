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

from functools import lru_cache

from qrisp._cache_config import qrisp_lru_compilation_cache
from jax.extend.core import JaxprEqn
import jax.numpy as jnp
from sympy import lambdify as _lambdify

from qrisp.jasp.interpreter_tools import exec_eqn, reinterpret, extract_invalues, insert_outvalues
from qrisp.jasp.primitives import quantum_gate_p
from qrisp.jasp.primitives.operation_primitive import greek_letters as _greek_letters
from qrisp.circuit.operation import U3Gate as _U3Gate
import qrisp.circuit.standard_operations as _std_ops

_greek_letter_order = {sym: i for i, sym in enumerate(_greek_letters)}

# Gate factories that produce a fresh U3Gate-derived gate whose params are
# exactly [alpha] (identity), so that append_impl and other consumers can call
# bind_parameters({alpha: val}) and get back a fully concrete gate with params=[val].
_U3_IDENTITY_FACTORIES = {
    "gphase": lambda: _std_ops.GPhaseGate(_greek_letters[0]),
    "rx": lambda: _std_ops.RXGate(_greek_letters[0]),
    "ry": lambda: _std_ops.RYGate(_greek_letters[0]),
    "rz": lambda: _std_ops.RZGate(_greek_letters[0]),
    "p": lambda: _std_ops.PGate(_greek_letters[0]),
    "u1": lambda: _std_ops.U1Gate(_greek_letters[0]),
    "u3": lambda: _U3Gate(_greek_letters[0], _greek_letters[1], _greek_letters[2]),
}


def _make_identity_param_op(op):
    """Return a version of op whose params are [alpha, beta, ...] (identity expressions).

    Two cases require different treatment:

    U3Gate instances (gphase, rx, ry, rz, p, ...):
        U3Gate.bind_parameters evaluates its internal fields (theta, phi, lam,
        global_phase) directly — it does NOT read self.params.  Patching params
        on a copy therefore has no effect.  Instead, a fresh gate must be
        constructed via its factory function so that the internal fields carry
        the identity symbolic structure (e.g. GPhaseGate(alpha) has
        global_phase=alpha, not -alpha/2).

    Generic Operation instances:
        Operation.bind_parameters lambdifies over self.params, so patching
        params=[alpha, beta, ...] and clearing the lambdify cache is sufficient.
    """
    n = len(op.params)
    if not n:
        return op
    if isinstance(op, _U3Gate):
        factory = _U3_IDENTITY_FACTORIES.get(op.name)
        if factory is None:
            raise NotImplementedError(
                f"_make_identity_param_op: no identity factory registered for "
                f"U3Gate '{op.name}'.  Add an entry to _U3_IDENTITY_FACTORIES."
            )
        return factory()
    # Generic Operation fallback: patch params and abstract_params on a copy.
    identity = op.copy()
    identity.params = list(_greek_letters[:n])
    identity.abstract_params = set(_greek_letters[:n])
    if hasattr(identity, "lambdified_params"):
        del identity.lambdified_params
    return identity


def _copy_eqn(eqn):
    return JaxprEqn(
        primitive=eqn.primitive,
        invars=list(eqn.invars),
        outvars=list(eqn.outvars),
        params=dict(eqn.params),
        source_info=eqn.source_info,
        effects=eqn.effects,
        ctx=eqn.ctx,
    )


def _apply_op(op, qubit_tracers, abs_qst, param_dict=None):
    """Recursively inline a gate.

    - Primitive gates (op.definition is None) are bound directly via quantum_gate_p.
    - Composite gates are expanded by iterating their definition circuit and
      recursively calling _apply_op for each sub-instruction.

    param_dict maps the sympy symbols that appear in sub-gate param expressions
    (i.e. the parent gate's abstract params) to their corresponding JAX tracers.
    """
    if param_dict is None:
        param_dict = {}

    if op.definition is None:
        if op.abstract_params and param_dict:
            # op.abstract_params is a set, so iteration order is non-deterministic.
            # lambdify maps positional arguments to symbols in the order given, so
            # the symbol list passed to lambdify and the tracer list must agree on
            # the same ordering.  We sort by _greek_letter_order (the canonical
            # index of each symbol in the greek_letters sequence) to get a stable,
            # deterministic ordering regardless of Python's set hash randomisation.
            sorted_syms = sorted(op.abstract_params, key=lambda s: _greek_letter_order[s])
            sorted_tracers = [param_dict[sym] for sym in sorted_syms]
            # Evaluate each symbolic parameter expression (e.g. -alpha/2) against
            # the parent gate's symbol->tracer bindings to produce a concrete JAX
            # tracer for each parameter slot.
            computed_tracers = [_lambdify(sorted_syms, expr, modules="jax")(*sorted_tracers) for expr in op.params]
            # Emit an identity-param version of the gate (params = [alpha, beta, ...])
            # together with the pre-computed tracers.  This is required because
            # append_impl later calls gate.bind_parameters({alpha: val}), which
            # must map val -> val (identity).  Using the original gate (e.g.
            # gphase(-alpha/2)) would cause the expression to be applied a second
            # time: bind_parameters({alpha: val}) would yield -val/2 instead of val.
            identity_op = _make_identity_param_op(op)
            return quantum_gate_p.bind(*qubit_tracers, *computed_tracers, abs_qst, gate=identity_op)
        if op.params:
            # Gate has only constant (non-symbolic) parameters — e.g. rz(-π/2)
            # emitted as an internal fixed rotation inside a composite gate
            # definition.  Without this branch the parameter value stays embedded
            # in the gate object and no param invar is emitted.  The MLIR lowering
            # expects every parametrised gate to have its angle(s) as explicit
            # input variables, so we promote each constant to a JAX float64 literal
            # tracer and emit an identity-param gate, exactly as in the symbolic
            # branch above.
            const_tracers = [jnp.float64(float(p)) for p in op.params]
            identity_op = _make_identity_param_op(op)
            return quantum_gate_p.bind(*qubit_tracers, *const_tracers, abs_qst, gate=identity_op)

        return quantum_gate_p.bind(*qubit_tracers, abs_qst, gate=op)

    defn = op.definition.transpile()
    for instr in defn.data:
        qubit_indices = [defn.qubits.index(qb) for qb in instr.qubits]
        sub_qubits = [qubit_tracers[i] for i in qubit_indices]
        abs_qst = _apply_op(instr.op, sub_qubits, abs_qst, param_dict=param_dict)

    return abs_qst


def decompose_eqn_evaluator(eqn, context_dic):
    """Equation evaluator that recursively decomposes composite quantum gates.

    Whenever a quantum_gate_p equation carries a composite gate (one that has
    a circuit definition), the gate is expanded in-place.

    For jit/while/cond equations whose sub-jaxprs may themselves contain
    composite gates, the sub-jaxprs are recursively transformed before the
    equation is re-executed.

    All other equations are passed through unchanged.
    """
    if eqn.primitive == quantum_gate_p and eqn.params["gate"].definition is not None:
        gate = eqn.params["gate"]
        invalues = extract_invalues(eqn, context_dic)
        qubit_tracers = invalues[: gate.num_qubits]
        abs_qst = invalues[-1]
        param_tracers = invalues[gate.num_qubits : -1]

        # Preserve the parent gate's symbol -> tracer association so primitive
        # sub-gates can recover the correct tracer even when they reference only
        # a subset of the parent's parameters.
        param_dict = {gate.params[i]: param_tracers[i] for i in range(len(param_tracers))}

        abs_qst = _apply_op(gate, qubit_tracers, abs_qst, param_dict=param_dict)
        insert_outvalues(eqn, context_dic, abs_qst)
        return False  # equation handled; skip default exec_eqn

    elif eqn.primitive.name == "jit":
        new_eqn = _copy_eqn(eqn)
        new_eqn.params["jaxpr"] = _decompose_sub_jaxpr(eqn.params["jaxpr"])
        exec_eqn(new_eqn, context_dic)
        return False

    elif eqn.primitive.name == "while":
        new_eqn = _copy_eqn(eqn)
        new_eqn.params["body_jaxpr"] = _decompose_sub_jaxpr(eqn.params["body_jaxpr"])
        new_eqn.params["cond_jaxpr"] = _decompose_sub_jaxpr(eqn.params["cond_jaxpr"])
        exec_eqn(new_eqn, context_dic)
        return False

    elif eqn.primitive.name == "cond":
        new_eqn = _copy_eqn(eqn)
        new_eqn.params["branches"] = tuple(_decompose_sub_jaxpr(branch) for branch in eqn.params["branches"])
        exec_eqn(new_eqn, context_dic)
        return False

    elif eqn.primitive.name == "scan":
        new_eqn = _copy_eqn(eqn)
        new_eqn.params["jaxpr"] = _decompose_sub_jaxpr(eqn.params["jaxpr"])
        exec_eqn(new_eqn, context_dic)
        return False

    return True  # fall back to default execution


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def _decompose_sub_jaxpr(jaxpr):
    """Apply decompose_composite_gates to a sub-jaxpr (ClosedJaxpr or Jaspr)."""
    from qrisp.jasp import Jaspr
    from jax.extend.core import ClosedJaxpr

    if isinstance(jaxpr, Jaspr):
        return decompose_composite_gates(jaxpr)
    elif isinstance(jaxpr, ClosedJaxpr):
        inner = reinterpret(jaxpr.jaxpr, decompose_eqn_evaluator)
        return ClosedJaxpr(inner, jaxpr.consts)
    else:
        return jaxpr


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def decompose_composite_gates(jaspr):
    """Return a new Jaspr with all composite (non-primitive) gates recursively inlined."""
    from qrisp.jasp import Jaspr

    return Jaspr(reinterpret(jaspr, decompose_eqn_evaluator))
