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

from jax.extend.core import JaxprEqn

from qrisp.jasp.interpreter_tools import exec_eqn, reinterpret, extract_invalues, insert_outvalues
from qrisp.jasp.primitives import quantum_gate_p
from qrisp.jasp.primitives.operation_primitive import greek_letters as _greek_letters

# quantum_gate_p.append_impl binds runtime parameter slot i to greek_letters[i].
# When a decomposed primitive gate uses only a subset of the parent symbols
# (for example a sub-gate depending only on beta), we must first recover the
# symbols in canonical greek_letters order and then remap them to a dense
# prefix alpha, beta, ... before binding the tracers positionally.
_greek_letter_order = {sym: i for i, sym in enumerate(_greek_letters)}


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
            # Recover the gate's free symbols in the same canonical order used
            # when traced operations are constructed: alpha, beta, gamma, ...
            # The order matters because append_impl later substitutes the
            # incoming parameter tracers positionally into greek_letters[i].
            sorted_syms = sorted(
                op.abstract_params,
                key=lambda s: _greek_letter_order.get(s, len(_greek_letters)),
            )
            # Re-index the gate onto a dense greek-letter prefix. This is the
            # critical step for sparse subsets such as {beta}: we normalize
            # beta -> alpha so that passing a single tracer still lands in slot 0.
            # The symbolic expressions themselves stay intact under this rename,
            # e.g. rz(beta) becomes rz(alpha) and -beta/2 becomes -alpha/2.
            remap = {sym: _greek_letters[i] for i, sym in enumerate(sorted_syms)}
            normalized_op = op.bind_parameters(remap)
            # Pass one tracer per normalized symbol in the same positional order.
            # We intentionally forward the raw tracers here; append_impl and
            # bind_parameters still need to evaluate the gate's sympy expression
            # (for example -alpha/2) at concrete execution time.
            param_tracer_list = [param_dict[sym] for sym in sorted_syms]
            return quantum_gate_p.bind(
                *qubit_tracers, *param_tracer_list, abs_qst, gate=normalized_op
            )
        else:
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
        new_eqn.params["branches"] = tuple(
            _decompose_sub_jaxpr(branch) for branch in eqn.params["branches"]
        )
        exec_eqn(new_eqn, context_dic)
        return False

    elif eqn.primitive.name == "scan":
        new_eqn = _copy_eqn(eqn)
        new_eqn.params["jaxpr"] = _decompose_sub_jaxpr(eqn.params["jaxpr"])
        exec_eqn(new_eqn, context_dic)
        return False

    return True  # fall back to default execution


@lru_cache(maxsize=int(1e5))
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


@lru_cache(maxsize=int(1e5))
def decompose_composite_gates(jaspr):
    """Return a new Jaspr with all composite (non-primitive) gates recursively inlined."""
    from qrisp.jasp import Jaspr
    return Jaspr(reinterpret(jaspr, decompose_eqn_evaluator))
