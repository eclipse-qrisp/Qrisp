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


def _apply_op(op, qubit_tracers, abs_qst):
    """Recursively inline a gate.

    - Primitive gates (op.definition is None) are bound directly via quantum_gate_p.
    - Composite gates are expanded by iterating their definition circuit and
      recursively calling _apply_op for each sub-instruction.
    """
    if op.definition is None:
        return quantum_gate_p.bind(*qubit_tracers, abs_qst, gate=op)

    defn = op.definition
    for instr in defn.data:
        qubit_indices = [defn.qubits.index(qb) for qb in instr.qubits]
        sub_qubits = [qubit_tracers[i] for i in qubit_indices]
        abs_qst = _apply_op(instr.op, sub_qubits, abs_qst)

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

        abs_qst = _apply_op(gate, qubit_tracers, abs_qst)
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
