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

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.extend.core import ClosedJaxpr
import pytest
import sympy

from qrisp import QuantumVariable, prepare, measure, x, h, cx, mcx, rxx, rzz, xxyy
from qrisp.jasp import make_jaspr, qache, quantum_kernel
from qrisp.jasp.interpreter_tools import decompose_composite_gates
from qrisp.jasp.primitives import quantum_gate_p


# ---------------------------------------------------------------------------
# Helper: walk all equations in a jaspr (top-level only; sub-jaxprs in jit/
# while/cond are inspected recursively) and collect gate objects.
# ---------------------------------------------------------------------------


def _collect_gates_from_jaxpr(jaxpr):
    """Yield every gate object found in quantum_gate_p equations, recursing
    into jit/while/cond sub-jaxprs."""

    for eqn in jaxpr.eqns:
        if eqn.primitive == quantum_gate_p:
            yield eqn.params["gate"]

        # Recurse into sub-jaxprs carried by control-flow primitives
        if eqn.primitive.name == "jit":
            sub = eqn.params["jaxpr"]
            inner = sub.jaxpr if isinstance(sub, ClosedJaxpr) else sub
            yield from _collect_gates_from_jaxpr(inner)

        elif eqn.primitive.name == "while":
            for key in ("body_jaxpr", "cond_jaxpr"):
                sub = eqn.params[key]
                inner = sub.jaxpr if isinstance(sub, ClosedJaxpr) else sub
                yield from _collect_gates_from_jaxpr(inner)

        elif eqn.primitive.name == "cond":
            for branch in eqn.params["branches"]:
                inner = branch.jaxpr if isinstance(branch, ClosedJaxpr) else branch
                yield from _collect_gates_from_jaxpr(inner)

        elif eqn.primitive.name == "scan":
            sub = eqn.params["jaxpr"]
            inner = sub.jaxpr if isinstance(sub, ClosedJaxpr) else sub
            yield from _collect_gates_from_jaxpr(inner)


def assert_composite_gates(jaspr):
    """Assert that the jaspr contains at least one composite gate (gate.definition is not None)."""
    all_gates = list(_collect_gates_from_jaxpr(jaspr))
    assert any(g.definition is not None for g in all_gates), (
        "Expected at least one composite gate in the original jaspr."
    )


def assert_no_composite_gates(jaspr):
    """Assert that all gates in the jaspr are primitive (definition is None)."""
    for gate in _collect_gates_from_jaxpr(jaspr):
        assert gate.definition is None, (
            f"Gate '{gate.name}' still has a composite definition after decomposition."
        )


def assert_primitive_gates_invars_match_abstract_params(jaspr):
    """Assert that every primitive gate with abstract_params has the correct number
    of invars corresponding to those abstract_params."""
    for eqn in jaspr.eqns:
        if eqn.primitive == quantum_gate_p:
            gate = eqn.params["gate"]
            if gate.abstract_params:
                num_param_invars = len(eqn.invars) - gate.num_qubits - 1
                assert num_param_invars == len(gate.abstract_params)


def assert_same_distribution(jaspr_a, jaspr_b):
    """Assert that two jasprs produce identical exact probability distributions
    via to_qc().run() (no shots), compared with np.allclose."""
    _, qc_a = jaspr_a.to_qc()
    _, qc_b = jaspr_b.to_qc()
    probs_a = qc_a.run()
    probs_b = qc_b.run()
    assert set(probs_a.keys()) == set(probs_b.keys()), (
        f"Outcome sets differ: {set(probs_a.keys())} vs {set(probs_b.keys())}"
    )
    keys = sorted(probs_a.keys())
    assert np.allclose(
        [probs_a[k] for k in keys],
        [probs_b[k] for k in keys],
    ), (
        f"Probability distributions differ:\n  original:   {probs_a}\n  decomposed: {probs_b}"
    )


def assert_same_unitary(jaspr_a, jaspr_b):
    """Assert that two jasprs produce identical unitary matrices."""
    qc_a = jaspr_a.to_qc()[-1]
    qc_b = jaspr_b.to_qc()[-1]
    u_a = qc_a.get_unitary()
    u_b = qc_b.get_unitary()
    assert np.allclose(u_a, u_b, atol=1e-6), (
        f"Unitary matrices differ:\n  original:   {u_a}\n  decomposed: {u_b}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_decompose_state_preparation():
    """prepare() inserts a state_init composite gate.  After decomposition,
    no composite gates should remain and the output distribution should be
    unchanged."""

    def main():
        qv = QuantumVariable(2)
        prepare(qv, np.array([0.2, 0.4, 0.7, 0.5]))
        return measure(qv)

    jaspr = make_jaspr(main)()

    # Verify that the original jaspr contains at least one composite gate
    assert_composite_gates(jaspr)

    decomposed = decompose_composite_gates(jaspr)

    assert_no_composite_gates(decomposed)
    assert_same_distribution(jaspr, decomposed)


def test_decompose_mcx():
    """mcx with control_amount > 1 produces a composite gate.  After
    decomposition the circuit must contain only primitive gates and give
    the correct deterministic measurement outcome."""

    def main():
        qv = QuantumVariable(4)
        x(qv[0])
        x(qv[1])
        x(qv[2])
        mcx([qv[0], qv[1], qv[2]], qv[3])
        return measure(qv)

    jaspr = make_jaspr(main)()

    # Sanity-check: original has composite gates
    assert_composite_gates(jaspr)

    decomposed = decompose_composite_gates(jaspr)

    assert_no_composite_gates(decomposed)
    assert_same_distribution(jaspr, decomposed)


def test_decompose_cx_is_noop():
    """cx is already a primitive gate (definition is None).  The decomposition
    pass must leave it untouched and the circuit must still be correct."""

    def main():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        return measure(qv)

    jaspr = make_jaspr(main)()

    decomposed = decompose_composite_gates(jaspr)

    assert_no_composite_gates(decomposed)
    assert_same_distribution(jaspr, decomposed)


def test_decompose_nested_jaspr():
    """BlockEncoding introduces deeply nested sub-jasprs containing composite
    gates.  The decomposition must recurse into jit/while/cond sub-jaxprs."""

    from qrisp.operators import X, Y, Z
    from qrisp.block_encodings import BlockEncoding

    H_op = X(0) * X(1) + Y(0) * Y(1) + Z(0) * Z(1)
    BE = BlockEncoding.from_operator(H_op)

    def main():
        qv = QuantumVariable(2)
        ancs = BE.apply(qv)
        return measure(qv)

    jaspr = make_jaspr(main)()
    decomposed = decompose_composite_gates(jaspr)

    assert_no_composite_gates(decomposed)

    # The jaspr must still be executable
    result = decomposed()
    assert result is not None


def test_call_graph_compression_preserved():
    """When the same @qache function is called twice, both jit equations in the
    original jaspr share the exact same ClosedJaxpr object (call-graph compression).
    After decompose_composite_gates, the two jit equations in the decomposed jaspr
    must still reference the same ClosedJaxpr object — not two distinct copies."""

    @qache
    def test(qv):
        prepare(qv, np.array([0.2, 0.4, 0.7, 0.5]))

    def main():
        qv = QuantumVariable(2)
        test(qv)
        test(qv)
        return measure(qv)

    jaspr = make_jaspr(main)()

    # Collect the ClosedJaxpr objects from all top-level jit equations
    def _jit_jaxprs(jaxpr):
        return [
            eqn.params["jaxpr"] for eqn in jaxpr.eqns if eqn.primitive.name == "jit"
        ]

    orig_jaxprs = _jit_jaxprs(jaspr)
    assert len(orig_jaxprs) >= 2
    # Verify that the original already shares the same object
    assert orig_jaxprs[0] is orig_jaxprs[1]

    decomposed = decompose_composite_gates(jaspr)

    decomp_jaxprs = _jit_jaxprs(decomposed)
    assert len(decomp_jaxprs) >= 2
    assert decomp_jaxprs[0] is decomp_jaxprs[1]

    # Correctness: decomposed jaspr must still produce the same distribution
    assert_same_distribution(jaspr, decomposed)


def test_decompose_scan_body():
    """Qrisp code called inside a jax.lax.scan body must also be decomposed.
    A @quantum_kernel using prepare() introduces a composite gate inside the
    scan body jaxpr.  decompose_composite_gates must recurse into it."""

    @quantum_kernel
    def random_number():
        qv = QuantumVariable(2)
        prepare(qv, np.array([0.3, 0.7, 0.2, 0.8]))
        return measure(qv)

    def main():
        def scan_body(carry, x):
            rnd = random_number()
            return carry + x + rnd, carry + x + rnd

        xs = jnp.array([1, 2, 3, 4, 5])
        final_carry, cumulative_sums = lax.scan(scan_body, 0, xs)
        return final_carry, cumulative_sums

    jaspr = make_jaspr(main)()

    # Confirm a scan equation is present at the top level
    scan_eqns = [e for e in jaspr.eqns if e.primitive.name == "scan"]
    assert len(scan_eqns) >= 1

    # Confirm the scan body contains composite gates before decomposition
    all_gates = list(_collect_gates_from_jaxpr(jaspr))
    assert any(g.definition is not None for g in all_gates)

    decomposed = decompose_composite_gates(jaspr)

    # After decomposition no composite gates must remain anywhere, including
    # inside the scan body
    assert_no_composite_gates(decomposed)

    # The decomposed jaspr must still be executable and return sensible values
    result = decomposed()
    assert result is not None


def test_decompose_constant_param_subgates():
    """Composite gates whose internal sub-gates carry plain numeric constants
    as params (empty abstract_params, non-empty params) must have those
    constants promoted to explicit JAX literal invars after decomposition.

    xxyy(phi, beta) is the canonical example: its definition includes sub-gates
    such as rz(-π/2) whose angle is a fixed Python float — not derived from phi
    or beta.  Before the fix those values were embedded in the gate object and
    not passed as invars, so the MLIR lowering received a gate without its
    expected angle input variable.

    Invariant checked here (distinct from test_decompose_parametrized_xxyy):
    after decomposition every gate's params list must contain only sympy
    expressions (symbols or symbolic expressions), never raw Python numerics.
    A raw float/int in gate.params means the angle is still embedded rather
    than lifted to an invar.
    """

    def circuit(phi, beta):
        qv = QuantumVariable(2)
        xxyy(phi, beta, qv[0], qv[1])
        return qv

    jaspr = make_jaspr(circuit)(0.5, 0.4)

    # Sanity-check: at least one sub-gate with constant (non-symbolic) params
    # exists in the original composite definition so the test is meaningful.
    found_const_param_gate = False
    for gate in _collect_gates_from_jaxpr(jaspr):
        if gate.definition is not None:
            for instr in gate.definition.data:
                sub = instr.op
                if sub.params and not sub.abstract_params:
                    found_const_param_gate = True
                    break
    assert found_const_param_gate, (
        "Expected at least one sub-gate with constant params inside the "
        "composite gate definition; the test may need updating."
    )

    decomposed = decompose_composite_gates(jaspr)
    assert_no_composite_gates(decomposed)

    # Core assertion: no gate may retain a raw Python numeric in its params
    # list.  All parameter values must be sympy expressions so that the MLIR
    # lowering can find them as explicit input variables.
    for eqn in decomposed.eqns:
        if eqn.primitive == quantum_gate_p:
            gate = eqn.params["gate"]
            for param in gate.params:
                assert isinstance(param, sympy.Basic), (
                    f"Gate '{gate.name}' has a raw numeric param {param!r} "
                    f"({type(param).__name__}) embedded in the gate object "
                    f"after decomposition.  It must be a sympy expression so "
                    f"that every angle is an explicit invar for MLIR lowering."
                )

    # Unitary check: use hardcoded params so to_qc() needs no arguments.
    def circuit_fixed():
        qv = QuantumVariable(2)
        xxyy(0.5, 0.4, qv[0], qv[1])
        return qv

    jaspr_fixed = make_jaspr(circuit_fixed)()
    decomposed_fixed = decompose_composite_gates(jaspr_fixed)
    assert_same_unitary(jaspr_fixed, decomposed_fixed)


@pytest.mark.parametrize("gate", [rxx, rzz])
def test_decompose_parametrized_rxx_rzz(gate):
    """rxx and rzz are composite gates whose sub-gates (gphase, p) carry parametrized
    expressions (-phi/2 and phi respectively).  Before the fix, the param
    tracers were silently dropped during decomposition, leaving those sub-gates
    with no dynamic parameter invars.

    After the fix:
    * Every primitive gate that has abstract_params must have exactly
      len(abstract_params) parameter invars in the decomposed jaspr.
    * Evaluating the decomposed jaspr must yield the same unitary as the original.
    """

    def circuit(phi):
        qv = QuantumVariable(2)
        gate(phi, qv[0], qv[1])
        return qv

    jaspr = make_jaspr(circuit)(0.5)
    decomposed = decompose_composite_gates(jaspr)

    # --- structural check ---
    assert_composite_gates(jaspr)
    assert_no_composite_gates(decomposed)
    assert_primitive_gates_invars_match_abstract_params(decomposed)

    # --- numerical check ---
    def circuit_fixed():
        qv = QuantumVariable(2)
        gate(0.5, qv[0], qv[1])
        return qv

    jaspr_fixed = make_jaspr(circuit_fixed)()
    decomposed_fixed = decompose_composite_gates(jaspr_fixed)
    assert_same_unitary(jaspr_fixed, decomposed_fixed)


def test_decompose_parametrized_xxyy():
    """xxyy is a composite gate parameterized by two angles (phi and beta) whose
    sub-gates carry expressions derived from both parameters.  This test verifies
    the same two invariants as test_decompose_parametrized_rxx for the two-parameter
    case:

    * Every primitive gate that has abstract_params must have exactly
      len(abstract_params) parameter invars in the decomposed jaspr.
    * Evaluating the decomposed jaspr must yield the same unitary as the original.
    """

    def circuit(phi, beta):
        qv = QuantumVariable(2)
        xxyy(phi, beta, qv[0], qv[1])
        return qv

    jaspr = make_jaspr(circuit)(0.5, 0.4)
    decomposed = decompose_composite_gates(jaspr)

    # --- structural check ---
    assert_composite_gates(jaspr)
    assert_no_composite_gates(decomposed)
    assert_primitive_gates_invars_match_abstract_params(decomposed)

    # --- numerical check ---
    def circuit_fixed():
        qv = QuantumVariable(2)
        h(qv)
        xxyy(1.1, 0.4, qv[0], qv[1])
        h(qv)
        return qv

    jaspr_fixed = make_jaspr(circuit_fixed)()
    decomposed_fixed = decompose_composite_gates(jaspr_fixed)
    assert_same_unitary(jaspr_fixed, decomposed_fixed)
