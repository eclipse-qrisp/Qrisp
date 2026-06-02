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
from qrisp import *
from qrisp.jasp import make_jaspr
from qrisp.jasp.interpreter_tools import decompose_composite_gates
from qrisp.jasp.primitives import quantum_gate_p


# ---------------------------------------------------------------------------
# Helper: walk all equations in a jaspr (top-level only; sub-jaxprs in jit/
# while/cond are inspected recursively) and collect gate objects.
# ---------------------------------------------------------------------------

def _collect_gates_from_jaxpr(jaxpr):
    """Yield every gate object found in quantum_gate_p equations, recursing
    into jit/while/cond sub-jaxprs."""
    from jax.extend.core import ClosedJaxpr
    from qrisp.jasp import Jaspr

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


def assert_no_composite_gates(jaspr):
    """Assert that all gates in the jaspr are primitive (definition is None)."""
    for gate in _collect_gates_from_jaxpr(jaspr):
        assert gate.definition is None, (
            f"Gate '{gate.name}' still has a composite definition after decomposition."
        )


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
    ), f"Probability distributions differ:\n  original:   {probs_a}\n  decomposed: {probs_b}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_decompose_state_preparation():
    """prepare() inserts a state_init composite gate.  After decomposition,
    no composite gates should remain and the output distribution should be
    unchanged."""

    def main():
        qv = QuantumFloat(2)
        prepare(qv, np.array([0.2, 0.4, 0.7, 0.5]))
        return measure(qv)

    jaspr = make_jaspr(main)()

    # Verify that the original jaspr contains at least one composite gate
    all_gates = list(_collect_gates_from_jaxpr(jaspr))
    assert any(g.definition is not None for g in all_gates), (
        "Expected at least one composite gate in the original jaspr."
    )

    decomposed = decompose_composite_gates(jaspr)

    assert_no_composite_gates(decomposed)
    assert_same_distribution(jaspr, decomposed)


def test_decompose_mcx():
    """mcx with control_amount > 1 produces a composite gate.  After
    decomposition the circuit must contain only primitive gates and give
    the correct deterministic measurement outcome."""

    def main():
        qv = QuantumFloat(4)
        x(qv[0])
        x(qv[1])
        x(qv[2])
        mcx([qv[0], qv[1], qv[2]], qv[3])
        return measure(qv)

    jaspr = make_jaspr(main)()

    # Sanity-check: original has composite gates
    all_gates = list(_collect_gates_from_jaxpr(jaspr))
    assert any(g.definition is not None for g in all_gates)

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
        qv = QuantumFloat(2)
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
        qv = QuantumFloat(2)
        test(qv)
        test(qv)
        return measure(qv)

    jaspr = make_jaspr(main)()

    # Collect the ClosedJaxpr objects from all top-level jit equations
    def _jit_jaxprs(jaxpr):
        return [
            eqn.params["jaxpr"]
            for eqn in jaxpr.eqns
            if eqn.primitive.name == "jit"
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
        qv = QuantumFloat(2)
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
