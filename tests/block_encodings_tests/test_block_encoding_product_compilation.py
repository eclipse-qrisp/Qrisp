"""********************************************************************************
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

"""Compilation-cost regression tests for ProductBlockEncoding.

The companion file test_block_encoding_lcu_compilation.py covers the linear
combination. A product has no trace amplification of its own, since it composes
its factors directly rather than through ``q_switch``, so the concern here is
narrower: the derived unitary, its per-factor steps, the ancilla layouts and the
normalization must be built once and reused.

That matters in two places. Applying the same product twice, or reusing a factor
in more than one product, should trace the factor once rather than once per use.
And a product nested inside a linear combination hands its unitary to that
combination's SELECT branch, so a product that rebuilds its unitary on every
access would stop the combination from caching anything.

Numerical behaviour is covered by test_block_encoding_arithmetic.py.
"""

import numpy as np
import pytest

from qrisp import QuantumFloat, QuantumVariable, make_jaspr, terminal_sampling, x
from qrisp.jasp import count_ops, depth, num_qubits
from qrisp.block_encodings import BlockEncoding, ProductBlockEncoding
from qrisp.operators import X, Y, Z

STRATEGIES = ["qubit_efficient", "separate"]


def _trace(encoding, operand_size=3):
    """Trace ``encoding.apply`` on a fresh operand and return the resulting Jaspr."""

    def circuit():
        operand = QuantumFloat(operand_size)
        encoding.apply(operand)
        return operand

    return make_jaspr(circuit)()


def _counting_encoding(executions, qubit=0, num_ancillas=0):
    """Build a block encoding whose unitary records every Python-level execution."""
    templates = [QuantumFloat(1).template() for _ in range(num_ancillas)]

    def unitary(*args):
        executions.append(None)
        x(args[-1][qubit])

    return BlockEncoding(1, templates, unitary)


#
# Derived state is reused
#


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_product_derived_state_is_cached(strategy):
    """The unitary, its steps, layouts and templates must be built exactly once.

    Jasp's caches are keyed on object identity, so a product that hands out a fresh
    unitary on every access silently disables them for itself and for any composite
    holding it.
    """
    product = ProductBlockEncoding(
        [BlockEncoding.from_eye(0), BlockEncoding.from_eye(1)],
        strategy=strategy,
    )

    assert product.unitary is product.unitary
    assert product._has_reusable_unitary
    assert all(first is second for first, second in zip(product._anc_templates, product._anc_templates))
    assert product.alpha == product.alpha


def test_product_ancilla_templates_are_not_shared_mutable_state():
    """Callers may concatenate the returned template list without corrupting the cache."""
    product = BlockEncoding.from_eye(0) @ BlockEncoding.from_eye(1)

    templates = product._anc_templates
    templates.append(None)

    assert len(product._anc_templates) == len(templates) - 1


def test_plain_block_encoding_reports_a_reusable_unitary():
    """A plain block encoding stores its unitary in a field, so it is always reusable."""
    assert BlockEncoding.from_eye(0)._has_reusable_unitary


#
# Tracing cost does not grow with reuse or with the number of factors
#


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_repeated_application_traces_each_factor_once(strategy):
    """Applying the same product twice must not trace its factors twice."""
    executions = []
    product = ProductBlockEncoding(
        [_counting_encoding(executions), BlockEncoding(1, [], lambda operand: x(operand[1]))],
        strategy=strategy,
    )

    def circuit():
        operand = QuantumFloat(3)
        product.apply(operand)
        product.apply(operand)
        return operand

    make_jaspr(circuit)()

    assert len(executions) == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_product_chain_traces_each_factor_once(strategy):
    """A longer product must not re-trace the factors it already composed."""
    trace_counts = []
    for num_factors in (2, 4, 8):
        executions = []
        factors = [_counting_encoding(executions)]
        factors += [BlockEncoding(1, [], lambda operand: x(operand[1])) for _ in range(num_factors - 1)]
        _trace(ProductBlockEncoding(factors, strategy=strategy))
        trace_counts.append(len(executions))

    assert trace_counts == [1, 1, 1], f"tracing cost grows with the factor count: {trace_counts}"


def test_product_nested_in_linear_combination_does_not_amplify_tracing():
    """Alternating products and linear combinations must not multiply the tracing cost.

    This is the shape a product of sums compiles to. The linear combination traces
    each SELECT branch several times, so a product that rebuilt its unitary per
    access would pay for the whole subtree on every one of those.
    """
    trace_counts = []
    for nesting_depth in (1, 2, 3):
        executions = []
        encoding = _counting_encoding(executions)
        for _ in range(nesting_depth):
            encoding = BlockEncoding.from_eye() + (encoding @ BlockEncoding(1, [], lambda operand: x(operand[1])))
        _trace(encoding)
        trace_counts.append(len(executions))

    assert len(set(trace_counts)) == 1, f"tracing cost grows with nesting depth: {trace_counts}"
    assert trace_counts[0] <= 2, f"innermost factor traced {trace_counts[0]} times"


def test_shared_factor_is_traced_once_across_products():
    """A factor reused by two different products must be traced once, not twice.

    This is the ``H.dagger() @ P @ H`` shape, where the same expensive encoding
    appears more than once in the flattened factor list.
    """
    executions = []
    shared = _counting_encoding(executions)
    other = BlockEncoding(1, [], lambda operand: x(operand[1]))

    _trace(shared @ other @ shared)

    assert len(executions) == 1


#
# Derived quantities stay concrete
#


def test_product_alpha_stays_concrete_during_tracing():
    """A concrete normalization must not be turned into a tracer by the derivation."""
    product = BlockEncoding(2.0, [], lambda operand: None) @ BlockEncoding(3.0, [], lambda operand: None)
    observed = {}

    def circuit():
        operand = QuantumFloat(3)
        observed["alpha"] = product.alpha
        product.apply(operand)
        return operand

    make_jaspr(circuit)()

    import jax

    assert not isinstance(observed["alpha"], jax.core.Tracer)
    assert np.isclose(observed["alpha"], 6.0)


#
# The restructuring did not change what is encoded
#


@pytest.mark.parametrize(
    "operators",
    [
        [X(0) + 0.4 * Z(0), Y(0) + 0.3 * Z(0)],
        [X(0) * X(1) + 0.2 * Z(0), Y(0) * Y(1) + 0.3 * X(1), Z(0) + 0.1 * Y(1)],
    ],
)
def test_cached_product_matches_separate_strategy(operators):
    """Both strategies must still encode the same operator after the caching rework."""
    factors = [BlockEncoding.from_operator(operator) for operator in operators]
    qubit_amount = max(operator.find_minimal_qubit_amount() for operator in operators)

    @terminal_sampling
    def main(block_encoding):
        return block_encoding.apply_rus(lambda: QuantumVariable(qubit_amount))()

    separate = main(ProductBlockEncoding(factors, strategy="separate"))
    qubit_efficient = main(ProductBlockEncoding(factors, strategy="qubit_efficient"))

    for state in range(2**qubit_amount):
        assert np.isclose(separate.get(state, 0), qubit_efficient.get(state, 0))


def test_product_of_linear_combinations_is_numerically_unchanged():
    """A product whose factors are sums must encode the product of those sums."""
    H1, H2 = X(0) + 0.5 * Z(0), Y(0) + 0.25 * Z(0)

    composite = (BlockEncoding.from_operator(X(0)) + 0.5 * BlockEncoding.from_operator(Z(0))) @ (
        BlockEncoding.from_operator(Y(0)) + 0.25 * BlockEncoding.from_operator(Z(0))
    )
    reference = BlockEncoding.from_operator(H1) @ BlockEncoding.from_operator(H2)

    @terminal_sampling
    def main(block_encoding):
        return block_encoding.apply_rus(lambda: QuantumVariable(1))()

    composite_result = main(composite)
    reference_result = main(reference)

    for state in range(2):
        assert np.isclose(composite_result.get(state, 0), reference_result.get(state, 0))


#
# Cached state must not outlive the trace that built it
#


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "name, build",
    [
        ("two factors", lambda: [BlockEncoding.from_eye(0), BlockEncoding.from_eye(1)]),
        ("single factor", lambda: [BlockEncoding.from_eye(1)]),
        (
            "factors with ancillas",
            lambda: [
                BlockEncoding(1, [QuantumFloat(2)], lambda a, o: x(o[0])),
                BlockEncoding(1, [QuantumFloat(1)], lambda a, o: x(o[1])),
            ],
        ),
        (
            "factors that are sums",
            lambda: [
                BlockEncoding(1, [], lambda o: x(o[0])) + BlockEncoding(1, [], lambda o: x(o[1])),
                BlockEncoding(1, [], lambda o: x(o[1])) + BlockEncoding(1, [], lambda o: x(o[0])),
            ],
        ),
    ],
)
def test_product_survives_reuse_across_transformations(name, build, strategy):
    """A cached template or unitary must not carry a tracer out of its own trace.

    Ancilla templates record their register size, and in Jasp that size is a tracer,
    so a template or a closure over a layout first built inside a trace cannot be
    cached. Caching them made a second transformation over the same product fail
    with UnexpectedTracerError.
    """
    product = ProductBlockEncoding(build(), strategy=strategy)

    def circuit():
        operand = QuantumFloat(4)
        product.apply(operand)
        return operand

    make_jaspr(circuit)()
    count_ops(meas_behavior="0")(circuit)()
    depth(meas_behavior="0")(circuit)()
    num_qubits(meas_behavior="0")(circuit)()
    make_jaspr(circuit)()
