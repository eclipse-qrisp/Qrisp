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

"""Tests for ProductBlockEncoding semantics, strategies, validation, and compilation."""

import jax
import numpy as np
import pytest
from _helpers import compare_distributions, sample_postselected
from jax.tree_util import tree_flatten, tree_unflatten

from qrisp import QuantumBool, QuantumFloat, QuantumVariable, make_jaspr, x, z
from qrisp.block_encodings import BlockEncoding, ProductBlockEncoding
from qrisp.jasp import count_ops
from qrisp.operators import X, Y, Z

STRATEGIES = ["qubit_efficient", "separate"]


def _counting_encoding(executions, qubit=0, num_ancillas=0):
    templates = [QuantumFloat(1).template() for _ in range(num_ancillas)]

    def unitary(*args):
        executions.append(None)
        x(args[-1][qubit])

    return BlockEncoding(1, templates, unitary)


#
# Structure, strategies, execution order, dagger, and validation
#


def test_flattened_separate_strategy_and_immutability():
    """Nested products flatten in mathematical order and expose factors immutably."""
    first = BlockEncoding(2, [QuantumFloat(2)], lambda ancilla, operand: None)
    second = BlockEncoding(3, [QuantumBool()], lambda ancilla, operand: None)
    third = BlockEncoding(5, [QuantumFloat(1)], lambda ancilla, operand: None)

    nested = ProductBlockEncoding([first, second], strategy="separate")
    product = ProductBlockEncoding([nested, third], strategy="separate")

    assert isinstance(product, ProductBlockEncoding)
    assert product.factors == (first, second, third)
    assert product.strategy == "separate"
    assert product.alpha == 30
    assert product.num_ancs == first.num_ancs + second.num_ancs + third.num_ancs
    assert [template.qv_size for template in product._anc_templates] == [2, 1, 1]

    with pytest.raises(TypeError):
        product.factors[0] = first
    with pytest.raises(AttributeError):
        product.factors = ()
    with pytest.raises(AttributeError):
        product._factors = ()


def test_default_qubit_efficient_strategy_and_pytree():
    """ProductBlockEncoding defaults to qubit_efficient and preserves strategy across PyTree and dagger."""
    calls = []
    factor = BlockEncoding(1, [QuantumBool()], lambda anc, op: calls.append("factor"))
    product = ProductBlockEncoding([factor])

    assert product.strategy == "qubit_efficient"
    assert (factor @ factor).strategy == "qubit_efficient"
    product.unitary(*product.create_ancillas(), QuantumVariable(1))
    assert calls == ["factor"]
    assert product.num_ancs == factor.num_ancs

    leaves, treedef = tree_flatten(product)
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed.strategy == "qubit_efficient"
    assert reconstructed.dagger().strategy == "qubit_efficient"

    with pytest.raises(ValueError, match="Unknown product strategy"):
        ProductBlockEncoding([factor], strategy="unknown")
    with pytest.raises(AttributeError):
        product.strategy = "separate"
    with pytest.raises(AttributeError):
        product._strategy = "separate"


def test_qubit_efficient_reuses_heterogeneous_ancillas():
    """Qubit-efficient products pack heterogeneous factor ancillas into one workspace plus counter."""
    first = BlockEncoding(1, [QuantumFloat(2), QuantumBool()], lambda *args: None)
    second = BlockEncoding(1, [QuantumFloat(1)], lambda *args: None)
    third = BlockEncoding(1, [QuantumBool(), QuantumBool()], lambda *args: None)

    product = ProductBlockEncoding([first, second, third], strategy="qubit_efficient")

    assert product.num_ancs == 2
    assert [template.qv_size for template in product._anc_templates] == [2, 3]


def test_qubit_efficient_counts_only_factors_with_ancillas():
    """Factors without ancillas neither enlarge the counter nor force a shared workspace."""
    first = BlockEncoding(1, [QuantumFloat(2)], lambda *args: None)
    second = BlockEncoding(1, [QuantumBool()], lambda *args: None)
    unitary = BlockEncoding(1, [], lambda *args: None)

    product = ProductBlockEncoding([unitary, first, unitary, second, unitary], strategy="qubit_efficient")
    assert [template.qv_size for template in product._anc_templates] == [1, 2]

    single_anc = ProductBlockEncoding([unitary, first, unitary], strategy="qubit_efficient")
    assert [template.qv_size for template in single_anc._anc_templates] == [2]


def test_qubit_efficient_keeps_few_quantum_bool_ancillas_separate():
    """Up to five single-QuantumBool factors stay separate; six or more use the shared workspace."""

    def bool_factor():
        return BlockEncoding(1, [QuantumBool()], lambda *args: None)

    unitary = BlockEncoding(1, [], lambda *args: None)
    five = ProductBlockEncoding([bool_factor() for _ in range(5)] + [unitary], strategy="qubit_efficient")
    six = ProductBlockEncoding([bool_factor() for _ in range(6)], strategy="qubit_efficient")
    mixed = ProductBlockEncoding(
        [bool_factor(), BlockEncoding(1, [QuantumFloat(1)], lambda *args: None)], strategy="qubit_efficient"
    )

    assert [t.qv_size for t in five._anc_templates] == [1] * 5
    assert [t.qv_size for t in six._anc_templates] == [3, 1]
    assert [t.qv_size for t in mixed._anc_templates] == [1, 1]

    observed = []
    make_jaspr(lambda p: observed.append(p.num_ancs) or QuantumFloat(1))(five)
    make_jaspr(lambda p: observed.append(p.num_ancs) or QuantumFloat(1))(six)
    assert observed == [5, 2]


@pytest.mark.parametrize(
    "operators",
    [
        [X(0) + 0.4 * Z(0), Y(0) + 0.3 * Z(0)],
        [X(0) + 0.4 * Z(0), Y(0) + 0.3 * Z(0), X(0) + 0.2 * Y(0)],
        [
            X(0) * X(1) + 0.2 * Z(0),
            Y(0) * Y(1) + 0.3 * X(1),
            Z(0) * Z(1) + 0.2 * X(0),
            X(0) + 0.1 * Y(1),
        ],
        [X(0), Y(0) + 0.3 * Z(0), Z(0)],
        [X(0) + 0.4 * Z(0), Y(0), X(0) + 0.2 * Y(0), Z(0)],
    ],
    ids=[
        "two_factors",
        "three_factors",
        "four_multiqubit_factors",
        "one_factor_with_ancillas",
        "factors_without_ancillas_interleaved",
    ],
)
def test_qubit_efficient_matches_separate_strategy(operators):
    """Qubit-efficient and separate product strategies produce identical output distributions."""
    factors = [BlockEncoding.from_operator(op) for op in operators]
    separate = ProductBlockEncoding(factors, strategy="separate")
    qubit_efficient = ProductBlockEncoding(factors, strategy="qubit_efficient")
    n = max(op.find_minimal_qubit_amount() for op in operators)
    prep = lambda: QuantumVariable(n)  # noqa: E731

    compare_distributions(sample_postselected(separate, prep), sample_postselected(qubit_efficient, prep), n)


def test_applies_factors_in_reverse_order():
    """Product A @ B applies B before A."""
    calls = []
    first = BlockEncoding(1, [QuantumBool()], lambda anc, op: calls.append("first"))
    second = BlockEncoding(1, [QuantumBool()], lambda anc, op: calls.append("second"))
    product = first @ second
    product.unitary(*product.create_ancillas(), QuantumVariable(1))
    assert calls == ["second", "first"]


def test_pytree_and_structural_dagger():
    """Products round-trip through PyTree and dagger reverses factors."""
    first = BlockEncoding(2, [], lambda operand: None)
    second = BlockEncoding(3, [], lambda operand: None)
    third = BlockEncoding(5, [], lambda operand: None)
    product = first @ second @ third

    leaves, treedef = tree_flatten(product)
    reconstructed = tree_unflatten(treedef, leaves)
    dagger = product.dagger()

    assert isinstance(reconstructed, ProductBlockEncoding)
    assert len(reconstructed.factors) == 3
    assert reconstructed.alpha == product.alpha
    assert isinstance(dagger, ProductBlockEncoding)
    assert [factor.alpha for factor in dagger.factors] == [third.alpha, second.alpha, first.alpha]
    assert dagger.alpha == product.alpha


def test_factors_are_kept_as_written_and_validated():
    """Products retain cancelling factors and reject operand mismatches even on zero-scaled factors."""
    A = BlockEncoding.from_operator(Z(0))
    B = BlockEncoding.from_operator(X(0))
    cancelling = A - A
    product = cancelling @ B
    assert product.factors == (cancelling, B)
    assert product.alpha == cancelling.alpha * B.alpha

    one = BlockEncoding(1, [], lambda a: None, num_ops=1)
    two = BlockEncoding(1, [], lambda a, b: None, num_ops=2)
    for bad_product in (lambda: (0 * one) @ two, lambda: one @ (0 * two), lambda: one @ two):
        with pytest.raises(ValueError, match="same number of operands"):
            bad_product()


#
# Compilation, tracing, and caching
#


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unitary_identity_is_stable(strategy):
    """Accessing product.unitary repeatedly returns the same callable object."""
    product = ProductBlockEncoding([BlockEncoding.from_eye(0), BlockEncoding.from_eye(1)], strategy=strategy)
    assert product.unitary is product.unitary
    assert product.alpha == product.alpha


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_repeated_application_traces_each_factor_once(strategy):
    """Applying the same product twice traces each factor unitary once."""
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
    """Longer product chains trace each factor once regardless of chain length."""
    for num_factors in (2, 4, 8):
        executions = []
        factors = [_counting_encoding(executions)] + [
            BlockEncoding(1, [], lambda operand: x(operand[1])) for _ in range(num_factors - 1)
        ]
        product = ProductBlockEncoding(factors, strategy=strategy)
        make_jaspr(lambda: product.apply(QuantumFloat(3)) and None)()
        assert len(executions) == 1


def test_shared_factor_is_traced_once_across_products():
    """A factor appearing multiple times in a product (H @ P @ H) is traced once."""
    executions = []
    shared = _counting_encoding(executions)
    other = BlockEncoding(1, [], lambda operand: x(operand[1]))

    product = shared @ other @ shared
    make_jaspr(lambda: product.apply(QuantumFloat(3)) and None)()
    assert len(executions) == 1


@pytest.mark.parametrize(
    "name, build",
    [
        ("separate", lambda a, b: ProductBlockEncoding((a, b), strategy="separate")),
        ("qubit_efficient, one factor", lambda a, b: ProductBlockEncoding((a,), strategy="qubit_efficient")),
        ("qubit_efficient, shared workspace", lambda a, b: ProductBlockEncoding((a, b))),
    ],
)
def test_pytree_product_traces_factors_once(name, build):
    """Passing a product across a PyTree boundary preserves unitary identity and traces factors once."""
    executions = []

    def first_unitary(ancilla, operand):
        executions.append(None)
        x(operand[0])

    first = BlockEncoding(1, [QuantumFloat(2)], first_unitary)
    second = BlockEncoding(1, [QuantumFloat(2)], lambda ancilla, operand: z(operand[0]))

    def main(product):
        assert product.unitary is product.unitary
        operand = QuantumFloat(2)
        product.apply(operand)
        product.apply(operand)
        return operand

    make_jaspr(main)(build(first, second))
    assert len(executions) == 1


def test_factors_sharing_unitary_receive_own_ancilla_templates():
    """Distinct factors sharing one unitary callable still receive their own ancilla attributes."""
    seen_exponents = []

    def shared_unitary(ancilla, operand):
        seen_exponents.append(ancilla.exponent)
        x(operand[0])

    first = BlockEncoding(1, [QuantumFloat(3, exponent=0)], shared_unitary)
    second = BlockEncoding(1, [QuantumFloat(3, exponent=-2)], shared_unitary)

    product = ProductBlockEncoding((first, second), strategy="qubit_efficient")
    product.apply(QuantumFloat(2))
    assert seen_exponents == [-2, 0]


def test_alpha_stays_concrete_during_tracing():
    """Product normalization remains a concrete scalar inside a trace."""
    product = BlockEncoding(2.0, [], lambda operand: None) @ BlockEncoding(3.0, [], lambda operand: None)
    observed = {}

    def circuit():
        observed["alpha"] = product.alpha
        operand = QuantumFloat(3)
        product.apply(operand)
        return operand

    make_jaspr(circuit)()
    assert not isinstance(observed["alpha"], jax.core.Tracer)
    assert np.isclose(observed["alpha"], 6.0)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "name, build",
    [
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
def test_reuse_across_transformations(name, build, strategy):
    """Reusing one product across multiple JAX transformations leaks no tracers."""
    product = ProductBlockEncoding(build(), strategy=strategy)

    def circuit():
        operand = QuantumFloat(4)
        product.apply(operand)
        return operand

    make_jaspr(circuit)()
    count_ops(meas_behavior="0")(circuit)()
    make_jaspr(circuit)()
