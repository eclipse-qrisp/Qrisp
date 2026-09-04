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

from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
import pytest

from qrisp import QuantumBool, QuantumFloat, QuantumVariable, jaspify, measure, terminal_sampling, x
from qrisp.block_encodings import BlockEncoding, LinearCombinationBlockEncoding, ProductBlockEncoding
from qrisp.operators import X, Y, Z


def _compare_results(res_dict_1, res_dict_2, n):
    for k in range(2**n):
        val_1 = res_dict_1.get(k, 0)
        val_2 = res_dict_2.get(k, 0)
        assert np.isclose(val_1, val_2)


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4), Z(0) + Z(1) + X(2)),
        (X(0) * X(1), Z(0) + 0.9 * Z(1) + X(3)),
    ],
)
def test_block_encoding_addition(H1, H2):
    """Test addition of block encodings corresponding to Hermitian operators."""

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 + H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_addition = BE1 + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_add = main(BE_addition)
    _compare_results(res_be3, res_be_add, n)


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4), Z(0) + Z(1) + X(2)),
        (X(0) * X(1), Z(0) + 0.9 * Z(1) + X(3)),
    ],
)
def test_block_encoding_subtraction(H1, H2):
    """Test subtraction of block encodings corresponding to Hermitian operators."""

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 - H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_subtraction = BE1 - BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_sub = main(BE_subtraction)
    _compare_results(res_be3, res_be_sub, n)


# The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.
# Thus, to ensure that the multiplication test is valid, we should choose pairs of operators that commute.
@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4), X(0) + X(4)),
        (X(0) * X(1), Z(0) * Z(1) + Y(3)),
    ],
)
def test_block_encoding_multiplication(H1, H2):
    """Test multiplication of block encodings corresponding to commuting Hermitian operators."""

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    H3 = H1 * H2
    BE3 = BlockEncoding.from_operator(H3)
    BE_multiplication = BE1 @ BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be3 = main(BE3)
    res_be_mul = main(BE_multiplication)
    _compare_results(res_be3, res_be_mul, n)


@pytest.mark.parametrize(
    "H1, H2, scalar",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2), -2),
        (0.5 * X(1) + 0.7 * Y(1), Z(0) + X(2), 0.5),
        (X(0), Z(0), 1),
    ],
)
def test_block_encoding_scalar_multiplication(H1, H2, scalar):
    H_target = scalar * H1 + H2
    BE_target = BlockEncoding.from_operator(H_target)

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    BE_left = scalar * BE1 + BE2
    BE_right = BE1 * scalar + BE2

    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_target = main(BE_target)
    res_left = main(BE_left)
    res_right = main(BE_right)
    _compare_results(res_target, res_left, n)
    _compare_results(res_target, res_right, n)


def test_block_encoding_flattened_linear_combination():
    """Verify that linear combinations of block encodings are flattened into one LCU."""

    H1 = X(0) * X(1)
    H2 = 0.4 * Y(0) + Z(1)
    H3 = 0.7 * X(2)
    coefficients = [2.0, -1.0, 0.5]

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    BE3 = BlockEncoding.from_operator(H3)

    BE_chain = coefficients[0] * BE1 + coefficients[1] * BE2 + coefficients[2] * BE3
    BE_direct = BlockEncoding.linear_combination([BE1, BE2, BE3], coefficients=coefficients)

    assert isinstance(BE_chain, LinearCombinationBlockEncoding)
    assert isinstance(BE_direct, LinearCombinationBlockEncoding)
    assert len(BE_chain.terms) == 3
    assert len(BE_direct.terms) == 3
    assert BE_chain._anc_templates[0].qv_size == 2
    # A linear combination of block encodings has exactly 2 ancillas:
    # one for the LCU selection and one for the workspace.
    assert BE_chain.num_ancs == 2

    leaves, treedef = tree_flatten(BE_chain)
    reconstructed = tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, LinearCombinationBlockEncoding)
    assert len(reconstructed.terms) == 3
    assert len((reconstructed + BE1).terms) == 4

    H_target = coefficients[0] * H1 + coefficients[1] * H2 + coefficients[2] * H3
    BE_target = BlockEncoding.from_operator(H_target)
    n = max(
        H1.find_minimal_qubit_amount(),
        H2.find_minimal_qubit_amount(),
        H3.find_minimal_qubit_amount(),
    )

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_target = main(BE_target)
    _compare_results(res_target, main(BE_chain), n)
    _compare_results(res_target, main(BE_direct), n)


def test_block_encoding_lcu_canonicalizes_repeated_references():
    """Verify that repeated references are merged into one weighted term."""
    block_encoding = BlockEncoding(1, [], lambda operand: None)

    combination = BlockEncoding.linear_combination(
        [block_encoding, block_encoding],
        coefficients=[2, 3],
    )

    assert combination.terms == ((5, block_encoding),)


def test_block_encoding_lcu_removes_static_zero_terms():
    """Verify that concrete zero coefficients are removed."""
    first = BlockEncoding(1, [], lambda operand: None)
    second = BlockEncoding(1, [], lambda operand: None)

    combination = BlockEncoding.linear_combination([first, second], coefficients=[0, 2])

    assert combination.terms == ((2, second),)


def test_block_encoding_lcu_rejects_all_zero_terms():
    """Verify that static cancellation cannot create a zero-normalization LCU."""
    block_encoding = BlockEncoding(1, [], lambda operand: None)

    with pytest.raises(ValueError, match="all-zero linear combination"):
        BlockEncoding.linear_combination([block_encoding, block_encoding], coefficients=[1, -1])


def test_block_encoding_lcu_preserves_dynamic_coefficients():
    """Verify that dynamic coefficients are not compared in Python."""
    import jax
    import jax.numpy as jnp

    block_encoding = BlockEncoding(2, [], lambda operand: None)

    def normalization(coefficient):
        combination = BlockEncoding.linear_combination([block_encoding], coefficients=[coefficient])
        return combination.alpha

    assert jax.jit(normalization)(jnp.array(3.0)) == 6.0


def test_block_encoding_linear_combination_validates_inputs():
    """Verify that linear combination input errors are reported."""
    block_encoding = BlockEncoding(1, [], lambda operand: None)
    two_operand_block_encoding = BlockEncoding(1, [], lambda first, second: None, num_ops=2)

    with pytest.raises(ValueError, match="At least one block-encoding is required"):
        BlockEncoding.linear_combination([])

    with pytest.raises(ValueError, match="number of coefficients"):
        BlockEncoding.linear_combination([block_encoding], coefficients=[1, 2])

    with pytest.raises(TypeError, match="Expected every item to be a BlockEncoding"):
        BlockEncoding.linear_combination([object()])

    with pytest.raises(ValueError, match="same number of operands"):
        BlockEncoding.linear_combination([block_encoding, two_operand_block_encoding])


def test_block_encoding_product_is_flattened_and_keeps_separate_ancillas():
    """Verify that nested products preserve factor order and ancilla ownership."""
    first = BlockEncoding(2, [QuantumFloat(2)], lambda ancilla, operand: None)
    second = BlockEncoding(3, [QuantumBool()], lambda ancilla, operand: None)
    third = BlockEncoding(5, [QuantumFloat(1)], lambda ancilla, operand: None)

    product = (first @ second) @ third

    assert isinstance(product, ProductBlockEncoding)
    assert product.factors == (first, second, third)
    assert product.alpha == 30
    assert product.num_ancs == first.num_ancs + second.num_ancs + third.num_ancs
    assert [template.qv_size for template in product._anc_templates] == [2, 1, 1]

    with pytest.raises(TypeError):
        product.factors[0] = first
    with pytest.raises(AttributeError):
        product.factors = ()
    with pytest.raises(AttributeError):
        product._factors = ()


def test_block_encoding_product_applies_factors_in_reverse_order():
    """Verify that A @ B applies B before A."""
    calls = []

    def first_unitary(ancilla, operand):
        calls.append("first")

    def second_unitary(ancilla, operand):
        calls.append("second")

    first = BlockEncoding(1, [QuantumBool()], first_unitary)
    second = BlockEncoding(1, [QuantumBool()], second_unitary)
    product = first @ second
    ancillas = product.create_ancillas()
    operand = QuantumVariable(1)

    product.unitary(*ancillas, operand)

    assert calls == ["second", "first"]


def test_block_encoding_product_supports_pytree_and_structural_dagger():
    """Verify product reconstruction and reversed factor daggers."""
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
    assert len(dagger.factors) == 3
    assert [factor.alpha for factor in dagger.factors] == [third.alpha, second.alpha, first.alpha]
    assert dagger.alpha == product.alpha


def test_linear_combination_block_encoding_has_immutable_derived_representation():
    """Verify that LCU terms are authoritative and exposed immutably."""
    first = BlockEncoding(2, [], lambda operand: None, is_hermitian=True)
    second = BlockEncoding(3, [], lambda operand: None, is_hermitian=True)
    combination = BlockEncoding.linear_combination([first, second], coefficients=[4, -1])

    assert isinstance(combination, LinearCombinationBlockEncoding)
    assert isinstance(combination.terms, tuple)
    assert combination.alpha == 11
    assert combination.num_ops == 1
    assert combination.num_ancs == 2
    assert combination.is_hermitian

    with pytest.raises(TypeError):
        combination.terms[0] = (1, first)
    with pytest.raises(AttributeError):
        combination.terms = ()
    with pytest.raises(AttributeError):
        combination._terms = ()


def test_block_encoding_lcu_reuses_heterogeneous_ancillas():
    """Verify that heterogeneous child ancillas share one workspace."""

    def two_ancilla_unitary(float_ancilla, bool_ancilla, operand):
        x(float_ancilla)
        x(bool_ancilla)

    def one_ancilla_unitary(float_ancilla, operand):
        x(float_ancilla)

    BE_two_ancillas = BlockEncoding(
        1,
        [QuantumFloat(2), QuantumBool()],
        two_ancilla_unitary,
    )
    BE_one_ancilla = BlockEncoding(1, [QuantumFloat(1)], one_ancilla_unitary)
    BE = BlockEncoding.linear_combination([BE_two_ancillas, BE_one_ancilla])

    # A linear combination of block encodings has exactly 2 ancillas:
    # one for the LCU selection and one for the workspace.
    assert BE.num_ancs == 2
    # The workspace ancilla is the second one, which is a QuantumVariable of size 3
    # (2 for the float and 1 for the bool).
    assert BE._anc_templates[1].qv_size == 3

    operand = QuantumVariable(1)
    BE.apply(operand)

    @jaspify
    def main(block_encoding):
        operand = QuantumVariable(1)
        block_encoding.apply(operand)
        return measure(operand)

    assert main(BE) == 0


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) - 0.2 * Y(0) * Y(1), 0.2 * Y(0) * Y(1) - X(0) * X(1)),
        (0.5 * X(1) - 0.7 * Y(1) + 0.3 * X(4), 0.7 * Y(1) - 0.5 * X(1) - 0.3 * X(4)),
        (Z(0) * Z(1) - Y(3), Y(3) - Z(0) * Z(1)),
    ],
)
def test_block_encoding_negation(H1, H2):
    """Test negation of block encodings corresponding to Hermitian operators."""

    BE1 = BlockEncoding.from_operator(H1)
    BE_neg = -BE1

    BE2 = BlockEncoding.from_operator(H2)

    n = H1.find_minimal_qubit_amount()

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(lambda: QuantumVariable(n))()

    res_be2 = main(BE2)
    res_be_neg = main(BE_neg)
    _compare_results(res_be2, res_be_neg, n)


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (X(0) * X(1), Z(0) * Z(1)),
    ],
)
def test_block_encoding_kron(H1, H2):
    """Test the Kronecker product of two block encodings corresponding to Hermitian operators."""

    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)

    BE_kron = BE1.kron(BE2)

    n1 = H1.find_minimal_qubit_amount()
    n2 = H2.find_minimal_qubit_amount()

    def operand_prep():
        qv1 = QuantumFloat(n1)
        qv2 = QuantumFloat(n2)
        return qv1, qv2

    @terminal_sampling
    def main(BE):
        return BE.apply_rus(operand_prep)()

    result_be_kron = main(BE_kron)

    @terminal_sampling
    def main(BE1, BE2):
        qv1 = BE1.apply_rus(lambda: QuantumFloat(n1))()
        qv2 = BE2.apply_rus(lambda: QuantumFloat(n2))()
        return qv1, qv2

    result_be1_be2 = main(BE1, BE2)

    for k in range(2**n1):
        for l in range(2**n2):
            val_be_kron = result_be_kron.get((k, l), 0)
            val_be1_be2 = result_be1_be2.get((k, l), 0)
            assert np.isclose(val_be_kron, val_be1_be2)
