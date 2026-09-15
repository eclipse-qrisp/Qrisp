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

from qrisp import (
    QuantumBool,
    QuantumFloat,
    QuantumVariable,
    control,
    h,
    jaspify,
    measure,
    s_dg,
    terminal_sampling,
    x,
    z,
)
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


def test_block_encoding_lcu_cancels_to_the_zero_encoding():
    """Verify that static cancellation yields the zero operator rather than an error.

    This used to raise. Rejecting the cancellation also rejected expressions that
    only cancel in part, such as ``A - A + B``, because Python associates to the
    left and builds the cancelling half first.
    """
    block_encoding = BlockEncoding(1, [], lambda operand: None)

    combination = BlockEncoding.linear_combination([block_encoding, block_encoding], coefficients=[1, -1])

    assert isinstance(combination, BlockEncoding)
    assert combination.alpha == 0


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

    nested_product = ProductBlockEncoding([first, second], strategy="separate")
    product = ProductBlockEncoding([nested_product, third], strategy="separate")

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


def test_block_encoding_product_defaults_to_qubit_efficient_strategy():
    """Verify the default optimized strategy's execution and structural metadata."""
    calls = []

    def factor_unitary(ancilla, operand):
        calls.append("factor")

    factor = BlockEncoding(1, [QuantumBool()], factor_unitary)
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


def test_block_encoding_qubit_efficient_product_reuses_heterogeneous_ancillas():
    """Verify that factors share the largest workspace plus a logarithmic shift register."""
    first = BlockEncoding(1, [QuantumFloat(2), QuantumBool()], lambda *args: None)
    second = BlockEncoding(1, [QuantumFloat(1)], lambda *args: None)
    third = BlockEncoding(1, [QuantumBool(), QuantumBool()], lambda *args: None)

    product = ProductBlockEncoding([first, second, third], strategy="qubit_efficient")

    assert product.num_ancs == 2
    assert [template.qv_size for template in product._anc_templates] == [2, 3]


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
    ],
    ids=["two_factors", "three_factors", "four_multiqubit_factors"],
)
def test_block_encoding_qubit_efficient_product_matches_separate_strategy(operators):
    """Verify shared-workspace products against the Jasp-compiled reference strategy."""
    factors = [BlockEncoding.from_operator(operator) for operator in operators]
    separate = ProductBlockEncoding(factors, strategy="separate")
    qubit_efficient = ProductBlockEncoding(factors, strategy="qubit_efficient")
    num_qubits = max(operator.find_minimal_qubit_amount() for operator in operators)

    @terminal_sampling
    def main(block_encoding):
        return block_encoding.apply_rus(lambda: QuantumVariable(num_qubits))()

    _compare_results(main(separate), main(qubit_efficient), num_qubits)


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


def _control_distribution_after_phase_kickback(coefficient, rotate_to_y_basis):
    """Read a single-term combination's global phase off a control qubit.

    A global phase is unobservable on its own, so the encoding is applied under a
    control prepared in |+>. The phase is kicked back onto the control, which then
    interferes:  (|0> + exp(i*arg)|1>)/sqrt(2). Measuring in the X basis gives
    cos^2(arg/2), which pins the magnitude; rotating to the Y basis first gives
    cos^2((arg - pi/2)/2), which separates arg from -arg.

    The child acts as Z on an operand left in |0>, so it leaves the operand alone
    and the control stays unentangled. Any residual entanglement would wash out the
    interference and make the measurement meaningless rather than merely wrong.
    """
    child = BlockEncoding(1, [], lambda operand: z(operand[0]))
    encoding = BlockEncoding.linear_combination([child], [coefficient])

    @terminal_sampling
    def main():
        control_qbl = QuantumBool()
        h(control_qbl)
        operand = QuantumFloat(3)
        with control(control_qbl[0]):
            encoding.apply(operand)
        if rotate_to_y_basis:
            s_dg(control_qbl[0])
        h(control_qbl)
        return control_qbl

    return main()


@pytest.mark.parametrize(
    "coefficient, argument",
    [
        (1.0, 0.0),
        (-1.0, np.pi),
        (1j, np.pi / 2),
        (-1j, -np.pi / 2),
        ((1 + 1j) / np.sqrt(2), np.pi / 4),
    ],
)
def test_single_term_combination_applies_the_coefficient_phase(coefficient, argument):
    """A one-term linear combination must encode its coefficient's phase.

    alpha carries the coefficient's magnitude, so the unitary has to supply
    coefficient / abs(coefficient), a global phase of arg(coefficient). A negative
    real coefficient is the arg == pi case; writing that as a comparison against
    zero raised a TypeError for a complex coefficient, even though the rest of the
    LCU construction handles complex coefficients.
    """
    x_basis = _control_distribution_after_phase_kickback(coefficient, rotate_to_y_basis=False)
    y_basis = _control_distribution_after_phase_kickback(coefficient, rotate_to_y_basis=True)

    assert np.isclose(x_basis.get(False, 0), np.cos(argument / 2) ** 2, atol=1e-4)
    assert np.isclose(y_basis.get(False, 0), np.cos((argument - np.pi / 2) / 2) ** 2, atol=1e-4)


@pytest.mark.parametrize(
    "name, build, expected",
    [
        ("real coefficients", lambda Z_be, X_be: Z_be + 2.0 * X_be, True),
        ("negative coefficient", lambda Z_be, X_be: Z_be - X_be, True),
        ("imaginary coefficient", lambda Z_be, X_be: Z_be + 1j * X_be, False),
        ("complex coefficient", lambda Z_be, X_be: Z_be + ((1 + 1j) / np.sqrt(2)) * X_be, False),
        ("single imaginary term", lambda Z_be, X_be: BlockEncoding.linear_combination([Z_be], [1j]), False),
        ("single negative term", lambda Z_be, X_be: BlockEncoding.linear_combination([Z_be], [-1.0]), True),
    ],
)
def test_linear_combination_reports_hermitian_only_for_real_coefficients(name, build, expected):
    """A complex coefficient makes a sum of Hermitian operators non-Hermitian.

    ``I + 1j * Z`` is the shortest example. Reporting it as Hermitian is not merely
    imprecise: qubitization selects the cheaper reflection construction on the
    strength of this flag, which is only valid for a Hermitian operator.
    """
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))

    assert build(Z_be, X_be).is_hermitian is expected


def test_complex_coefficients_reach_the_non_hermitian_qubitization():
    r"""The flag has to change which construction qubitization builds, not just its value.

    The Hermitian branch reuses the encoding's own ancillas, while the general one
    adds a control ancilla to build a Hermitian operator out of $(A + A^\dagger)/2$.
    The extra ancilla is therefore a direct witness of the branch taken.
    """
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))

    real_combination = Z_be + 2.0 * X_be
    complex_combination = Z_be + 1j * X_be

    assert real_combination.qubitization().num_ancs == real_combination.num_ancs
    assert complex_combination.qubitization().num_ancs == complex_combination.num_ancs + 1


def test_coefficient_reality_survives_the_pytree_boundary():
    """Passing a combination through a jit boundary must not change the verdict.

    The coefficients become tracers there, so their value is no longer available,
    but their dtype is, and that is what the check relies on. Were the dtype not
    preserved, every combination would be reported non-Hermitian under tracing.
    """
    Z_be = BlockEncoding.from_operator(Z(0))
    X_be = BlockEncoding.from_operator(X(0))
    observed = {}

    @jaspify
    def main(block_encoding, key):
        observed[key] = block_encoding.is_hermitian
        qv = QuantumFloat(1)
        block_encoding.apply(qv)
        return measure(qv)

    main(Z_be + 2.0 * X_be, "real")
    main(Z_be + 1j * X_be, "complex")

    assert observed["real"] is True
    assert observed["complex"] is False


#
# Zero block encodings
#


def _zero_cases():
    A = BlockEncoding.from_operator(Z(0))
    B = BlockEncoding.from_operator(X(0))
    return A, B


@pytest.mark.parametrize(
    "name, build",
    [
        ("scaled by zero", lambda A, B: 0.0 * A),
        ("exact cancellation", lambda A, B: A - A),
        ("cancellation after merging", lambda A, B: 2 * A - A - A),
        ("two cancelling pairs", lambda A, B: (A - A) + (B - B)),
    ],
)
def test_fully_cancelling_combination_is_the_zero_encoding(name, build):
    """A combination that cancels must stay a block encoding, of the zero operator.

    Rejecting it at construction would also reject expressions whose result is
    perfectly ordinary, since Python associates to the left and ``A - A + B`` builds
    the cancelling part first.
    """
    A, B = _zero_cases()
    encoding = build(A, B)

    assert isinstance(encoding, BlockEncoding)
    assert encoding.alpha == 0


@pytest.mark.parametrize(
    "name, build",
    [
        ("zero term first", lambda A, B: 0 * A + B),
        ("cancellation first", lambda A, B: A - A + B),
        ("cancellation parenthesised", lambda A, B: (A - A) + B),
        ("cancellation across the sum", lambda A, B: A + B - A),
    ],
)
def test_zero_terms_are_absorbed_by_a_combination(name, build):
    """A term contributing nothing must leave no trace in the result.

    A term contributes ``coefficient * child.alpha``, so it drops out when either
    factor is zero. Dropping it is what lets the surviving single term be handed
    back unwrapped, rather than as a combination carrying a dead SELECT branch.
    """
    A, B = _zero_cases()

    assert build(A, B) is B


def test_zero_encoding_cannot_be_applied():
    """Applying the zero operator has no meaning and must say so.

    Its block encoding can never succeed: the probability of projecting onto the
    ancillas being zero is itself zero. The error belongs here rather than at
    construction, where it would also reject expressions that cancel only in part.
    """
    A, _ = _zero_cases()
    zero = A - A

    with pytest.raises(ValueError, match="zero operator"):
        zero.apply(QuantumFloat(1))

    with pytest.raises(ValueError, match="zero operator"):
        zero.apply_rus(lambda: QuantumFloat(1))


@pytest.mark.parametrize(
    "name, build",
    [
        ("zero on the left", lambda A, B: (A - A) @ B),
        ("zero on the right", lambda A, B: B @ (A - A)),
        ("zero late in a chain", lambda A, B: A @ B @ (A - A)),
        ("zero from a zero coefficient", lambda A, B: (0 * A) @ B),
    ],
)
def test_zero_annihilates_a_product(name, build):
    """A zero factor makes the whole product the zero operator.

    Where a sum absorbs a zero term, a product is annihilated by one, so there is
    nothing to gain from building the remaining factors.
    """
    A, B = _zero_cases()
    product = build(A, B)

    assert product.alpha == 0
    assert not isinstance(product, ProductBlockEncoding)


#
# Terms are validated as supplied
#


def _one_and_two_operand_encodings():
    one = BlockEncoding(1, [], lambda a: None, num_ops=1)
    two = BlockEncoding(1, [], lambda a, b: None, num_ops=2)
    return one, two


@pytest.mark.parametrize(
    "name, build",
    [
        ("zero coefficient", lambda one, two: BlockEncoding.linear_combination([one, two], coefficients=[1, 0])),
        ("zero encoding on the right", lambda one, two: one + (two - two)),
        ("zero encoding on the left", lambda one, two: (two - two) + one),
    ],
)
def test_a_zero_term_does_not_hide_an_operand_mismatch(name, build):
    """Operand counts are compared before any term is dropped.

    Canonicalization removes a term that contributes nothing, so a mismatched
    child carrying a zero coefficient used to disappear before being checked, and
    the combination silently returned the surviving encoding instead of rejecting
    the call.
    """
    one, two = _one_and_two_operand_encodings()

    with pytest.raises(ValueError, match="same number of operands"):
        build(one, two)


def test_an_invalid_child_is_reported_as_a_type_error():
    """A child that is not a block encoding must raise the documented TypeError.

    Canonicalization reads each child's normalization, so an invalid one used to
    surface as an AttributeError raised from the merging step.
    """
    with pytest.raises(TypeError, match="BlockEncoding"):
        LinearCombinationBlockEncoding([(1, "not a block encoding")])


@pytest.mark.parametrize(
    "name, build",
    [
        ("zero factor on the left", lambda one, two: (one - one) @ two),
        ("zero factor on the right", lambda one, two: two @ (one - one)),
    ],
)
def test_a_zero_factor_does_not_hide_an_operand_mismatch(name, build):
    """A product checks its factors against each other before zero annihilates it.

    A zero factor collapses the product without the remaining factors being built,
    so the operand counts have to be compared first or the mismatch is swallowed
    by the zero result.
    """
    one, two = _one_and_two_operand_encodings()

    with pytest.raises(ValueError, match="same number of operands"):
        build(one, two)
