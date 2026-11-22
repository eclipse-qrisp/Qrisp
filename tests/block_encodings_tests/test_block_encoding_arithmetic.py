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

"""Tests for BlockEncoding arithmetic operators (+, -, *, @, neg, kron) and cross-composite behavior."""

import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import compare_distributions, postselect_results, sample_postselected

from qrisp import QuantumBool, QuantumFloat, QuantumVariable, h, terminal_sampling, x, z
from qrisp.block_encodings import BlockEncoding
from qrisp.operators import X, Y, Z


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4), Z(0) + Z(1) + X(2)),
        (X(0) * X(1), Z(0) + 0.9 * Z(1) + X(3)),
    ],
)
def test_block_encoding_addition(H1, H2):
    """Addition of block encodings matches direct encoding of H1 + H2."""
    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    BE_target = BlockEncoding.from_operator(H1 + H2)
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())
    prep = lambda: QuantumVariable(n)  # noqa: E731

    compare_distributions(sample_postselected(BE_target, prep), sample_postselected(BE1 + BE2, prep), n)


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4), Z(0) + Z(1) + X(2)),
        (X(0) * X(1), Z(0) + 0.9 * Z(1) + X(3)),
    ],
)
def test_block_encoding_subtraction(H1, H2):
    """Subtraction of block encodings matches direct encoding of H1 - H2."""
    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    BE_target = BlockEncoding.from_operator(H1 - H2)
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())
    prep = lambda: QuantumVariable(n)  # noqa: E731

    compare_distributions(sample_postselected(BE_target, prep), sample_postselected(BE1 - BE2, prep), n)


# Commuting Hermitian pairs ensure H1 * H2 is Hermitian for BlockEncoding.from_operator.
@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (0.5 * X(1) + 0.7 * Y(1) + 0.3 * X(4), X(0) + X(4)),
        (X(0) * X(1), Z(0) * Z(1) + Y(3)),
    ],
)
def test_block_encoding_multiplication(H1, H2):
    """Product of block encodings matches direct encoding of H1 * H2."""
    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    BE_target = BlockEncoding.from_operator(H1 * H2)
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())
    prep = lambda: QuantumVariable(n)  # noqa: E731

    compare_distributions(sample_postselected(BE_target, prep), sample_postselected(BE1 @ BE2, prep), n)


@pytest.mark.parametrize(
    "H1, H2, scalar",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2), -2),
        (0.5 * X(1) + 0.7 * Y(1), Z(0) + X(2), 0.5),
        (X(0), Z(0), 1),
    ],
)
def test_block_encoding_scalar_multiplication(H1, H2, scalar):
    """Left and right scalar multiplication match direct encoding of scalar * H1 + H2."""
    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    BE_target = BlockEncoding.from_operator(scalar * H1 + H2)
    n = max(H1.find_minimal_qubit_amount(), H2.find_minimal_qubit_amount())
    prep = lambda: QuantumVariable(n)  # noqa: E731

    res_target = sample_postselected(BE_target, prep)
    compare_distributions(res_target, sample_postselected(scalar * BE1 + BE2, prep), n)
    compare_distributions(res_target, sample_postselected(BE1 * scalar + BE2, prep), n)


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) - 0.2 * Y(0) * Y(1), 0.2 * Y(0) * Y(1) - X(0) * X(1)),
        (0.5 * X(1) - 0.7 * Y(1) + 0.3 * X(4), 0.7 * Y(1) - 0.5 * X(1) - 0.3 * X(4)),
        (Z(0) * Z(1) - Y(3), Y(3) - Z(0) * Z(1)),
    ],
)
def test_block_encoding_negation(H1, H2):
    """Negation of a block encoding matches direct encoding of -H1."""
    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    n = H1.find_minimal_qubit_amount()
    prep = lambda: QuantumVariable(n)  # noqa: E731

    compare_distributions(sample_postselected(BE2, prep), sample_postselected(-BE1, prep), n)


@pytest.mark.parametrize(
    "H1, H2",
    [
        (X(0) * X(1) + 0.2 * Y(0) * Y(1), Z(0) * Z(1) + X(2)),
        (X(0) * X(1), Z(0) * Z(1)),
    ],
)
def test_block_encoding_kron(H1, H2):
    """Kronecker product of two block encodings matches independent application."""
    BE1 = BlockEncoding.from_operator(H1)
    BE2 = BlockEncoding.from_operator(H2)
    n1 = H1.find_minimal_qubit_amount()
    n2 = H2.find_minimal_qubit_amount()

    res_kron = sample_postselected(BE1.kron(BE2), lambda: (QuantumFloat(n1), QuantumFloat(n2)))
    res_1 = sample_postselected(BE1, lambda: QuantumFloat(n1))
    res_2 = sample_postselected(BE2, lambda: QuantumFloat(n2))

    for k in range(2**n1):
        for m in range(2**n2):
            assert np.isclose(res_kron.get((k, m), 0), res_1.get(k, 0) * res_2.get(m, 0))


def test_product_of_linear_combinations():
    """Product of linear combinations matches direct product of the summed operators."""
    H1 = X(0) + 0.5 * Z(0)
    H2 = Y(0) + 0.25 * Z(0)
    composite = (BlockEncoding.from_operator(X(0)) + 0.5 * BlockEncoding.from_operator(Z(0))) @ (
        BlockEncoding.from_operator(Y(0)) + 0.25 * BlockEncoding.from_operator(Z(0))
    )
    reference = BlockEncoding.from_operator(H1) @ BlockEncoding.from_operator(H2)
    prep = lambda: QuantumVariable(1)  # noqa: E731

    compare_distributions(sample_postselected(composite, prep), sample_postselected(reference, prep), 1)


@pytest.mark.parametrize(
    "name, build",
    [
        ("product factor", lambda c, A, B, C: (((c * A) @ B) + C, C)),
        ("linear_combination item", lambda c, A, B, C: (BlockEncoding.linear_combination([c * A, B]), B)),
        ("dagger", lambda c, A, B, C: ((c * A).dagger() + B, B)),
    ],
)
def test_zero_scaled_nested_encoding_contributes_nothing(name, build):
    """A zero-scaled nested encoding contributes zero amplitude for both static and traced zeros."""
    A = BlockEncoding.from_operator(X(0) + 0.5 * Z(1))
    B = BlockEncoding.from_operator(Z(0) * Z(1))
    C = BlockEncoding.from_operator(X(1))

    def operand_prep():
        qf = QuantumFloat(2)
        h(qf[0])
        return qf

    block_encoding, reference = build(0, A, B, C)

    @terminal_sampling
    def main_traced(c):
        operand = operand_prep()
        ancillas = build(c, A, B, C)[0].apply(operand)
        return operand, *ancillas

    res_ref = sample_postselected(reference, operand_prep)
    compare_distributions(res_ref, sample_postselected(block_encoding, operand_prep), 2)
    compare_distributions(res_ref, postselect_results(main_traced(0.0), num_ops=1), 2)


def test_composite_ancillas_are_quantum_floats():
    """Ancillas allocated by linear combinations and products are QuantumFloat instances."""
    be1 = BlockEncoding(1, [QuantumBool()], lambda anc, op: x(op[0]))
    be2 = BlockEncoding(1, [QuantumFloat(3)], lambda anc, op: z(op[0]))

    for composite in (BlockEncoding.linear_combination([be1, be2], coefficients=[1.0, 1.0]), be1 @ be2):
        for anc in composite.apply(QuantumFloat(3)):
            assert isinstance(anc, QuantumFloat)


def test_block_encoding_alpha_validation_and_detaching():
    """BlockEncoding rejects non-scalar or complex alpha values and detaches 0-d NumPy arrays."""

    def unitary(operand):
        x(operand[0])

    with pytest.raises(ValueError, match="scalar"):
        BlockEncoding([1.0, 2.0], [], unitary)
    with pytest.raises(ValueError, match="scalar"):
        BlockEncoding(np.array([1.0, 2.0]), [], unitary)
    for complex_alpha in (1 + 2j, np.complex128(2.0), jnp.array(1 + 2j)):
        with pytest.raises(ValueError, match="real scalar"):
            BlockEncoding(complex_alpha, [], unitary)

    mutable_alpha = np.array(2.5)
    encoding = BlockEncoding(mutable_alpha, [], unitary)
    mutable_alpha[...] = 9.0
    assert encoding.alpha == 2.5
    assert not isinstance(encoding.alpha, np.ndarray)
