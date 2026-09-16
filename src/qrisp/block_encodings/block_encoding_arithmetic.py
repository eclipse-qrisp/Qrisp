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

"""Implements the arithmetic operators of block encodings.

``BlockEncoding`` in block_encoding_base.py defines what a block encoding *is*,
together with the ``_get_lcu_terms`` protocol by which an encoding reports how it
should be flattened into a linear combination. This module defines how block
encodings *combine*: the operators, and the factories that build the composite
encodings they produce.

The composite encodings themselves live beside this module, one per kind, and are
imported from here. That keeps the package acyclic: the arithmetic needs them in
order to build them, and they need ``BlockEncoding`` in order to subclass it, so
the dependency runs one way only. The operators are attached to ``BlockEncoding``
in block_encoding.py, exactly as the constructors in constructors/ and the
transformations in transformations/ already are.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import NotImplementedType
from typing import Any

import numpy as np
from jax.typing import ArrayLike

from qrisp.block_encodings.block_encoding_base import BlockEncoding, _LCUTerm
from qrisp.block_encodings.block_encoding_linear_combination import (
    LinearCombinationBlockEncoding,
    _validate_lcu_terms,
)
from qrisp.block_encodings.block_encoding_product import ProductBlockEncoding, _validate_product_factors


# The functions below are written as methods but defined at module level, and
# block_encoding.py attaches them to BlockEncoding. pydocstyle only exempts a
# receiver argument for functions written inside a class body, so it reports the
# self and cls of each one as an undocumented parameter; the noqa markers below
# suppress that. Sphinx renders them as bound methods, so the receiver never
# reaches the reader and documenting it would only add noise.
def build_linear_combination(  # noqa: D417
    cls,
    block_encodings: list[BlockEncoding],
    coefficients: "ArrayLike | None" = None,
) -> BlockEncoding:
    r"""Returns a BlockEncoding of a linear combination of operators.

    This method implements the linear combination $\sum_i\alpha_iA_i$ via the LCU
    (Linear Combination of Unitaries) framework, where $A_i$ are
    the operators encoded by the respective instances, and $\alpha_i$ are scalar,
    potentially complex coefficients.

    Hermiticity is only retained for real coefficients: a complex coefficient makes
    the combination non-Hermitian even when every $A_i$ is Hermitian.

    Equivalently, block-encoded operators can also be combined via ``+``, ``-`` and scalar multiplication.

    Parameters
    ----------
    block_encodings : list[BlockEncoding]
        Block-encodings of operators acting on the same operands.
    coefficients : ArrayLike, optional
        Coefficients multiplying the encoded operators. Defaults to one
        for every block-encoding.

    Returns
    -------
    BlockEncoding
        A block-encoding of the linear combination.

    Raises
    ------
    ValueError
        If no block-encodings are supplied, the coefficient count does
        not match, or operand counts differ.
    TypeError
        If an item is not a BlockEncoding.

    """
    if len(block_encodings) == 0:
        raise ValueError("At least one block-encoding is required.")

    if coefficients is None:
        coefficients = [1] * len(block_encodings)
    elif len(coefficients) != len(block_encodings):
        raise ValueError("The number of coefficients must match the number of block-encodings.")

    terms = []
    for block_encoding, coefficient in zip(block_encodings, coefficients):
        if not isinstance(block_encoding, BlockEncoding):
            raise TypeError(f"Expected every item to be a BlockEncoding, but got {type(block_encoding).__name__}.")
        terms.append((coefficient, block_encoding))

    return cls._from_lcu_terms(terms)


def build_from_lcu_terms(cls, terms: Sequence[_LCUTerm]) -> BlockEncoding:
    """Build a linear-combination block encoding from weighted terms."""
    terms = _validate_lcu_terms(terms)
    if len(terms) == 1:
        coefficient, block_encoding = terms[0]
        if isinstance(coefficient, (int, float, complex, np.number)) and coefficient == 1:
            return block_encoding
    return LinearCombinationBlockEncoding(terms)


def apply_add(self, other: BlockEncoding) -> BlockEncoding:  # noqa: D417
    r"""Returns a BlockEncoding of the sum of two operators.

    This method implements the addition $A + B$ via the LCU
    (Linear Combination of Unitaries) framework, where $A$ and $B$ are
    the operators encoded by the respective instances.

    Parameters
    ----------
    other : BlockEncoding
        The BlockEncoding instance to be added.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the operator sum.

    Notes
    -----
    - Can only be used when both BlockEncodings have the same operand structure.
    - The ``+`` operator should be used sparingly, primarily to combine a few block encodings.
      For larger-scale polynomial transformations,
      Quantum Signal Processing (QSP) is the superior method.

    Examples
    --------
    Define two block-encodings and add them.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
        H2 = Z(0)*Z(1) + X(2)
        H3 = H1 + H2

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)
        BE3 = BlockEncoding.from_operator(H3)

        BE_add = BE1 + BE2

        def operand_prep():
            qv = QuantumFloat(3)
            return qv

        @terminal_sampling
        def main(BE):
            qv = BE.apply_rus(operand_prep)()
            return qv

        res_be3 = main(BE3)
        res_be_add = main(BE_add)
        print("Result from BE of H1 + H2: ", res_be3)
        print("Result from BE1 + BE2: ", res_be_add)
        # Result from BE of H1 + H2:  {0: 0.37878788804466035, 4: 0.37878788804466035, 3: 0.24242422391067933}
        # Result from BE1 + BE2:  {0: 0.37878789933341894, 4: 0.37878789933341894, 3: 0.24242420133316217}

    """
    if not isinstance(other, BlockEncoding):
        return NotImplemented

    return type(self)._from_lcu_terms(self._get_lcu_terms() + other._get_lcu_terms())


def apply_sub(self, other: BlockEncoding) -> BlockEncoding:  # noqa: D417
    r"""Returns a BlockEncoding of the difference between two operators.

    This method implements the subtraction $A - B$ via the LCU
    (Linear Combination of Unitaries) framework, where $A$ and $B$ are
    the operators encoded by the respective instances.

    Parameters
    ----------
    other : BlockEncoding
        The BlockEncoding instance to be subtracted.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding representing the operator difference.

    Notes
    -----
    - Can only be used when both BlockEncodings have the same operand structure.
    - The ``-`` operator should be used sparingly, primarily to combine a few block encodings.
      For larger-scale polynomial transformations,
      Quantum Signal Processing (QSP) is the superior method.

    Examples
    --------
    Define two block-encodings and subtract them.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
        H2 = Z(0)*Z(1) + X(2)
        H3 = H1 - H2

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)
        BE3 = BlockEncoding.from_operator(H3)

        BE_sub = BE1 - BE2

        def operand_prep():
            qv = QuantumFloat(3)
            return qv

        @terminal_sampling
        def main(BE):
            qv = BE.apply_rus(operand_prep)()
            return qv

        res_be3 = main(BE3)
        res_be_sub = main(BE_sub)
        print("Result from BE of H1 - H2: ", res_be3)
        print("Result from BE1 - BE2: ", res_be_sub)
        # Result from BE of H1 - H2:  {0: 0.37878788804466035, 4: 0.37878788804466035, 3: 0.24242422391067933}
        # Result from BE1 - BE2:  {0: 0.37878789933341894, 4: 0.37878789933341894, 3: 0.24242420133316217}

    """
    if not isinstance(other, BlockEncoding):
        return NotImplemented

    other_terms = tuple((-coefficient, block_encoding) for coefficient, block_encoding in other._get_lcu_terms())
    return type(self)._from_lcu_terms(self._get_lcu_terms() + other_terms)


def apply_mul(self, other: "ArrayLike") -> BlockEncoding:  # noqa: D417
    r"""Returns a BlockEncoding of the scaled operator.

    This method implements the scalar multiplication $c \cdot A$, where $A$
    is the operator encoded by this instance and $c$ is the
    provided scalar.

    Parameters
    ----------
    other : ArrayLike
        The scalar scaling factor (coefficient) to apply. Can be a Python float,
        a JAX/NumPy scalar, or a 0-dimensional array.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the scaled operator.

    Notes
    -----
    - Multiplying by a scalar $c$ encodes $cA$, multiplies the normalization by $|c|$,
      and applies the argument of $c$ as a phase.

    Examples
    --------
    Define two block-encodings and implement their scaled sum as a new block encoding.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        # Commuting operators H1 and H2
        H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
        H2 = Z(0)*Z(1) + X(2)
        H3 = 2*H1 + H2

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)
        BE3 = BlockEncoding.from_operator(H3)

        BE_mul = 2*BE1 + BE2
        BE_mul_r = BE1*2 + BE2

        def operand_prep():
            qv = QuantumFloat(3)
            return qv

        @terminal_sampling
        def main(BE):
            qv = BE.apply_rus(operand_prep)()
            return qv

        res_be3 = main(BE3)
        res_be_mul = main(BE_mul)
        res_be_mul_r = main(BE_mul_r)

        print("Result from BE of 2 * H1 + H2: ", res_be3)
        print("Result from 2 * BE1 + BE2: ", res_be_mul)
        print("Result from BE1 * 2 + BE2: ", res_be_mul_r)
        # Result from BE of 2 * H1 + H2:  {3.0: 0.5614033770142979, 0.0: 0.21929831149285103, 4.0: 0.21929831149285103}
        # Result from 2 * BE1 + BE2:  {3.0: 0.5614033770142979, 0.0: 0.21929831149285103, 4.0: 0.21929831149285103}
        # Result from BE1 * 2 + BE2:  {3.0: 0.5614033770142979, 0.0: 0.21929831149285103, 4.0: 0.21929831149285103}

    """
    if isinstance(other, ArrayLike):
        terms = [(other * coefficient, block_encoding) for coefficient, block_encoding in self._get_lcu_terms()]
        return type(self)._from_lcu_terms(terms)

    return NotImplemented


def apply_matmul(self, other: BlockEncoding) -> BlockEncoding:  # noqa: D417
    r"""Returns a BlockEncoding of the product of two operators.

    This method implements the operator product $A \cdot B$ by composing
    two BlockEncodings, where $A$ and $B$ are the operators encoded by the respective instances.

    Parameters
    ----------
    other : BlockEncoding
        The BlockEncoding instance to be multiplied.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding representing the operator product.

    Notes
    -----
    - Can only be used when both BlockEncodings have the same operand structure.
    - The ``@`` operator should be used sparingly, primarily to combine a few block encodings.
      For larger-scale polynomial transformations,
      Quantum Signal Processing (QSP) is the superior method.
    - The product of two Hermitian operators A and B is Hermitian if and only if they commute, i.e., AB = BA.

    Examples
    --------
    Define two block-encodings and multiply them.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        # Commuting operators H1 and H2
        H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
        H2 = Z(0)*Z(1) + X(2)
        H3 = H1 * H2

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)
        BE3 = BlockEncoding.from_operator(H3)

        BE_mul = BE1 @ BE2

        def operand_prep():
            qv = QuantumFloat(3)
            return qv

        @terminal_sampling
        def main(BE):
            qv = BE.apply_rus(operand_prep)()
            return qv

        res_be3 = main(BE3)
        res_be_mul = main(BE_mul)
        print("Result from BE of H1 * H2: ", res_be3)
        print("Result from BE1 @ BE2: ", res_be_mul)
        # Result from BE of H1 * H2:  {3.0: 0.5, 7.0: 0.5}
        # Result from BE1 @ BE2:  {3.0: 0.5, 7.0: 0.5}

    """
    if not isinstance(other, BlockEncoding):
        return NotImplemented
    factors = _validate_product_factors(self._get_product_factors() + other._get_product_factors())
    return ProductBlockEncoding(factors)


def apply_radd(self, other: Any) -> BlockEncoding | NotImplementedType:
    """Support adding a BlockEncoding to the start value used by sum()."""
    if other == 0:
        return self
    return NotImplemented


def apply_kron(self, other: BlockEncoding) -> BlockEncoding:  # noqa: D417
    r"""Returns a BlockEncoding of the Kronecker product (tensor product) of two operators.

    This method implements the operator $A \otimes B$, where $A$ and $B$ are
    the operators encoded by the respective instances. Following the
    construction in Chapter 10.2 in `Dalzell et al. <https://arxiv.org/abs/2310.03011>`_,
    the resulting BlockEncoding is formed by the tensor product of the underlying unitaries, $U_A \otimes U_B$.

    Parameters
    ----------
    other : BlockEncoding
        The BlockEncoding instance to be tensored.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding representing the tensor product $A \otimes B$.

    Notes
    -----
    - **Normalization**: The normalization factors ($\alpha$) are combined multiplicatively.
    - The ``kron`` operator maps the operands of self to the first set of operands and the
      operands of other to the remaining operands in a single unified unitary.
    - The ``kron`` operator should be used sparingly, primarily to combine a few block encodings.
    - A more qubit-efficient implementation of the Kronecker product can be found in
      `this paper <https://arxiv.org/pdf/2509.15779>`_ and will be implemented in future updates.

    Examples
    --------
    **Example 1:**

    Define two block-encodings and perform their Kronecker product.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H1 = X(0)*X(1) + 0.2*Y(0)*Y(1)
        H2 = Z(0)*Z(1) + X(2)

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)

        BE_composed = BE1.kron(BE2)

        n1 = H1.find_minimal_qubit_amount()
        n2 = H2.find_minimal_qubit_amount()

        def operand_prep():
            qv1 = QuantumVariable(n1)
            qv2 = QuantumVariable(n2)
            return qv1, qv2

        @terminal_sampling
        def main(BE):
            return BE.apply_rus(operand_prep)()

        result = main(BE_composed)
        print("Result from BE1.kron(BE2): ", result)

    **Example 2:**

    Perform multiple Kronecker products of block-encodings in sequence.

    ::

        from qrisp import *
        from qrisp.operators import X, Y, Z

        H1 = X(0)*X(1)
        H2 = Z(0)*Z(1)
        H3 = Y(0)*Y(1)

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)
        BE3 = BlockEncoding.from_operator(H3)

        # Compose BE1 with the composition of BE2 and BE3
        BE_composed = BE1.kron(BE2.kron(BE3))

        n1 = H1.find_minimal_qubit_amount()
        n2 = H2.find_minimal_qubit_amount()
        n3 = H3.find_minimal_qubit_amount()

        def operand_prep():
            qv1 = QuantumVariable(n1)
            qv2 = QuantumVariable(n2)
            qv3 = QuantumVariable(n3)
            return qv1, qv2, qv3

        @terminal_sampling
        def main(BE):
            return BE.apply_rus(operand_prep)()

        result = main(BE_composed)
        print("Result from BE1.kron(BE2.kron(BE3)): ", result)

    """
    m = len(self._anc_templates)
    n = len(other._anc_templates)

    def new_unitary(*args):
        self_ancs = args[:m]
        other_ancs = args[m : m + n]
        operands = args[m + n :]

        self.unitary(*self_ancs, *operands[: self.num_ops])
        other.unitary(*other_ancs, *operands[self.num_ops :])

    new_anc_templates = self._anc_templates + other._anc_templates
    new_alpha = self.alpha * other.alpha
    return BlockEncoding(
        new_alpha,
        new_anc_templates,
        new_unitary,
        num_ops=self.num_ops + other.num_ops,
        is_hermitian=self.is_hermitian and other.is_hermitian,
    )


def apply_neg(self) -> BlockEncoding:
    r"""Returns a BlockEncoding of the negated operator.

    This method implements the transformation $A \to -A$ by scaling the
    encoded operator by $-1$.

    Returns
    -------
    BlockEncoding
        A new BlockEncoding instance representing the operator $-A$.

    Examples
    --------
    Define a block-encoding and negate it.

    ::

        from qrisp import *
        from qrisp.block_encodings import BlockEncoding
        from qrisp.operators import X, Y, Z

        H1 = X(0)*X(1) - 0.2*Y(0)*Y(1)
        H2 = 0.2*Y(0)*Y(1) - X(0)*X(1)

        BE1 = BlockEncoding.from_operator(H1)
        BE2 = BlockEncoding.from_operator(H2)
        BE3 = -BE1

        def operand_prep():
            qv = QuantumFloat(3)
            return qv

        @terminal_sampling
        def main(BE):
            qv = BE.apply_rus(operand_prep)()
            return qv

        res_be2 = main(BE2)
        res_be_neg = main(BE3)

        print("Result from BE of H2 = - H1: ", res_be2)
        print("Result from - BE1: ", res_be_neg)
        # Result from BE of H2 = - H1:  {3.0: 1.0}
        # Result from - BE1:  {3.0: 1.0}

    """
    terms = tuple((-coefficient, block_encoding) for coefficient, block_encoding in self._get_lcu_terms())
    return type(self)._from_lcu_terms(terms)
