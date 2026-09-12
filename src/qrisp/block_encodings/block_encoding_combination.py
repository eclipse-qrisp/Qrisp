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

"""Implements block-encoding arithmetic and the composite encodings it produces.

``BlockEncoding`` in block_encoding_base.py defines what a block encoding *is*,
together with the ``_get_lcu_terms`` protocol by which an encoding reports how it
should be flattened into a linear combination. This module defines how block
encodings *combine*: the arithmetic operators, and the composite subclasses they
construct.

Splitting it this way keeps the package acyclic. The arithmetic needs the
composite classes in order to build them, and the composite classes need
``BlockEncoding`` in order to subclass it; keeping both here means the dependency
runs one way only, from this module to the base. The operators are attached to
``BlockEncoding`` in block_encoding.py, exactly as the constructors in
constructors/ and the transformations in transformations/ already are.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import NotImplementedType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import TracerArrayConversionError
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings.ancilla_layout import _AncillaLayout, _maximum_layout_size
from qrisp.block_encodings.block_encoding_base import (
    BlockEncoding,
    _LCUTerm,
    _LCUTerms,
    _ProductFactors,
    _ProductStrategy,
)
from qrisp.core import QuantumVariable, mcx
from qrisp.core.gate_application_functions import gphase
from qrisp.environments import conjugate, control, invert
from qrisp.jasp import q_switch, qache
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumBool, QuantumFloat


def _is_non_negative_real(value: Any) -> bool:
    try:
        value = np.asarray(value)
    except Exception:
        return False
    return bool(np.all(np.isreal(value)) and np.all(np.real(value) >= 0))


def build_linear_combination(
    cls,
    block_encodings: list[BlockEncoding],
    coefficients: "ArrayLike | None" = None,
) -> BlockEncoding:
    r"""Returns a BlockEncoding of a linear combination of operators.

    This method implements the linear combination $\sum_i\alpha_iA_i$ via the LCU
    (Linear Combination of Unitaries) framework, where $A_i$ are
    the operators encoded by the respective instances, and $\alpha_i$ are real coefficients.

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
    terms = _canonicalize_lcu_terms(terms)
    if len(terms) == 0:
        raise ValueError("Cannot construct a block encoding from an all-zero linear combination.")
    if len(terms) == 1:
        coefficient, block_encoding = terms[0]
        if isinstance(coefficient, (int, float, complex, np.number)) and coefficient == 1:
            return block_encoding
    return LinearCombinationBlockEncoding(terms)


def apply_add(self, other: BlockEncoding) -> BlockEncoding:
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


def apply_sub(self, other: BlockEncoding) -> BlockEncoding:
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


def apply_mul(self, other: "ArrayLike") -> BlockEncoding:
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
    - Multiplying by a scalar $c$ results in a new BlockEncoding of $cA$ by updating $\alpha \rightarrow c\alpha$.

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


def apply_matmul(self, other: BlockEncoding) -> BlockEncoding:
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
    - The ``@`` operator should be used sparingly, primarily to combine a few block encodings. For larger-scale polynomial transformations, Quantum Signal Processing (QSP) is the superior method.
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

    return ProductBlockEncoding(self._get_product_factors() + other._get_product_factors())


def apply_radd(self, other: Any) -> BlockEncoding | NotImplementedType:
    """Support adding a BlockEncoding to the start value used by sum()."""
    if other == 0:
        return self
    return NotImplemented


def apply_kron(self, other: BlockEncoding) -> BlockEncoding:
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
    - The ``kron`` operator maps the operands of self to the first set of operands and the operands of other to the remaining operands in a single unified unitary.
    - The ``kron`` operator should be used sparingly, primarily to combine a few block encodings.
    - A more qubit-efficient implementation of the Kronecker product can be found in `this paper <https://arxiv.org/pdf/2509.15779>`_ and will be implemented in future updates.

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

    def new_unitary(*args):
        self.unitary(*args)
        gphase(np.pi, args[0][0])

    return BlockEncoding(
        self.alpha,
        self._anc_templates,
        new_unitary,
        num_ops=self.num_ops,
        is_hermitian=self.is_hermitian,
    )


def _is_statically_zero(value: Any) -> bool:
    """Return whether ``value`` is a concrete scalar zero."""
    if isinstance(value, jax.core.Tracer):
        return False
    try:
        value = np.asarray(value)
    except Exception:
        return False
    return value.ndim == 0 and bool(value == 0)


def _canonicalize_lcu_terms(terms: Sequence[_LCUTerm]) -> _LCUTerms:
    """Merge identity-equal child encodings and remove concrete zero terms."""
    merged_terms: list[_LCUTerm] = []
    for coefficient, block_encoding in terms:
        for index, (_, existing_block_encoding) in enumerate(merged_terms):
            if existing_block_encoding is block_encoding:
                merged_terms[index] = (merged_terms[index][0] + coefficient, block_encoding)
                break
        else:
            merged_terms.append((coefficient, block_encoding))

    return tuple(
        (coefficient, block_encoding)
        for coefficient, block_encoding in merged_terms
        if not _is_statically_zero(coefficient)
    )


def _template_of_size(quantum_variable: QuantumVariable, size: Any) -> QuantumVariableTemplate:
    """Return a template for ``quantum_variable`` that records ``size`` statically.

    A template records the size of the register it constructs, and it reads that
    size off the variable, which inside a Jasp trace is a tracer. A template built
    that way cannot be reused outside the trace that produced it: handing it to a
    later transformation raises an UnexpectedTracerError. Whenever the size is
    already known as a plain int, recording that int instead keeps the template
    independent of any trace, and therefore cacheable.
    """
    template = quantum_variable.template()
    if isinstance(size, int):
        template.qv_size = size
    return template


def _is_trace_independent(templates: Sequence[QuantumVariableTemplate]) -> bool:
    """Return whether ancilla templates can be reused outside the trace that built them.

    Templates whose size is a tracer belong to the trace that built them and must
    not be cached; see :func:`_template_of_size`.
    """
    return all(not isinstance(template.qv_size, jax.core.Tracer) for template in templates)


def _make_lcu_branch(
    child_unitary: Callable[..., None],
    layout: _AncillaLayout,
    name: str,
) -> Callable[..., None]:
    """Build one SELECT branch applying ``child_unitary`` to views into the workspace.

    The shared workspace is taken as the branch's first operand rather than being
    captured from the enclosing scope, so the branch can be built once, ahead of
    any tracing, and reused for every invocation of the LCU unitary.
    """

    def branch(shared_ancilla, *operands):
        child_unitary(*layout.construct_views(shared_ancilla), *operands)

    branch.__name__ = name
    # Caching the branch body is what keeps the repeated tracing of q_switch cheap:
    # the tree-based q_switch traces every branch more than once while unrolling its
    # walk, and custom_control/custom_inversion speculatively trace the controlled
    # and inverted variants on top of that. With a qached body all but the first of
    # those become pjit cache hits.
    return qache(branch)


def _identity_lcu_branch(shared_ancilla: QuantumVariable, *operands: QuantumVariable) -> None:
    """Pad the SELECT to a power of two; selected only for zero-amplitude indices."""


def _make_product_step(child_unitary: Callable[..., None], name: str) -> Callable[..., None]:
    """Build one product step applying ``child_unitary`` to the ancillas it is handed.

    Used by the separate strategy, where every factor owns its ancillas and simply
    receives them from the product's argument list.
    """

    def step(*args):
        child_unitary(*args)

    step.__name__ = name
    return qache(step)


def _make_product_workspace_step(
    child_unitary: Callable[..., None],
    layout: _AncillaLayout,
    name: str,
) -> Callable[..., None]:
    """Build one product step applying ``child_unitary`` to views into the workspace.

    As for the LCU branches, the shared workspace is taken as the step's first
    operand rather than captured from the enclosing scope, so the step can be built
    once, ahead of any tracing, and reused for every invocation of the unitary.
    """

    def step(shared_workspace, *operands):
        child_unitary(*layout.construct_views(shared_workspace), *operands)

    step.__name__ = name
    return qache(step)


def _build_product_steps(
    factors: _ProductFactors,
    layouts: tuple[_AncillaLayout, ...] | None,
    name_prefix: str,
) -> tuple[Callable[..., None], ...]:
    """Build one qached step per factor, sharing the step between repeated factors.

    A factor may appear more than once in a product -- ``H.dagger() @ P @ H`` is the
    common case -- and the repeated occurrences apply the same unitary to the same
    ancillas. Giving them one step rather than one per position means the factor is
    traced once instead of once per occurrence.

    Steps are shared only when the key is hashable, which requires the factor to
    expose a stable unitary and the layout to have statically known sizes. A factor
    that rebuilds its unitary per access cannot be shared, and must not be.
    """
    steps: list[Callable[..., None]] = []
    steps_by_key: dict[Any, Callable[..., None]] = {}

    for index, factor in enumerate(factors):
        child_unitary = factor.unitary
        layout = layouts[index] if layouts is not None else None
        name = f"{name_prefix}_{index}"

        try:
            key = hash((child_unitary, layout)), child_unitary, layout
        except TypeError:
            key = None

        step = steps_by_key.get(key) if key is not None else None
        if step is None:
            step = (
                _make_product_step(child_unitary, name)
                if layout is None
                else _make_product_workspace_step(child_unitary, layout, name)
            )
            if key is not None:
                steps_by_key[key] = step
        steps.append(step)

    return tuple(steps)


@register_pytree_node_class
class LinearCombinationBlockEncoding(BlockEncoding):
    """A block encoding represented by an immutable tuple of LCU terms."""

    def __init__(self, terms: Sequence[_LCUTerm]) -> None:
        r"""Initialize a linear-combination block encoding from weighted terms.

        Each term is a ``(coefficient, block_encoding)`` pair. If the child
        block-encoding represents $A_i / \alpha_i$, the combined LCU uses
        ``coefficient * child.alpha`` as the coefficient of its child unitary.
        The resulting normalization is therefore
        ``sum(abs(coefficient * child.alpha))``.

        The child ancillas are packed into one shared workspace after one
        selector register. The workspace is sized to the largest child layout,
        and each selected branch reconstructs its typed ancillas as views into
        that workspace. Since the selector chooses exactly one child unitary,
        nested sums are lowered to one preparation/select/preparation sequence.
        The terms are retained as the immutable authoritative representation;
        the compiled unitary and metadata are derived from them.

        Parameters
        ----------
        terms : tuple[tuple[ArrayLike, BlockEncoding], ...]
            Weighted block-encoding terms. The terms must be non-empty, and all
            child block-encodings must have the same number of operands.

        Raises
        ------
        ValueError
            If no terms are supplied or the child operand counts differ.
        TypeError
            If a term does not contain a BlockEncoding.

        Notes
        -----
        A single term with coefficient ``1`` is returned as the original child
        by the factory that constructs this class. A single term with another
        coefficient uses the child's ancillas directly and applies the required
        phase for a negative coefficient.

        Everything derived from the terms — the unitary, its SELECT branches, the
        ancilla layouts and templates, the coefficients and the normalization — is
        built once and cached, and derived quantities that are known at build time
        are kept as NumPy values rather than JAX arrays. Both are required for the
        compilation cost to stay proportional to the size of the expression: Jasp's
        caches are keyed on object identity, and ``prepare`` selects its algorithm
        by whether its amplitude vector converts to NumPy. Derived values that
        capture a JAX tracer are not cached, since reusing them in a later trace
        would leak the tracer.

        """
        terms = _canonicalize_lcu_terms(terms)
        if len(terms) == 0:
            raise ValueError("Cannot construct a block encoding from an all-zero linear combination.")
        if any(not isinstance(block_encoding, BlockEncoding) for _, block_encoding in terms):
            raise TypeError("Expected every item to be a BlockEncoding.")

        num_ops = terms[0][1].num_ops
        if any(block_encoding.num_ops != num_ops for _, block_encoding in terms):
            raise ValueError("All block-encodings must have the same number of operands.")

        self._terms = terms

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the authoritative term representation."""
        if name == "_terms" and hasattr(self, "_terms"):
            raise AttributeError("Linear-combination terms are immutable.")
        object.__setattr__(self, name, value)

    @property
    def terms(self) -> _LCUTerms:
        """The immutable weighted child block encodings."""
        return self._terms

    #
    # Derived state
    #
    # Everything below is derived from the authoritative terms. It is cached on
    # first access, because rebuilding it would produce fresh Python objects every
    # time, and Jasp's caches (qache, and hence jax.jit) are keyed on object
    # identity. Without the cache, every enclosing q_switch re-traces the whole
    # subtree from scratch, and nested linear combinations multiply that cost once
    # per nesting level.
    #
    # A derived value is only cached when it is independent of any JAX trace.
    # Values that capture tracers must be rebuilt per trace: a tracer leaked into
    # a later trace raises a JAX leak error.
    #

    @property
    def _lcu_selector_size(self) -> int:
        """Return the number of qubits addressing the SELECT branches."""
        return (len(self.terms) - 1).bit_length()

    @property
    def _lcu_layouts(self) -> tuple[_AncillaLayout, ...]:
        """Return the packing of every child's ancillas into the shared workspace."""
        cached = self.__dict__.get("_cached_layouts")
        if cached is not None:
            return cached

        layouts = tuple(
            _AncillaLayout.from_templates(block_encoding._anc_templates) for _, block_encoding in self.terms
        )
        if all(layout.has_static_sizes for layout in layouts):
            object.__setattr__(self, "_cached_layouts", layouts)
        return layouts

    @property
    def _lcu_coefficients(self) -> ArrayLike:
        r"""Return $\alpha_i c_i$ for every term, i.e. the LCU coefficients of the child unitaries.

        The result is a concrete NumPy array whenever the coefficients and the child
        normalizations are known at build time. Staying out of ``jnp`` here is not a
        micro-optimization: inside a JAX trace even ``jnp.abs`` of a compile-time
        constant returns a tracer, and :func:`qrisp.prepare` dispatches on whether its
        amplitude vector converts to NumPy. A traced amplitude vector silently
        downgrades the state preparation from the single-gate ``prepare_qiskit`` path
        to the fully symbolic ``prepare_qswitch`` path, which costs orders of magnitude
        more equations and is applied twice (PREP and its inverse).

        The coefficients are genuinely traced when the block encoding crossed a jit
        boundary as a pytree, in which case the symbolic path is the correct one.
        """
        cached = self.__dict__.get("_cached_coefficients")
        if cached is not None:
            return cached

        try:
            coefficients = np.array(
                [
                    np.asarray(coefficient) * np.asarray(block_encoding.alpha)
                    for coefficient, block_encoding in self.terms
                ],
                dtype=complex,
            )
        except TracerArrayConversionError:
            return jnp.array(
                [coefficient * block_encoding.alpha for coefficient, block_encoding in self.terms],
                dtype=complex,
            )

        object.__setattr__(self, "_cached_coefficients", coefficients)
        return coefficients

    @property
    def _lcu_amplitudes(self) -> ArrayLike:
        """Return the PREP amplitudes, padded to the selector dimension."""
        coefficients = self._lcu_coefficients
        xp = np if isinstance(coefficients, np.ndarray) else jnp

        padded = xp.pad(coefficients, (0, (1 << self._lcu_selector_size) - len(self.terms)))
        return xp.sqrt(padded / xp.sum(xp.abs(padded)))

    @property
    def alpha(self) -> ArrayLike:
        """Return the normalization derived from the weighted child encodings."""
        coefficients = self._lcu_coefficients
        xp = np if isinstance(coefficients, np.ndarray) else jnp
        return xp.sum(xp.abs(coefficients))

    @property
    def _anc_templates(self) -> list[QuantumVariableTemplate]:
        cached = self.__dict__.get("_cached_anc_templates")
        if cached is not None:
            return list(cached)

        if len(self.terms) == 1:
            templates = tuple(self.terms[0][1]._anc_templates)
            if _is_trace_independent(templates):
                object.__setattr__(self, "_cached_anc_templates", templates)
            return list(templates)

        layouts = self._lcu_layouts
        selector_size = self._lcu_selector_size
        workspace_size = _maximum_layout_size(layouts)
        templates = (
            _template_of_size(QuantumFloat(selector_size), selector_size),
            _template_of_size(QuantumVariable(workspace_size), workspace_size),
        )
        if _is_trace_independent(templates):
            object.__setattr__(self, "_cached_anc_templates", templates)
        return list(templates)

    @property
    def unitary(self) -> Callable[..., None]:
        """Return the PREP-SELECT-PREP unitary derived from the terms.

        The returned callable is built once and cached, and so are the SELECT
        branches it dispatches to. Both matter for compilation cost: ``q_switch``
        traces each branch several times per call, and re-deriving the branches on
        every invocation would make each of those a full-price trace instead of a
        cache hit.
        """
        cached = self.__dict__.get("_cached_unitary")
        if cached is not None:
            return cached

        if len(self.terms) == 1:
            coefficient, block_encoding = self.terms[0]
            child_unitary = block_encoding.unitary

            def unitary(*args):
                child_unitary(*args)
                with control(coefficient < 0):
                    gphase(np.pi, args[0][0])

            if not isinstance(coefficient, jax.core.Tracer) and block_encoding._has_reusable_unitary:
                object.__setattr__(self, "_cached_unitary", unitary)
            return unitary

        layouts = self._lcu_layouts
        child_unitaries = [block_encoding.unitary for _, block_encoding in self.terms]

        branches = [
            _make_lcu_branch(child_unitary, layout, f"lcu_branch_{term_index}")
            for term_index, (child_unitary, layout) in enumerate(zip(child_unitaries, layouts))
        ]
        # Padding to the full selector dimension covers the selector states that no
        # term addresses. Those states carry zero amplitude, so the branch chosen
        # for them never contributes; it only has to leave the operands untouched.
        branches += [_identity_lcu_branch] * ((1 << self._lcu_selector_size) - len(branches))

        # PREP acts on real amplitudes whenever the linear combination has no
        # negative or complex coefficient, which lets the uncomputation be expressed
        # as a conjugation instead of a separately traced inverse.
        has_real_amplitudes = all(_is_non_negative_real(coefficient) for coefficient, _ in self.terms)

        def unitary(*args):
            selector = args[0]
            shared_ancilla = args[1]
            operands = args[2:]

            amplitudes = self._lcu_amplitudes
            xp = np if isinstance(amplitudes, np.ndarray) else jnp

            if has_real_amplitudes:
                with conjugate(prepare)(selector, xp.abs(amplitudes)):
                    q_switch(selector, branches, shared_ancilla, *operands)
            else:
                prepare(selector, amplitudes)
                q_switch(selector, branches, shared_ancilla, *operands)
                with invert():
                    prepare(selector, xp.conjugate(amplitudes))

        cacheable = all(layout.has_static_sizes for layout in layouts) and all(
            block_encoding._has_reusable_unitary for _, block_encoding in self.terms
        )
        if cacheable:
            object.__setattr__(self, "_cached_unitary", unitary)
        return unitary

    @property
    def num_ops(self) -> int:
        """Return the common number of operands in the child encodings."""
        return self.terms[0][1].num_ops

    @property
    def num_ancs(self) -> int:
        """Return the number of selector and shared-workspace ancillas."""
        return len(self._anc_templates)

    @property
    def is_hermitian(self) -> bool:
        """Return whether every child encoding has a Hermitian unitary."""
        return all(block_encoding.is_hermitian for _, block_encoding in self.terms)

    def _get_lcu_terms(self) -> _LCUTerms:
        return self.terms

    @property
    def _has_reusable_unitary(self) -> bool:
        """Return whether the derived unitary was cached and can therefore be reused."""
        return "_cached_unitary" in self.__dict__

    def tree_flatten(self) -> tuple[tuple[ArrayLike | BlockEncoding, ...], int]:
        """Flatten only the authoritative terms for JAX pytree handling."""
        children = tuple(value for term in self.terms for value in term)
        return children, len(self.terms)

    @classmethod
    def tree_unflatten(
        cls: type[LinearCombinationBlockEncoding],
        aux_data: int,
        children: tuple[ArrayLike | BlockEncoding, ...],
    ) -> LinearCombinationBlockEncoding:
        """Reconstruct a linear combination from flattened terms."""
        terms = tuple((children[index], children[index + 1]) for index in range(0, 2 * aux_data, 2))
        return cls(terms)


@register_pytree_node_class
class ProductBlockEncoding(BlockEncoding):
    """A block encoding represented by an immutable product of factors.

    Factors are stored in mathematical order. The unitary applies them in
    reverse order, so ``A @ B`` applies ``B`` before ``A``. By default, the
    qubit-efficient strategy packs the factor ancillas into one shared
    workspace and uses a shift register to separate the garbage generated at
    successive steps. The separate strategy retains distinct ancillas for
    every factor.

    Parameters
    ----------
    factors : Sequence[BlockEncoding]
        Block-encoding factors in mathematical order.
    strategy : {"separate", "qubit_efficient"}
        Unitary implementation strategy. ``"qubit_efficient"`` is the default;
        it reuses one workspace and records failed factor applications in a
        shift register. ``"separate"`` uses distinct ancillas for every factor.

    """

    def __init__(
        self,
        factors: Sequence[BlockEncoding],
        strategy: _ProductStrategy = "qubit_efficient",
    ) -> None:
        """Initialize a product block encoding with the given factors and strategy."""
        if strategy not in ("separate", "qubit_efficient"):
            raise ValueError(f"Unknown product strategy: {strategy}")

        flattened_factors: list[BlockEncoding] = []
        for factor in factors:
            if not isinstance(factor, BlockEncoding):
                raise TypeError(f"Expected every factor to be a BlockEncoding, but got {type(factor).__name__}.")
            flattened_factors.extend(factor._get_product_factors())

        if len(flattened_factors) == 0:
            raise ValueError("At least one product factor is required.")

        num_ops = flattened_factors[0].num_ops
        if any(factor.num_ops != num_ops for factor in flattened_factors):
            raise ValueError("All product factors must have the same number of operands.")

        self._factors = tuple(flattened_factors)
        self._strategy = strategy

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the authoritative factor representation."""
        if name in {"_factors", "_strategy"} and hasattr(self, name):
            raise AttributeError("Product representation is immutable.")
        object.__setattr__(self, name, value)

    @property
    def factors(self) -> _ProductFactors:
        """The immutable product factors in mathematical order."""
        return self._factors

    @property
    def strategy(self) -> _ProductStrategy:
        """The unitary implementation strategy used by the product."""
        return self._strategy

    #
    # Derived state
    #
    # As for LinearCombinationBlockEncoding, everything below is derived from the
    # authoritative factors and cached on first access. A product does not go
    # through q_switch, so it has no trace amplification of its own, but rebuilding
    # the unitary hands a fresh closure to every enclosing composite and re-traces
    # the whole subtree each time the product is applied. Values that capture a JAX
    # tracer are not cached, since reusing them in a later trace would leak it.
    #

    @property
    def alpha(self) -> ArrayLike:
        """Return the normalization derived from the product factors."""
        cached = self.__dict__.get("_cached_alpha")
        if cached is not None:
            return cached

        alpha = 1
        for factor in self.factors:
            alpha = alpha * factor.alpha

        if not isinstance(alpha, jax.core.Tracer):
            object.__setattr__(self, "_cached_alpha", alpha)
        return alpha

    @property
    def _product_layouts(self) -> tuple[_AncillaLayout, ...]:
        """Return the packing of every factor's ancillas into the shared workspace."""
        cached = self.__dict__.get("_cached_layouts")
        if cached is not None:
            return cached

        layouts = tuple(_AncillaLayout.from_templates(factor._anc_templates) for factor in self.factors)
        if all(layout.has_static_sizes for layout in layouts):
            object.__setattr__(self, "_cached_layouts", layouts)
        return layouts

    @property
    def _anc_templates(self) -> list[QuantumVariableTemplate]:
        cached = self.__dict__.get("_cached_anc_templates")
        if cached is not None:
            return list(cached)

        cacheable = True
        if self.strategy == "qubit_efficient":
            if len(self.factors) == 1:
                templates = tuple(self.factors[0]._anc_templates)
                cacheable = _is_trace_independent(templates)
            elif all(factor.num_ancs == 0 for factor in self.factors):
                templates = ()
            else:
                layouts = self._product_layouts
                shift_size = (len(self.factors) - 1).bit_length()
                workspace_size = _maximum_layout_size(layouts)
                templates = (
                    _template_of_size(QuantumFloat(shift_size), shift_size),
                    _template_of_size(QuantumVariable(workspace_size), workspace_size),
                )
                cacheable = _is_trace_independent(templates)
        else:
            templates = tuple(template for factor in self.factors for template in factor._anc_templates)
            cacheable = _is_trace_independent(templates)

        if cacheable:
            object.__setattr__(self, "_cached_anc_templates", templates)
        return list(templates)

    @property
    def unitary(self) -> Callable[..., None]:
        """Return the unitary selected by the product implementation strategy."""
        cached = self.__dict__.get("_cached_unitary")
        if cached is not None:
            return cached

        unitary = (
            self._build_unitary_qubit_efficient()
            if self.strategy == "qubit_efficient"
            else self._build_unitary_separate()
        )

        # The closure captures the factor layouts, so those have to be free of
        # traced sizes too, not just the factor unitaries it dispatches to.
        cacheable = all(factor._has_reusable_unitary for factor in self.factors) and all(
            layout.has_static_sizes for layout in self._product_layouts
        )
        if cacheable:
            object.__setattr__(self, "_cached_unitary", unitary)
        return unitary

    @property
    def _has_reusable_unitary(self) -> bool:
        """Return whether the derived unitary was cached and can therefore be reused."""
        return "_cached_unitary" in self.__dict__

    def _build_unitary_separate(self) -> Callable[..., None]:
        """Build the unitary that composes factors with separate ancillas."""
        factor_ancilla_counts = tuple(factor.num_ancs for factor in self.factors)
        total_ancillas = sum(factor_ancilla_counts)
        steps = _build_product_steps(self.factors, None, "product_step")

        def unitary(*args):
            operands = args[total_ancillas:]
            offset = 0
            step_args = []
            for step, num_ancillas in zip(steps, factor_ancilla_counts):
                step_args.append((step, args[offset : offset + num_ancillas]))
                offset += num_ancillas

            for step, factor_ancillas in reversed(step_args):
                step(*factor_ancillas, *operands)

        return unitary

    def _build_unitary_qubit_efficient(self) -> Callable[..., None]:
        """Build the unitary using one shared workspace and a shift register."""
        if len(self.factors) == 1 or all(factor.num_ancs == 0 for factor in self.factors):
            num_ancs = self.num_ancs
            steps = _build_product_steps(self.factors, None, "product_step")

            def unitary(*args):
                operands = args[num_ancs:]
                factor_ancillas = args[:num_ancs]
                for step in reversed(steps):
                    step(*factor_ancillas, *operands)

            return unitary

        factor_layouts = self._product_layouts
        workspace_steps = _build_product_steps(self.factors, factor_layouts, "product_step")

        def unitary(*args):
            shift_register = args[0]
            shared_workspace = args[1]
            operands = args[2:]
            zero_flag = QuantumBool()

            for factor_index in range(len(self.factors)):
                layout_index = len(self.factors) - 1 - factor_index
                factor_layout = factor_layouts[layout_index]

                # Only the shift-zero sector carries the still-valid product
                # branch. Other sectors contain garbage from earlier factors.
                with conjugate(mcx)(shift_register, zero_flag, ctrl_state=0):
                    with control(zero_flag):
                        workspace_steps[layout_index](shared_workspace, *operands)

                if factor_index == len(self.factors) - 1 or len(factor_layout.sizes) == 0:
                    continue

                # Implement |s, w> -> |s + 1, w> for w != 0 and leave w = 0
                # fixed on the workspace prefix used by this factor. This
                # reversible permutation moves newly generated garbage out of
                # shift zero without modifying its workspace value. The
                # compute-control-uncompute pattern restores the temporary
                # predicate qubit after every permutation.
                active_workspace = shared_workspace.reg[: factor_layout.total_size]
                with conjugate(mcx)(active_workspace, zero_flag, ctrl_state=0):
                    with control(zero_flag, ctrl_state=0):
                        shift_register += 1

            zero_flag.delete()

        return unitary

    @property
    def num_ops(self) -> int:
        """Return the common number of operands in the product factors."""
        return self.factors[0].num_ops

    @property
    def num_ancs(self) -> int:
        """Return the number of ancilla variables used by the strategy."""
        if self.strategy == "qubit_efficient":
            return len(self._anc_templates)
        return sum(factor.num_ancs for factor in self.factors)

    @property
    def is_hermitian(self) -> bool:
        """Return whether the product is known to have a Hermitian unitary."""
        return len(self.factors) == 1 and self.factors[0].is_hermitian

    def _get_product_factors(self) -> _ProductFactors:
        return self.factors

    def dagger(self) -> ProductBlockEncoding:
        """Return the product dagger with reversed, individually inverted factors."""
        return ProductBlockEncoding(
            tuple(factor.dagger() for factor in reversed(self.factors)),
            strategy=self.strategy,
        )

    def tree_flatten(self) -> tuple[_ProductFactors, _ProductStrategy]:
        """Flatten the authoritative product factors for JAX pytree handling."""
        return self.factors, self.strategy

    @classmethod
    def tree_unflatten(
        cls: type[ProductBlockEncoding],
        aux_data: _ProductStrategy,
        children: tuple[BlockEncoding, ...],
    ) -> ProductBlockEncoding:
        """Reconstruct a product from flattened factors."""
        return cls(children, strategy=aux_data)
