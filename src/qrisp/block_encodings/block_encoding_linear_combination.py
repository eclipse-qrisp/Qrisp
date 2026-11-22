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

"""Implements the block encoding of a linear combination of block encodings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings.ancilla_layout import (
    _AncillaLayout,
    _apply_qached,
    _maximum_layout_size,
    _qf_template,
)
from qrisp.block_encodings.block_encoding_base import BlockEncoding, _LCUTerm, _LCUTerms
from qrisp.block_encodings.predicates import (
    _as_static_size,
    _is_non_negative_real,
    _is_real,
    _is_static_scalar,
    _validate_scalar,
)
from qrisp.core.gate_application_functions import gphase
from qrisp.environments import conjugate
from qrisp.jasp import q_switch
from qrisp.jasp.tracing_logic import QuantumVariableTemplate


def _conj(value: ArrayLike) -> ArrayLike:
    """Return the complex conjugate of a scalar while preserving Python/NumPy scalar types."""
    if isinstance(value, (jax.core.Tracer, jax.Array)):
        return jnp.conj(value)
    return value.conjugate()


def _validate_lcu_terms(terms: Sequence[_LCUTerm]) -> _LCUTerms:
    """Validate LCU ``(coefficient, block_encoding)`` pairs, flatten nested LCUs, and detach 0-d NumPy scalars."""
    checked: list[_LCUTerm] = []
    for coefficient, block_encoding in terms:
        if not isinstance(block_encoding, BlockEncoding):
            raise TypeError(f"Expected every item to be a BlockEncoding, but got {type(block_encoding).__name__}.")
        detached = _validate_scalar(coefficient, "every coefficient")
        if isinstance(block_encoding, LinearCombinationBlockEncoding):
            for inner_coeff, inner_be in block_encoding._get_lcu_terms():
                checked.append((detached * inner_coeff, inner_be))
        else:
            checked.append((detached, block_encoding))

    terms = tuple(checked)
    if not terms:
        raise ValueError("At least one block-encoding is required.")

    num_ops = terms[0][1].num_ops
    if any(block_encoding.num_ops != num_ops for _, block_encoding in terms):
        raise ValueError("All block-encodings must have the same number of operands.")
    return terms


def _make_lcu_branch(
    child_unitary: Callable[..., None],
    layout: _AncillaLayout,
    has_workspace: bool,
    phase: ArrayLike,
    applies_phase: bool,
) -> Callable[..., None]:
    """Build one SELECT branch that slices the shared workspace outside qache."""
    if not has_workspace:

        def branch(*operands):
            _apply_qached(child_unitary, *operands)
            if applies_phase:
                gphase(phase, operands[0][0])

        return branch

    def branch(shared_ancilla, *operands):
        _apply_qached(child_unitary, *layout.construct_views(shared_ancilla), *operands)
        if applies_phase:
            gphase(phase, operands[0][0])

    return branch


@register_pytree_node_class
class LinearCombinationBlockEncoding(BlockEncoding):
    r"""A block encoding represented by an immutable tuple of LCU terms.

    Each term is a ``(coefficient, block_encoding)`` pair. If the child
    block-encoding represents $A_i / \alpha_i$, the combined LCU uses
    ``coefficient * child.alpha`` as the coefficient of its child unitary.
    The resulting normalization is therefore
    ``sum(abs(coefficient * child.alpha))``.

    The child ancillas are packed into one shared workspace after one
    selector variable. The workspace is sized to the largest child layout,
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

    """

    def __init__(self, terms: Sequence[_LCUTerm]) -> None:
        r"""Initialize a linear-combination block encoding from weighted terms."""
        self._terms = _validate_lcu_terms(terms)
        self.unitary = self._build_unitary()

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the authoritative term representation."""
        if name in {"_terms", "unitary"} and hasattr(self, name):
            raise AttributeError("Linear-combination representation is immutable.")
        object.__setattr__(self, name, value)

    @property
    def terms(self) -> _LCUTerms:
        """The immutable weighted child block encodings."""
        return self._terms

    @property
    def _has_traced_coefficients(self) -> bool:
        """Return whether any coefficient or child normalization is a JAX tracer."""
        return any(
            isinstance(coefficient, jax.core.Tracer) or isinstance(block_encoding.alpha, jax.core.Tracer)
            for coefficient, block_encoding in self.terms
        )

    @property
    def _lcu_selector_size(self) -> int:
        """Return the number of qubits addressing the SELECT branches."""
        return (len(self.terms) - 1).bit_length()

    @property
    def _lcu_layouts(self) -> tuple[_AncillaLayout, ...]:
        """Return the packing of every child's ancillas into the shared workspace."""
        return tuple(_AncillaLayout.from_templates(block_encoding._anc_templates) for _, block_encoding in self.terms)

    @property
    def _lcu_coefficients(self) -> ArrayLike:
        r"""Return $\alpha_i c_i$ as a NumPy array when concrete, or a JAX array when traced."""
        if self._has_traced_coefficients:
            return jnp.array(
                [coefficient * block_encoding.alpha for coefficient, block_encoding in self.terms],
                dtype=complex,
            )

        return np.array(
            [np.asarray(coefficient) * np.asarray(block_encoding.alpha) for coefficient, block_encoding in self.terms],
            dtype=complex,
        )

    @property
    def _lcu_amplitudes(self) -> ArrayLike:
        """Return magnitude-only PREP amplitudes padded to the selector dimension."""
        coefficients = self._lcu_coefficients
        xp = np if isinstance(coefficients, np.ndarray) else jnp

        padded = xp.pad(coefficients, (0, (1 << self._lcu_selector_size) - len(self.terms)))
        magnitudes = xp.abs(padded)
        return xp.sqrt(magnitudes / xp.sum(magnitudes))

    @property
    def _lcu_phases(self) -> ArrayLike:
        r"""Return $\arg(c_i \alpha_i)$ for every term."""
        coefficients = self._lcu_coefficients
        xp = np if isinstance(coefficients, np.ndarray) else jnp
        return xp.angle(coefficients)

    @property
    def alpha(self) -> ArrayLike:
        """Return the normalization derived from the weighted child encodings."""
        coefficients = self._lcu_coefficients
        xp = np if isinstance(coefficients, np.ndarray) else jnp
        return xp.sum(xp.abs(coefficients))

    @property
    def _anc_templates(self) -> list[QuantumVariableTemplate]:
        if len(self.terms) == 1:
            return list(self.terms[0][1]._anc_templates)

        selector_size = self._lcu_selector_size
        workspace_size = _maximum_layout_size(self._lcu_layouts)
        if _as_static_size(workspace_size) == 0:
            return [_qf_template(selector_size)]
        return [_qf_template(selector_size), _qf_template(workspace_size)]

    def _build_unitary(self) -> Callable[..., None]:
        """Build the PREP-SELECT-PREP unitary once per instance."""
        if len(self.terms) == 1:
            coefficient, block_encoding = self.terms[0]

            def unitary(*args):
                _apply_qached(block_encoding.unitary, *args)
                effective_coefficient = coefficient * block_encoding.alpha
                if isinstance(effective_coefficient, jax.core.Tracer):
                    gphase(jnp.angle(effective_coefficient), args[-1][0])
                elif not _is_non_negative_real(effective_coefficient):
                    gphase(float(np.angle(effective_coefficient)), args[-1][0])

            return unitary

        def unitary(selector, *args):
            layouts = self._lcu_layouts
            has_workspace = _as_static_size(_maximum_layout_size(layouts)) != 0
            phases = self._lcu_phases

            branches: list[Callable[..., None]] = []
            for term_index, ((coefficient, block_encoding), layout) in enumerate(zip(self.terms, layouts)):
                effective_coefficient = coefficient * block_encoding.alpha
                applies_phase = isinstance(effective_coefficient, jax.core.Tracer) or not _is_non_negative_real(
                    effective_coefficient
                )
                branches.append(
                    _make_lcu_branch(
                        block_encoding.unitary,
                        layout,
                        has_workspace,
                        phases[term_index],
                        applies_phase,
                    )
                )

            with conjugate(prepare)(selector, self._lcu_amplitudes):
                q_switch(selector, branches, *args)

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
        """Return whether the LCU unitary is known to be Hermitian (real coefficients and Hermitian children)."""
        return all(_is_real(coefficient) and block_encoding.is_hermitian for coefficient, block_encoding in self.terms)

    def _get_lcu_terms(self) -> _LCUTerms:
        return self.terms

    def dagger(self) -> LinearCombinationBlockEncoding:
        """Return the linear-combination dagger with conjugated coefficients and child encodings."""
        return LinearCombinationBlockEncoding(
            tuple((_conj(coefficient), block_encoding.dagger()) for coefficient, block_encoding in self.terms)
        )

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten dynamic coefficients and child encodings; keep static scalars in aux_data."""
        children: list[Any] = []
        static_coeffs: list[Any] = []
        for coefficient, block_encoding in self.terms:
            if _is_static_scalar(coefficient):
                static_coeffs.append(coefficient)
            else:
                static_coeffs.append(None)
                children.append(coefficient)
            children.append(block_encoding)
        return tuple(children), tuple(static_coeffs)

    @classmethod
    def tree_unflatten(
        cls: type[LinearCombinationBlockEncoding],
        aux_data: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> LinearCombinationBlockEncoding:
        """Reconstruct a linear combination from flattened terms."""
        it = iter(children)
        terms = tuple((next(it) if static_coeff is None else static_coeff, next(it)) for static_coeff in aux_data)
        return cls(terms)
