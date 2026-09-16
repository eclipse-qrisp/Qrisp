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

"""Implements the block encoding of a linear combination of block encodings.

A ``LinearCombinationBlockEncoding`` holds the terms it was built from and derives
everything else -- the selector width, the ancilla layouts, the coefficients and
the normalization -- from them. The arithmetic that produces one lives in
block_encoding_arithmetic.py, which imports this module; the dependency runs one
way only, so this module knows nothing about the operators that build it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.errors import TracerArrayConversionError
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from qrisp.alg_primitives.state_preparation import prepare
from qrisp.block_encodings.ancilla_layout import (
    _AncillaLayout,
    _is_trace_independent,
    _maximum_layout_size,
    _template_of_size,
)
from qrisp.block_encodings.block_encoding_base import BlockEncoding, _LCUTerm, _LCUTerms
from qrisp.block_encodings.predicates import _is_non_negative_real, _is_real, _is_statically_zero
from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import gphase
from qrisp.environments import conjugate
from qrisp.jasp import q_switch, qache
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumFloat


def _validate_lcu_terms(terms: Sequence[_LCUTerm]) -> _LCUTerms:
    """Check the terms as supplied, and reject the ones that describe no encoding.

    The terms are taken as written. ``A - A`` builds the two-term combination it
    reads as, rather than being recognized as the zero operator and simplified
    away, so that what is applied is what was asked for.

    The shape check reads ``ndim`` instead of testing for a NumPy array, so that a
    JAX array or a traced one is rejected the same way rather than silently giving
    the derived amplitudes an extra axis. Objects without ``ndim`` are left alone,
    which keeps plain scalars and the placeholders JAX builds its argument metadata
    from out of the check.

    A NumPy array coefficient is also detached here. The terms are the
    authoritative representation and everything else is derived from them and
    cached, so a coefficient the caller can still write to would leave those
    derived values describing a combination that no longer exists. Only NumPy
    arrays need this: every other kind of coefficient that reaches here, tracers
    included, is already immutable.
    """
    checked: list[_LCUTerm] = []
    for coefficient, block_encoding in terms:
        if not isinstance(block_encoding, BlockEncoding):
            raise TypeError("Expected every item to be a BlockEncoding.")
        if getattr(coefficient, "ndim", 0) != 0:
            raise ValueError(
                f"Expected every coefficient to be a scalar, but got an array of shape {coefficient.shape}."
            )
        detached = coefficient.item() if isinstance(coefficient, np.ndarray) else coefficient
        checked.append((detached, block_encoding))

    terms = tuple(checked)
    if not terms:
        raise ValueError("At least one block-encoding is required.")

    num_ops = terms[0][1].num_ops
    if any(block_encoding.num_ops != num_ops for _, block_encoding in terms):
        raise ValueError("All block-encodings must have the same number of operands.")
    return terms


def _identity_lcu_branch(shared_ancilla: QuantumVariable, *operands: QuantumVariable) -> None:
    """Pad the SELECT to a power of two; selected only for zero-amplitude indices."""


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
        self._terms = _validate_lcu_terms(terms)

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
        """Return the PREP amplitudes, padded to the selector dimension.

        These are magnitudes only: a term's argument is applied by its SELECT
        branch instead, which keeps PREP real and non-negative whatever the
        coefficients are. See :attr:`_lcu_phases`.
        """
        coefficients = self._lcu_coefficients
        xp = np if isinstance(coefficients, np.ndarray) else jnp

        padded = xp.pad(coefficients, (0, (1 << self._lcu_selector_size) - len(self.terms)))
        magnitudes = xp.abs(padded)
        return xp.sqrt(magnitudes / xp.sum(magnitudes))

    @property
    def _lcu_phases(self) -> ArrayLike:
        r"""Return $\arg(c_i \alpha_i)$ for every term.

        A term contributes ``coefficient * child.alpha`` to the encoded operator.
        Splitting that into a magnitude and an argument lets PREP act on the
        magnitudes alone, with the argument applied inside the term's SELECT
        branch, where the selector turns it into a relative phase.

        Keeping PREP real and non-negative is what allows the uncomputation to be
        a conjugation rather than a separately traced inverse, and it is also what
        makes the construction Hermitian for real coefficients.
        """
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

    def _make_lcu_branch(
        self,
        child_unitary: Callable[..., None],
        layout: _AncillaLayout,
        term_index: int,
    ) -> Callable[..., None]:
        """Build one SELECT branch applying ``child_unitary`` to views into the workspace.

        The shared workspace is taken as the branch's first operand rather than being
        captured from the enclosing scope, so the branch can be built once, ahead of
        any tracing, and reused for every invocation of the LCU unitary.

        The branch also carries its term's argument, which the selector turns into a
        relative phase. That value is read per call rather than captured here: for
        traced coefficients it is a tracer, and this closure outlives the trace that
        builds it, since the unitary holding it is cached. Whether a phase is applied
        at all is decided here, because a contribution that is known and non-negative
        needs no gate.
        """
        coefficient, block_encoding = self.terms[term_index]
        effective_coefficient = coefficient * block_encoding.alpha
        if isinstance(effective_coefficient, jax.core.Tracer):
            applies_phase = True
        else:
            applies_phase = not _is_non_negative_real(effective_coefficient)

        def branch(shared_ancilla, *operands):
            child_unitary(*layout.construct_views(shared_ancilla), *operands)
            if applies_phase:
                # Anchored on an operand qubit: the shared workspace is empty whenever
                # no child carries ancillas, and a global phase needs some qubit.
                gphase(self._lcu_phases[term_index], operands[0][0])

        branch.__name__ = f"lcu_branch_{term_index}"
        # Caching the branch body is what keeps the repeated tracing of q_switch cheap:
        # the tree-based q_switch traces every branch more than once while unrolling its
        # walk, and custom_control/custom_inversion speculatively trace the controlled
        # and inverted variants on top of that. With a qached body all but the first of
        # those become pjit cache hits.
        return qache(branch)

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

        if _is_statically_zero(self.alpha):
            # Every term contributes zero, so there is nothing for the unitary to
            # be: PREP would be asked for a state whose amplitudes are sqrt(0/0),
            # and the single-term path would otherwise apply the child as though
            # its coefficient were one. This is the only combination that is
            # refused. One that merely cancels, such as ``A - A``, has an ordinary
            # unitary built from terms with normalizations of their own; that it
            # encodes zero shows up as a postselection which never succeeds.
            raise ValueError("Cannot build the unitary for an all-zero linear combination.")

        if len(self.terms) == 1:
            coefficient, block_encoding = self.terms[0]
            child_unitary = block_encoding.unitary

            effective_coefficient = coefficient * block_encoding.alpha
            if isinstance(effective_coefficient, jax.core.Tracer):
                phase = jnp.angle(effective_coefficient)
                applies_phase = True
            else:
                phase = float(np.angle(effective_coefficient))
                # A contribution that is real and positive needs no gate at all.
                applies_phase = phase != 0.0

            def unitary(*args):
                child_unitary(*args)
                if applies_phase:
                    gphase(phase, args[0][0])

            if not isinstance(effective_coefficient, jax.core.Tracer) and block_encoding._has_reusable_unitary:
                object.__setattr__(self, "_cached_unitary", unitary)
            return unitary

        layouts = self._lcu_layouts
        child_unitaries = [block_encoding.unitary for _, block_encoding in self.terms]

        branches = [
            self._make_lcu_branch(child_unitary, layout, term_index)
            for term_index, (child_unitary, layout) in enumerate(zip(child_unitaries, layouts))
        ]
        # Padding to the full selector dimension covers the selector states that no
        # term addresses. Those states carry zero amplitude, so the branch chosen
        # for them never contributes; it only has to leave the operands untouched.
        branches += [_identity_lcu_branch] * ((1 << self._lcu_selector_size) - len(branches))

        # PREP always acts on magnitudes, because every term's argument is applied
        # by its own branch. The uncomputation is therefore always a conjugation,
        # which traces PREP once instead of tracing a separate inverse, and the
        # construction is Hermitian whenever the branch phases are signs.
        def unitary(*args):
            selector = args[0]
            shared_ancilla = args[1]
            operands = args[2:]

            with conjugate(prepare)(selector, self._lcu_amplitudes):
                q_switch(selector, branches, shared_ancilla, *operands)

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
        """Return whether the block-encoding unitary is known to be Hermitian.

        The attribute describes the unitary, not the encoded operator. PREP acts on
        magnitudes and SELECT applies each term's argument, so the unitary is
        ``PREP* (D SELECT) PREP`` with ``D`` the diagonal of those arguments. That
        is Hermitian when SELECT is, which needs Hermitian children, and when ``D``
        is real, which needs real coefficients. A sign is allowed; a genuine complex
        phase is not.
        """
        return all(_is_real(coefficient) and block_encoding.is_hermitian for coefficient, block_encoding in self.terms)

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
