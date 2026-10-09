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

"""Implements the block encoding of a product of block encodings.

A ProductBlockEncoding stores its factors in mathematical order, which is
opposite to the order in which they are applied. For example, the product
``A @ B`` is stored as ``(A, B)`` but applies ``B`` first and then ``A``.
Everything else is derived from this factor sequence. The arithmetic that
produces a product lives in block_encoding_arithmetic.py, which imports this
module; the dependency runs one way only, so this module knows nothing about
the operators that build it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from qrisp.block_encodings.ancilla_layout import (
    _AncillaLayout,
    _is_trace_independent,
    _maximum_layout_size,
    _template_of_size,
)
from qrisp.block_encodings.block_encoding_base import (
    BlockEncoding,
    _ProductFactors,
    _ProductStrategy,
)
from qrisp.core import mcx
from qrisp.environments import conjugate, control
from qrisp.jasp import qache
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumBool, QuantumFloat

# For m factors with a single QuantumBool ancilla each, the shared workspace needs
# one workspace qubit, a counter of (m - 1).bit_length() qubits and a temporary
# flag. That is fewer than the m qubits of separate ancillas only from m = 6 on,
# and separate ancillas need no counter arithmetic.
_MAX_SEPARATE_QUANTUM_BOOL_FACTORS = 5


def _validate_product_factors(factors: _ProductFactors) -> _ProductFactors:
    factors = tuple(factors)
    for factor in factors:
        if not isinstance(factor, BlockEncoding):
            raise TypeError(f"Expected every factor to be a BlockEncoding, but got {type(factor).__name__}.")

    if factors:
        num_ops = factors[0].num_ops
        if any(factor.num_ops != num_ops for factor in factors):
            raise ValueError("All product factors must have the same number of operands.")
    return factors


def _make_product_step(child_unitary: Callable[..., None], factor_index: int) -> Callable[..., None]:
    """Build one product step applying ``child_unitary`` to the ancillas it is handed.

    Used by the separate strategy, where every factor owns its ancillas and simply
    receives them from the product's argument list.
    """

    def step(*args):
        child_unitary(*args)

    step.__name__ = f"product_step_{factor_index}"
    return qache(step)


def _make_product_workspace_step(
    child_unitary: Callable[..., None],
    layout: _AncillaLayout,
    factor_index: int,
) -> Callable[..., None]:
    """Build one product step applying ``child_unitary`` to views into the workspace.

    As for the LCU branches, the shared workspace is taken as the step's first
    operand rather than captured from the enclosing scope, so the step can be built
    once, ahead of any tracing, and reused for every invocation of the unitary.
    """

    def step(shared_workspace, *operands):
        child_unitary(*layout.construct_views(shared_workspace), *operands)

    step.__name__ = f"product_step_{factor_index}"
    return qache(step)


def _build_product_steps(
    factors: _ProductFactors,
    layouts: tuple[_AncillaLayout, ...] | None,
) -> tuple[Callable[..., None], ...]:
    """Build one qached step per factor, sharing the step between repeated factors.

    A factor may appear more than once in a product -- ``H @ P @ H`` is the
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

        try:
            key = hash((id(factor), child_unitary, layout)), id(factor), child_unitary, layout
        except TypeError:
            key = None

        step = steps_by_key.get(key) if key is not None else None
        if step is None:
            step = (
                _make_product_step(child_unitary, index)
                if layout is None
                else _make_product_workspace_step(child_unitary, layout, index)
            )
            if key is not None:
                steps_by_key[key] = step
        steps.append(step)

    return tuple(steps)


@register_pytree_node_class
class ProductBlockEncoding(BlockEncoding):
    """A block encoding represented by an immutable product of factors.

    Factors are stored in mathematical order. The unitary applies them in
    reverse order, so ``A @ B`` applies ``B`` before ``A``. By default, the
    qubit-efficient strategy packs the factor ancillas into one shared
    workspace and uses a counter to separate the garbage generated at
    successive steps (the compression gadget of Low and Wiebe,
    arXiv:1805.00675). The separate strategy retains distinct ancillas for
    every factor.

    Parameters
    ----------
    factors : Sequence[BlockEncoding]
        Block-encoding factors in mathematical order.
    strategy : {"separate", "qubit_efficient"}
        Unitary implementation strategy. ``"qubit_efficient"`` is the default;
        it reuses one workspace and records failed factor applications in a
        counter. It uses separate ancillas wherever sharing would not save
        qubits: when at most one factor owns ancillas, or when at most five do
        and each of them owns a single QuantumBool.
        ``"separate"`` uses distinct ancillas for every factor.

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
    def _num_factors_with_ancillas(self) -> int:
        """Return how many factors own ancillas and can therefore leave garbage."""
        return sum(factor.num_ancs != 0 for factor in self.factors)

    @property
    def _uses_shared_workspace(self) -> bool:
        """Return whether the factor ancillas are packed into one workspace with a counter.

        For factors owning at most one QuantumBool each, the choice depends on their
        number alone. Unlike the ancilla sizes, which are traced once the product is
        passed to a traced function, this number is always known statically.
        """
        num_factors_with_ancillas = self._num_factors_with_ancillas
        if self.strategy != "qubit_efficient" or num_factors_with_ancillas <= 1:
            return False

        only_quantum_bools = all(
            factor.num_ancs == 0 or (factor.num_ancs == 1 and isinstance(factor._anc_templates[0].qv, QuantumBool))
            for factor in self.factors
        )
        return not only_quantum_bools or num_factors_with_ancillas > _MAX_SEPARATE_QUANTUM_BOOL_FACTORS

    @property
    def _anc_templates(self) -> list[QuantumVariableTemplate]:
        cached = self.__dict__.get("_cached_anc_templates")
        if cached is not None:
            return list(cached)

        if self._uses_shared_workspace:
            # The counter is incremented after every factor with ancillas except
            # the last one applied, and it must never wrap around to zero.
            counter_size = (self._num_factors_with_ancillas - 1).bit_length()
            workspace_size = _maximum_layout_size(self._product_layouts)
            templates = (
                _template_of_size(QuantumFloat(counter_size), counter_size),
                _template_of_size(QuantumFloat(workspace_size), workspace_size),
            )
        else:
            templates = tuple(template for factor in self.factors for template in factor._anc_templates)

        if _is_trace_independent(templates):
            object.__setattr__(self, "_cached_anc_templates", templates)
        return list(templates)

    @property
    def unitary(self) -> Callable[..., None]:
        """Return the unitary selected by the product implementation strategy."""
        cached = self.__dict__.get("_cached_unitary")
        if cached is not None:
            return cached

        uses_shared_workspace = self._uses_shared_workspace
        unitary = self._build_unitary_qubit_efficient() if uses_shared_workspace else self._build_unitary_separate()

        cacheable = all(factor._has_reusable_unitary for factor in self.factors) and (
            not uses_shared_workspace or all(layout.has_static_sizes for layout in self._product_layouts)
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
        steps = _build_product_steps(self.factors, None)

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
        """Build the unitary using one shared workspace and a counter."""
        factor_layouts = self._product_layouts
        workspace_steps = _build_product_steps(self.factors, factor_layouts)
        # Factors are applied in reverse order, so this is the last factor with
        # ancillas to be applied. No later factor touches the workspace, so the
        # final projection onto a zero workspace discards its garbage directly.
        last_with_ancillas = next(index for index, factor in enumerate(self.factors) if factor.num_ancs != 0)

        def unitary(counter, shared_workspace, *operands):
            zero_flag = QuantumBool()

            for index in reversed(range(len(self.factors))):
                factor_layout = factor_layouts[index]
                workspace_steps[index](shared_workspace, *operands)

                if index == last_with_ancillas or len(factor_layout.sizes) == 0:
                    continue

                # Implement |s, w> -> |s + 1, w> for w != 0 and leave w = 0
                # fixed on the workspace used by this factor.
                # This permutation moves newly generated garbage out of
                # counter zero without modifying its workspace value.
                active_workspace = shared_workspace.reg[: factor_layout.total_size]
                with conjugate(mcx)(active_workspace, zero_flag, ctrl_state=0):
                    with control(zero_flag, ctrl_state=0):
                        counter += 1

            zero_flag.delete()

        return unitary

    @property
    def num_ops(self) -> int:
        """Return the common number of operands in the product factors."""
        return self.factors[0].num_ops

    @property
    def num_ancs(self) -> int:
        """Return the number of ancilla variables used by the strategy."""
        return len(self._anc_templates)

    @property
    def is_hermitian(self) -> bool:
        """Return whether the product is known to have a Hermitian unitary."""
        return len(self.factors) == 1 and self.factors[0].is_hermitian

    def _get_product_factors(self) -> _ProductFactors:
        return self.factors

    def dagger(self) -> ProductBlockEncoding:
        """Return the product dagger with reversed, individually Hermitian conjugated factors."""
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
