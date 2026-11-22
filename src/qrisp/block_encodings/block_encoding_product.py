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

"""Implements the block encoding of a product of block encodings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from qrisp.block_encodings.ancilla_layout import (
    _AncillaLayout,
    _apply_qached,
    _maximum_layout_size,
    _qf_template,
)
from qrisp.block_encodings.block_encoding_base import (
    BlockEncoding,
    _ProductFactors,
    _ProductStrategy,
)
from qrisp.core import mcx
from qrisp.environments import conjugate, control
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumBool

# For m factors with a single QuantumBool ancilla each, the shared workspace needs
# one workspace qubit, a counter of (m - 1).bit_length() qubits and a temporary
# flag. That is fewer than the m qubits of separate ancillas only from m = 6 on,
# and separate ancillas need no counter arithmetic.
_MAX_SEPARATE_QUANTUM_BOOL_FACTORS = 5


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
        self.unitary = self._build_unitary()

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent reassignment of the authoritative factor representation."""
        if name in {"_factors", "_strategy", "unitary"} and hasattr(self, name):
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

    @property
    def alpha(self) -> ArrayLike:
        """Return the normalization derived from the product factors."""
        alpha: ArrayLike = 1
        for factor in self.factors:
            alpha = alpha * factor.alpha
        return alpha

    @property
    def _product_layouts(self) -> tuple[_AncillaLayout, ...]:
        """Return the packing of every factor's ancillas into the shared workspace."""
        return tuple(_AncillaLayout.from_templates(factor._anc_templates) for factor in self.factors)

    @property
    def _num_factors_with_ancillas(self) -> int:
        """Return how many factors own ancillas and can therefore leave garbage."""
        return sum(factor.num_ancs != 0 for factor in self.factors)

    @property
    def _uses_shared_workspace(self) -> bool:
        """Return whether factor ancillas are packed into a shared workspace with a counter."""
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
        if self._uses_shared_workspace:
            # The counter is incremented after every factor with ancillas except
            # the last one applied, and it must never wrap around to zero.
            counter_size = (self._num_factors_with_ancillas - 1).bit_length()
            workspace_size = _maximum_layout_size(self._product_layouts)
            return [_qf_template(counter_size), _qf_template(workspace_size)]

        return [template for factor in self.factors for template in factor._anc_templates]

    def _build_unitary(self) -> Callable[..., None]:
        """Build the unitary selected by the product implementation strategy."""
        return self._build_unitary_qubit_efficient() if self._uses_shared_workspace else self._build_unitary_separate()

    def _build_unitary_separate(self) -> Callable[..., None]:
        """Build the unitary that composes factors with separate ancillas."""
        factor_ancilla_counts = tuple(factor.num_ancs for factor in self.factors)
        total_ancillas = sum(factor_ancilla_counts)

        def unitary(*args):
            operands = args[total_ancillas:]
            offset = 0
            factor_args = []
            for factor, num_ancillas in zip(self.factors, factor_ancilla_counts):
                factor_args.append((factor.unitary, args[offset : offset + num_ancillas]))
                offset += num_ancillas

            for child_unitary, factor_ancillas in reversed(factor_args):
                _apply_qached(child_unitary, *factor_ancillas, *operands)

        return unitary

    def _build_unitary_qubit_efficient(self) -> Callable[..., None]:
        """Build the unitary using one shared workspace and a counter."""
        # Factors are applied in reverse order, so this is the last factor with
        # ancillas to be applied. No later factor touches the workspace, so the
        # final projection onto a zero workspace discards its garbage directly.
        last_with_ancillas = next(index for index, factor in enumerate(self.factors) if factor.num_ancs != 0)

        def unitary(counter, shared_workspace, *operands):
            factor_layouts = self._product_layouts
            zero_flag = QuantumBool()

            for index in reversed(range(len(self.factors))):
                factor_layout = factor_layouts[index]
                views = factor_layout.construct_views(shared_workspace)
                _apply_qached(self.factors[index].unitary, *views, *operands)

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
