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

"""Utilities for packing block-encoding ancillas into shared workspaces."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from qrisp.block_encodings.predicates import _as_static_size
from qrisp.core import QuantumVariable
from qrisp.jasp import qache
from qrisp.jasp.tracing_logic import QuantumVariableTemplate
from qrisp.qtypes import QuantumFloat

# Built once outside any tracing session so _qf_template never allocates
# throwaway qubits in TracingQuantumSession or captures a traced qv_size.
_QF_PROTOTYPE = QuantumFloat(1).template()


@qache(static_argnums=0)
def _apply_qached(unitary: Callable[..., None], *args: Any) -> None:
    """Invoke ``unitary`` behind a qache boundary keyed on ``unitary`` identity."""
    unitary(*args)


def _qf_template(size: Any) -> QuantumVariableTemplate:
    """Return a QuantumFloat template of ``size`` without allocating qubits."""
    template = copy.copy(_QF_PROTOTYPE)
    template.qv = copy.copy(_QF_PROTOTYPE.qv)
    template.qv_size = size
    return template


@dataclass(frozen=True)
class _AncillaLayout:
    """Describes typed ancillas packed consecutively into a shared workspace variable.

    Static sizes are kept as plain Python ints (so ``construct_views`` emits
    static slices); traced JAX sizes are preserved as-is for Jasp execution.
    """

    templates: tuple[QuantumVariableTemplate, ...]
    sizes: tuple[Any, ...]
    total_size: Any

    @classmethod
    def from_templates(
        cls, templates: list[QuantumVariableTemplate] | tuple[QuantumVariableTemplate, ...]
    ) -> _AncillaLayout:
        """Create a layout from an ordered collection of ancilla templates."""
        templates = tuple(templates)
        sizes = tuple(template.qv_size for template in templates)

        static_sizes = tuple(_as_static_size(size) for size in sizes)
        if all(size is not None for size in static_sizes):
            return cls(templates, static_sizes, sum(static_sizes))

        return cls(templates, sizes, sum(sizes, start=jnp.array(0, dtype=int)))

    @property
    def has_static_sizes(self) -> bool:
        """Return whether the layout is independent of any JAX trace."""
        return isinstance(self.total_size, int)

    def construct_views(self, shared_ancilla: QuantumVariable) -> list[QuantumVariable]:
        """Construct typed views into the beginning of ``shared_ancilla``."""
        # A plain int start keeps the slices static for static sizes, and still
        # promotes to a traced offset as soon as one size is a traced value.
        offset = 0
        views = []
        for template, size in zip(self.templates, self.sizes):
            views.append(template.construct(reg=shared_ancilla.reg[offset : offset + size]))
            offset += size
        return views


def _maximum_layout_size(layouts: list[_AncillaLayout] | tuple[_AncillaLayout, ...]) -> Any:
    """Return the largest total workspace size across ``layouts`` (Python int when all are static)."""
    if all(layout.has_static_sizes for layout in layouts):
        return max((layout.total_size for layout in layouts), default=0)

    maximum_size = jnp.array(0, dtype=int)
    for layout in layouts:
        maximum_size = jnp.maximum(maximum_size, layout.total_size)
    return maximum_size
