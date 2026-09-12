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

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from qrisp.core import QuantumVariable
from qrisp.jasp.tracing_logic import QuantumVariableTemplate


def _as_static_size(size: Any) -> int | None:
    """Return ``size`` as a Python int, or ``None`` if it is only known at runtime."""
    if isinstance(size, jax.core.Tracer):
        return None
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _AncillaLayout:
    """Describe typed ancillas packed consecutively into one register.

    A layout stores the original templates and their individual register sizes.
    It can then reconstruct typed views into a shared workspace. The sizes may be
    traced JAX values, which keeps the same implementation usable in static and
    Jasp execution.

    Sizes that are known at build time are kept as plain Python ints rather than
    as JAX arrays. Inside a trace, even ``jnp.add`` of two compile-time constants
    returns a tracer, so deriving the sizes with ``jnp`` would make the layout
    depend on the surrounding trace. Static sizes keep the layout reusable across
    traces (see ``LinearCombinationBlockEncoding``, which caches its layouts) and
    let ``construct_views`` emit static slices instead of dynamic ones.
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
    """Return the largest total workspace size across the supplied layouts.

    The result is a plain Python int whenever every layout has static sizes, so
    that the shared workspace is allocated with a statically known size.
    """
    if all(layout.has_static_sizes for layout in layouts):
        return max((layout.total_size for layout in layouts), default=0)

    maximum_size = jnp.array(0, dtype=int)
    for layout in layouts:
        maximum_size = jnp.maximum(maximum_size, layout.total_size)
    return maximum_size
