"""Utilities for packing block-encoding ancillas into shared workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from qrisp.core import QuantumVariable
from qrisp.jasp.tracing_logic import QuantumVariableTemplate


@dataclass(frozen=True)
class _AncillaLayout:
    """Describe typed ancillas packed consecutively into one register.

    A layout stores the original templates and their individual register sizes.
    It can then reconstruct typed views into a shared workspace. The sizes may be
    traced JAX values, which keeps the same implementation usable in static and
    Jasp execution.
    """

    templates: tuple[QuantumVariableTemplate, ...]
    sizes: tuple[Any, ...]
    total_size: Any

    @classmethod
    def from_templates(cls, templates: list[QuantumVariableTemplate] | tuple[QuantumVariableTemplate, ...]):
        """Create a layout from an ordered collection of ancilla templates."""
        templates = tuple(templates)
        sizes = tuple(template.qv_size for template in templates)
        total_size = sum(sizes, start=jnp.array(0, dtype=int))
        return cls(templates, sizes, total_size)

    def construct_views(self, shared_ancilla: QuantumVariable) -> list[QuantumVariable]:
        """Construct typed views into the beginning of ``shared_ancilla``."""
        offset = jnp.array(0, dtype=int)
        views = []
        for template, size in zip(self.templates, self.sizes):
            views.append(template.construct(reg=shared_ancilla.reg[offset : offset + size]))
            offset += size
        return views


def _maximum_layout_size(layouts: list[_AncillaLayout] | tuple[_AncillaLayout, ...]):
    """Return the largest total workspace size across the supplied layouts."""
    maximum_size = jnp.array(0, dtype=int)
    for layout in layouts:
        maximum_size = jnp.maximum(maximum_size, layout.total_size)
    return maximum_size
