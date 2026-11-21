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

"""Build-time scalar predicates and validation helpers for block encodings."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _is_static_scalar(value: Any) -> bool:
    """Return whether ``value`` is a hashable Python or NumPy scalar (kept in pytree ``aux_data``)."""
    return isinstance(value, (int, float, complex, np.number))


def _validate_scalar(value: Any, label: str = "value", *, real_only: bool = False) -> Any:
    """Validate that ``value`` is a scalar, detach 0-d NumPy arrays, and optionally reject complex types."""
    if isinstance(value, Sequence):
        raise ValueError(f"Expected {label} to be a scalar, but got a {type(value).__name__}.")
    if getattr(value, "ndim", 0) != 0:
        raise ValueError(f"Expected {label} to be a scalar, but got an array of shape {value.shape}.")
    detached = value.item() if isinstance(value, np.ndarray) else value
    if real_only and (
        isinstance(detached, (complex, np.complexfloating))
        or (hasattr(detached, "dtype") and jnp.issubdtype(detached.dtype, jnp.complexfloating))
    ):
        raise ValueError(f"Expected {label} to be a real scalar, but got complex {type(detached).__name__}.")
    return detached


def _is_real(value: Any) -> bool:
    """Return whether ``value`` is known to be real (checking dtype for JAX tracers)."""
    if isinstance(value, jax.core.Tracer):
        return not jnp.issubdtype(value.dtype, jnp.complexfloating)
    try:
        return bool(np.all(np.isreal(value)))
    except Exception:
        return False


def _is_non_negative_real(value: Any) -> bool:
    try:
        value = np.asarray(value)
    except Exception:
        return False
    return bool(np.all(np.isreal(value)) and np.all(np.real(value) >= 0))


def _as_static_size(size: Any) -> int | None:
    """Return ``size`` as a Python int, or ``None`` if it is only known at runtime."""
    if isinstance(size, jax.core.Tracer):
        return None
    try:
        return int(size)
    except (TypeError, ValueError):
        return None
