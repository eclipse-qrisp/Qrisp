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

"""Build-time questions about the scalars a block encoding is built from.

Coefficients, normalizations and register sizes decide which construction a block
encoding uses, and that choice is made while tracing rather than while running. A
traced value cannot be inspected for its value, only for its dtype, so every
predicate here answers what is knowable at build time and errs towards the general
construction when it is not.

The module holds no block-encoding types and imports nothing from the package, so
any module in it can use these without regard to import order.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _is_statically_zero(value: Any) -> bool:
    """Return whether ``value`` is a concrete scalar zero."""
    if isinstance(value, jax.core.Tracer):
        return False
    try:
        value = np.asarray(value)
    except Exception:
        return False
    return value.ndim == 0 and bool(value == 0)


def _is_real(value: Any) -> bool:
    """Return whether a value is real, as far as is knowable at build time.

    A traced value has no value yet, but it does have a dtype, and a dtype that is
    not complex cannot carry an imaginary part. A complex dtype may still hold a
    real value at run time; reporting that as possibly complex is the safe
    direction, because the answer selects an implementation statically.

    An array counts as real only if every entry does, matching
    :func:`_is_non_negative_real`.
    """
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
