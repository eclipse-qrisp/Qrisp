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

"""Shared helpers for all the Qrisp adder implementations."""

import jax.numpy as jnp
import numpy as np

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable
from qrisp.jasp import DynamicQubitArray, check_for_tracing_mode


def _extract_bit(a_int, digit_index):
    """Extract one bit from a classical scalar as a JAX boolean.

    Automatically detects BigInteger values by checking for a ``get_bit`` method.

    Parameters
    ----------
    a_int : int, jnp.ndarray scalar, or BigInteger
        Classical value whose bit is queried.
    digit_index : int
        Zero-based bit index to read (little-endian convention).

    Examples
    --------
    >>> bool(_extract_bit(0b1010, 1))
    True
    >>> bool(_extract_bit(0b1010, 0))
    False

    """
    # BigInteger (and other big-int wrappers) expose get_bit
    if hasattr(a_int, "get_bit"):
        return jnp.bool_(a_int.get_bit(digit_index))
    return jnp.bool_((a_int >> digit_index) & 1)


def _validate_adder_inputs(a, b):
    """Validate that ``(a, b)`` is a supported adder input pair.

    Supported combinations are classical-quantum (classical ``a``, quantum
    ``b``) and quantum-quantum (quantum ``a``, quantum ``b``).

    Returns
    -------
    a_is_quantum : bool
        Whether ``a`` is a quantum register.
    b_is_quantum : bool
        Whether ``b`` is a quantum register.

    Raises
    ------
    ValueError
        If the pair is not classical-quantum or quantum-quantum.

    """
    b_is_quantum = isinstance(b, (QuantumVariable, DynamicQubitArray)) or (
        isinstance(b, list) and len(b) > 0 and all(isinstance(qb, Qubit) for qb in b)
    )
    # Empty list is valid for a (treated as a zero-size quantum register).
    a_is_quantum = isinstance(a, (QuantumVariable, DynamicQubitArray)) or (
        isinstance(a, list) and all(isinstance(qb, Qubit) for qb in a)
    )

    is_valid_classical = isinstance(a, (int, np.integer, str)) or (
        check_for_tracing_mode()
        and (
            hasattr(a, "get_bit")
            or (getattr(a, "ndim", None) == 0 and jnp.issubdtype(getattr(a, "dtype", None), jnp.integer))
        )
    )
    if not (b_is_quantum and (a_is_quantum or is_valid_classical)):
        raise ValueError(
            "The adder expects inputs to be either classical-quantum "
            "(classical a, quantum b) or quantum-quantum (quantum a, quantum b)."
        )
    return a_is_quantum, b_is_quantum
