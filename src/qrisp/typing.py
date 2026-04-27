"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.core
import numpy as np

from qrisp.circuit.clbit import Clbit
from qrisp.circuit.qubit import Qubit

__all__ = ["QubitLike", "ClbitLike", "ArrayLike"]

QubitLike: TypeAlias = Qubit | int | list
"""Accepted as a qubit specifier in circuit methods and gate functions.

A single qubit can be identified either by its :class:`~qrisp.circuit.Qubit`
object or by its integer index within the circuit. A list of either
represents multiple qubits.
"""

ClbitLike: TypeAlias = Clbit | int | list
"""Accepted as a classical bit specifier in circuit methods.

A single classical bit can be identified either by its
:class:`~qrisp.circuit.Clbit` object or by its integer index within the
circuit. A list of either represents multiple classical bits.
"""

ArrayLike: TypeAlias = (
    int | float | complex | bool | np.ndarray | np.generic | jax.Array | jax.core.Tracer
)
"""A type for all array-like numeric data accepted by Qrisp.

Covers Python scalars, NumPy arrays and scalars, JAX arrays, and JAX tracers.
JAX tracers appear whenever Qrisp code runs inside a Jasp-traced function
(e.g. under ``@jaspify`` or ``jax.jit``).

Because this alias contains only concrete types, ``isinstance`` checks work
at runtime:

Examples
--------

>>> from qrisp import ArrayLike
>>> import numpy as np
>>> isinstance(3.14, ArrayLike)
True
>>> isinstance(np.array([1, 2, 3]), ArrayLike)
True
>>> isinstance(np.float32(1.0), ArrayLike)
True
"""
