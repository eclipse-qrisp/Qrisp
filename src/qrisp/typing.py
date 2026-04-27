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

from typing import Sequence, TypeAlias

import jax
import jax.core
import numpy as np

from qrisp.circuit.clbit import Clbit
from qrisp.circuit.qubit import Qubit

__all__ = ["QubitLike", "ClbitLike", "ScalarLike", "NDArrayLike", "ArrayLike"]

QubitLike: TypeAlias = Qubit | int | Sequence[Qubit | int]
"""Accepted as a qubit specifier in circuit methods and gate functions.

A single qubit can be identified either by its ``Qubit`` object or by its
integer index within the circuit. A sequence of either represents multiple
qubits.
"""

ClbitLike: TypeAlias = Clbit | int | Sequence[Clbit | int]
"""Accepted as a classical bit specifier in circuit methods.

A single classical bit can be identified either by its ``Clbit`` object or by
its integer index within the circuit. A sequence of either represents multiple
classical bits.
"""

ScalarLike: TypeAlias = int | float | complex | bool | np.generic
"""A Python or NumPy scalar value.

Covers Python built-in scalars (``int``, ``float``, ``complex``, ``bool``) and
all NumPy scalar types (``np.float64``, ``np.int32``, etc. via ``np.generic``).

Examples
--------

>>> from qrisp import ScalarLike
>>> import numpy as np
>>> isinstance(3.14, ScalarLike)
True
>>> isinstance(np.float32(1.0), ScalarLike)
True
"""

NDArrayLike: TypeAlias = np.ndarray | jax.Array | jax.core.Tracer
"""A multi-dimensional array value.

Covers NumPy arrays, JAX arrays, and JAX tracers (the latter appear when Qrisp
code runs inside a Jasp-traced function, e.g. under ``@jaspify``). 

Examples
--------

>>> from qrisp import NDArrayLike
>>> import numpy as np
>>> isinstance(np.array([1, 2, 3]), NDArrayLike)
True
"""

ArrayLike: TypeAlias = ScalarLike | NDArrayLike
"""A scalar or multi-dimensional array value.

Union of :data:`ScalarLike` and :data:`NDArrayLike`. Use this type when a
parameter accepts either scalars or arrays. Use the narrower aliases when only
one kind is expected, to avoid spurious Pylance warnings about missing
attributes such as ``.shape``.

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
