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

"""Type aliases (QubitLike, ScalarLike, ArrayLike, FloatLike, etc.) used for type hints throughout Qrisp."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import jax
import jax.core
import numpy as np
from sympy import Expr

from qrisp.circuit.clbit import Clbit
from qrisp.circuit.qubit import Qubit

if TYPE_CHECKING:
    from qrisp.core.quantum_variable import QuantumVariable

__all__ = [
    "QubitLike",
    "ClbitLike",
    "ScalarLike",
    "NDArrayLike",
    "ArrayLike",
    "FloatLike",
    "ControlLike",
]

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

ScalarLike: TypeAlias = int | float | complex | bool | np.generic | jax.core.Tracer
"""A Python, NumPy, or JAX scalar value.

Covers Python built-in scalars, all NumPy scalar types, and JAX tracers.

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

Covers NumPy arrays, JAX arrays, and JAX tracers. 

Examples
--------

>>> from qrisp import NDArrayLike
>>> import numpy as np
>>> isinstance(np.array([1, 2, 3]), NDArrayLike)
True
"""

ArrayLike: TypeAlias = ScalarLike | NDArrayLike
"""A scalar or multi-dimensional array value.

Union of :data:`ScalarLike` and :data:`NDArrayLike`. Useful when a
parameter accepts either scalars or arrays.

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

FloatLike: TypeAlias = float | int | np.floating | np.integer | Expr | jax.Array | jax.core.Tracer
"""A gate parameter value.

Covers all types accepted as gate parameters throughout Qrisp: Python numeric
scalars (``float``, ``int``), NumPy floating-point and integer scalars
(``np.float64``, ``np.int32``, etc.), symbolic expressions
(``sympy.Symbol``, ``sympy.Expr``, and any SymPy expression), concrete JAX
arrays (``jax.Array``, including 0-d arrays), and JAX tracers.

Examples
--------

>>> from qrisp.typing import FloatLike
>>> import sympy
>>> isinstance(1.5, FloatLike)
True
>>> isinstance(sympy.Symbol("phi"), FloatLike)
True
"""

if TYPE_CHECKING:
    _Control: TypeAlias = Qubit | QuantumVariable | bool | np.bool_ | jax.Array | jax.core.Tracer

    ControlLike: TypeAlias = _Control | Sequence[_Control]
    """Accepted by :func:`qrisp.control`.

    Qubits select quantum control. Booleans, including JAX arrays and traced
    measurement results, select classical control.

    Examples
    --------

    >>> from qrisp import QuantumBool
    >>> from qrisp.typing import ControlLike
    >>> isinstance(QuantumBool(), ControlLike)
    True
    >>> isinstance(True, ControlLike)
    True
    """


def __getattr__(name: str) -> Any:
    """Build the aliases that need ``qrisp.core`` on first access (:pep:`562`).

    ``qrisp.jasp`` imports this module while ``qrisp.core`` is being initialized, so
    ``QuantumVariable`` can only be imported here. The unions match the
    ``TYPE_CHECKING`` definitions above.

    Parameters
    ----------
    name : str
        The requested attribute.

    Returns
    -------
    Any
        The alias, which is then cached in the module namespace.

    Raises
    ------
    AttributeError
        If ``name`` is not such an alias.

    """
    if name != "ControlLike":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from qrisp.core.quantum_variable import QuantumVariable

    control_operand = Qubit | QuantumVariable | bool | np.bool_ | jax.Array | jax.core.Tracer
    control_like = control_operand | Sequence[control_operand]
    globals()[name] = control_like
    return control_like
