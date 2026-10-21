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

from typing import TypeAlias

from jax.core import Tracer

from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable
from qrisp.jasp import DynamicQubitArray, check_for_tracing_mode, fuse_qb_array
from qrisp.jasp.primitives import AbstractQubit
from qrisp.qtypes import QuantumBool, QuantumChar, QuantumString

# An optional control/carry qubit may be passed as a QuantumBool or a bare Qubit.
OptionalQubit: TypeAlias = QuantumBool | Qubit


def _is_quantum_register(obj):
    """Return True if ``obj`` is a quantum register.

    A quantum register is a QuantumVariable (or subclass thereof), a
    DynamicQubitArray or a list of Qubits (including traced qubits).
    """
    if isinstance(obj, (QuantumVariable, DynamicQubitArray)):
        return True
    if isinstance(obj, list):
        return all(
            isinstance(qb, Qubit) or (isinstance(qb, Tracer) and isinstance(qb.aval, AbstractQubit)) for qb in obj
        )
    return False


def _reject_unsupported_quantum_types(a: object, b: object) -> None:
    """Reject quantum types that cannot be interpreted as additive registers.

    ``QuantumChar`` and ``QuantumString`` are quantum types but do not represent
    numeric registers, so silently applying the adder to their qubits would
    produce meaningless results. Both operands are checked and rejected if
    either one has an unsupported type.
    """
    for value in (a, b):
        if isinstance(value, (QuantumChar, QuantumString)):
            raise ValueError(
                f"{type(value).__name__} is not supported by this adder. Supported quantum types are "
                "QuantumVariable, QuantumBool, QuantumFloat and QuantumModulus."
            )


def _normalize_optional_input_arg(value: OptionalQubit | None, name: str) -> Qubit | None:
    """Normalize an optional QuantumBool or Qubit argument to a bare Qubit.

    The purpose of this helper is to centralize the coercion and validation of
    optional carry-in, carry-out and control parameters, each of which accepts
    a QuantumBool, a bare Qubit or None, while the gate logic works only with
    bare Qubits. Defining it once avoids repeating the same unwrapping and
    validation logic for every parameter.

    A QuantumBool is converted to its underlying Qubit via ``value[0]``, a bare
    Qubit or None is passed through unchanged, and any other type raises a
    TypeError in static mode. The helper neither touches the circuit nor
    allocates qubits, so it can run before any ancilla is created, letting an
    invalid argument fail cleanly instead of leaving a half-applied circuit
    behind.
    """
    if value is None:
        return None

    if isinstance(value, QuantumBool):
        return value[0]

    if not check_for_tracing_mode() and not isinstance(value, Qubit):
        raise TypeError(f"{name} must be of type QuantumBool or Qubit, not {type(value)}")

    return value


def _binary_string_to_int(value: str) -> int:
    """Parse a little-endian binary string into an integer.

    Binary strings are read little-endian, i.e. the leftmost character is the
    least significant bit, so ``"10"`` means bit 0 is 1 (the value 1). An empty
    string represents 0. Only the characters ``0`` and ``1`` are allowed.

    Parameters
    ----------
    value : str
        The little-endian binary string to parse.

    Returns
    -------
    int
        The integer encoded by ``value``.

    Raises
    ------
    ValueError
        If ``value`` contains characters other than ``0`` and ``1``.

    """
    if not set(value) <= {"0", "1"}:
        raise ValueError(f"a must be a binary string of 0s and 1s (base 2), got {value!r}")
    return int(value[::-1], 2) if value else 0


def _normalize_list_operand(value):
    """Convert a ``list[Qubit]`` operand to a ``DynamicQubitArray`` while tracing.

    Python lists require concrete indices, so they cannot be sliced or indexed
    with the traced values used by :func:`jrange`. Lists are left unchanged in
    static mode, and other operand types (including existing
    ``DynamicQubitArray`` inputs) are returned as they are.
    """
    if not isinstance(value, list) or not check_for_tracing_mode() or not value:
        return value

    if not _is_quantum_register(value):
        return value

    if len(value) == 1:
        # Fuse the qubit with itself to obtain an array, then slice back to a
        # single element. Wrapping a bare qubit directly would not produce a
        # qubit-array value at runtime.
        return DynamicQubitArray(fuse_qb_array(value[0], value[0]))[:1]

    result = value[0]
    for qb in value[1:]:
        result = result + qb
    return result
