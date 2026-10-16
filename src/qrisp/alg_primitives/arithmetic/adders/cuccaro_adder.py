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

"""Implements the Cuccaro ripple-carry in-place adder for quantum and classical-quantum addition."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, TypeAlias

import jax.numpy as jnp

from qrisp.alg_primitives.arithmetic.adders.adder_utilities import _is_quantum_register
from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, mcx, x
from qrisp.environments import conjugate, custom_control
from qrisp.jasp import DynamicQubitArray, check_for_tracing_mode, jlen, jrange
from qrisp.misc import int_encoder
from qrisp.qtypes import QuantumBool

# ``BigInteger`` only appears in type hints (lazy-evaluated thanks to
# ``from __future__ import annotations``) and is never needed at runtime.
# Importing it at module level can trigger a circular import.
if TYPE_CHECKING:
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
        BigInteger,
    )

# An optional control/carry qubit may be passed as a QuantumBool or a bare Qubit.
OptionalQubit: TypeAlias = QuantumBool | Qubit

# The quantum operands are QuantumVariables (or subclasses), DynamicQubitArrays
# or plain lists of Qubits (including traced qubits).
QuantumRegister: TypeAlias = QuantumVariable | DynamicQubitArray | list[Qubit]


def _normalize_optional_input_arg(value: OptionalQubit | None, name: str) -> Qubit | None:
    """Normalize an optional QuantumBool or Qubit argument to a bare Qubit.

    The purpose of this helper is to centralize the coercion and validation of
    the optional ``c_in``, ``c_out`` and ``ctrl`` parameters of
    :func:`cuccaro_adder`, each of which accepts a QuantumBool, a bare Qubit or
    None, while the gate logic works only with bare Qubits. Defining it once
    avoids repeating the same unwrapping and validation logic for every
    parameter.

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
    string represents 0.

    Parameters
    ----------
    value : str
        The little-endian binary string to parse.

    Returns
    -------
    int
        The integer encoded by ``value``.

    """
    return int(value[::-1], 2) if value else 0


def _apply_maj_gates(
    a: QuantumRegister,
    b: QuantumRegister,
    ancilla: QuantumVariable,
    dim_a: int,
) -> None:
    """Apply the majority (maj) gate chain of the Cuccaro adder.

    The maj gates compute the carry bits into the ``ancilla`` (reusing the
    ``a`` qubits as scratch space) while leaving the sum bits of ``b``
    untouched. The first few gates set up the carry at the least significant
    bit, and the loop then propagates it upward across the remaining ``dim_a``
    bits.
    """
    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])


def _apply_uma_gates(
    a: QuantumRegister,
    b: QuantumRegister,
    ancilla: QuantumVariable,
    ctrl: Qubit | None,
    dim_a: int,
) -> None:
    """Apply the unmajority (uma) gate chain of the Cuccaro adder.

    The uma gates run in reverse over the qubits and use the carries stored in
    the previous phase to write the result of the addition back into ``b``.
    There are two variants: the uncontrolled version toggles qubits with ``x``
    to save an ancilla, while the controlled version (``ctrl`` given) replaces
    those X gates with Toffolis controlled on ``ctrl`` so the addition only
    happens when the control is |1>.
    """
    if ctrl is None:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1

            x(b[i])
            cx(a[i - 1], b[i])
            mcx([a[i - 1], b[i]], a[i])
            x(b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        x(b[0])
        cx(ancilla[0], b[0])
        mcx([ancilla[0], b[0]], a[0])
        x(b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])

    else:
        for j in jrange(dim_a - 1):
            i = dim_a - j - 1

            mcx([a[i - 1], b[i]], a[i])
            mcx([ctrl, a[i - 1]], b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        mcx([ancilla[0], b[0]], a[0])
        mcx([ctrl, ancilla[0]], b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])


def _apply_c_out(c_out: Qubit | None, a: QuantumRegister, ctrl: Qubit | None) -> None:
    """Copy the final carry into the carry-out qubit.

    After the maj phase the most significant carry still sits in the top ``a``
    qubit. If a carry-out qubit was requested, this helper copies it over
    before the uma phase uncomputes the carries. For a controlled addition,
    the copy is also controlled so the carry-out remains unchanged when the
    control is disabled.
    """
    if c_out is not None:
        if ctrl is None:
            cx(a[-1], c_out)
        else:
            mcx([ctrl, a[-1]], c_out)


def _apply_c_in_cx(c_in: Qubit | None, ancilla: QuantumVariable) -> None:
    """Toggle the carry ancilla with the carry-in value.

    ``_normalize_optional_input_arg`` resolves the carry-in without touching the circuit.
    This helper applies the CNOT that seeds the carry ancilla with the carry-in
    value before the maj phase, and is applied a second time after the uma phase
    to clear the ancilla again so it can be safely deleted.
    """
    if c_in is not None:
        cx(c_in, ancilla[0])


@custom_control
def cuccaro_adder(
    a: int | str | BigInteger | QuantumRegister,
    b: QuantumRegister,
    c_in: OptionalQubit | None = None,
    c_out: OptionalQubit | None = None,
    ctrl: OptionalQubit | None = None,
) -> None:
    """In-place adder as introduced in https://arxiv.org/abs/quant-ph/0410184

    This function works in both static and dynamic modes.

    The two inputs may both be quantum, or one may be quantum and the other classical. All supported
    :ref:`quantum types <QuantumTypes>` (e.g. QuantumFloat, QuantumBool,
    QuantumModulus, ...) can be used as quantum inputs, along with lists of
    Qubits and :class:`~qrisp.jasp.DynamicQubitArray` objects.

    - A ``DynamicQubitArray`` is simply an array of qubits that is created automatically
      in Jasp mode; you never need to construct one yourself.
    - A quantum input can also be a list of bare Qubits. Bare Qubits are obtained by
      indexing a quantum variable, e.g. ``qv[0]`` yields a single Qubit and ``qv[0:2]``
      yields a list of Qubits; a bare Qubit may likewise be passed directly as the carry
      or control argument.

    The table below lists the accepted input forms and the behaviour in the special
    cases.

    .. list-table::
        :widths: 30 70
        :header-rows: 1

        * - Input
          - Behaviour
        * - Quantum first input ``a``
          - Any quantum type, ``list[Qubit]`` or ``DynamicQubitArray``. Its size
            sets the modulus of the addition.
        * - Quantum second input ``b``
          - Must be a quantum register (``QuantumVariable``, ``list[Qubit]`` or
            ``DynamicQubitArray``); it holds the result in place.
        * - Classical first input
          - An ``int``, a little-endian binary ``str``, or a ``BigInteger``
            (dynamic mode only). It is encoded into a temporary quantum variable
            of the same size as ``b`` and uncomputed afterwards, so no ancilla
            is left over. The second input cannot be classical, since the
            addition is performed in place on it.
        * - Classical addend larger than ``b``
          - Truncated modulo ``2**len(b)`` before it is encoded.
        * - Quantum ``a`` larger than ``b``
          - ``a`` is truncated by slicing off its high-order qubits, so the
            addition is performed modulo ``2**len(b)``. No qubits are added or
            removed and ``a`` keeps its value and size.
        * - Quantum ``a`` smaller than ``b``
          - ``a`` is padded with ancilla qubits in the ``|0>`` state so it can
            be processed at the size of ``b``. The ancillas are deleted once the
            addition is done, and ``a`` itself is not modified.
        * - ``QuantumBool``
          - A single-qubit register, so the addition wraps around modulo 2.
        * - ``QuantumModulus``
          - Works as long as the sum stays below the modulus.
        * - ``c_in`` and ``c_out``
          - Optional carry-in and carry-out; each accepts a ``QuantumBool`` or a
            bare ``Qubit``. ``c_in`` adds an extra 1 to the sum, and ``c_out``
            records the overflow.
        * - ``ctrl``
          - Optional control qubit; accepts a ``QuantumBool`` or a bare
            ``Qubit``. The addition is only applied when the control is in the
            ``|1>`` state. Equivalently, the call can be wrapped in a
            :func:`~qrisp.control` environment.
        * - Dynamic mode
          - Inside a :func:`~qrisp.jasp.jaspify` function the quantum inputs may
            be ``DynamicQubitArray`` objects such as ``a.reg`` or a full slice
            ``a[:]``. The result is read out with ``measure``.

    The custom control implementation is based on Theorem 2.12 of https://arxiv.org/abs/2407.20167

    .. note::

        If the first input is quantum and the second classical, the function cannot work as addition is
        performed "in-place" on the second input.


    Parameters
    ----------
    a : int, str, BigInteger, QuantumVariable, list[Qubit] or DynamicQubitArray
        The value that should be added. A ``str`` is interpreted as a
        little-endian binary string (``"10"`` means bit 0 is 1, i.e. the value
        1). A ``BigInteger`` is only supported in dynamic (tracing) mode.
    b : QuantumVariable or list[Qubit] or DynamicQubitArray
        The value that should be modified in the in-place addition.
    c_in : QuantumBool or Qubit, optional
        An optional carry in value. The default is None.
    c_out : QuantumBool or Qubit, optional
        An optional carry out value. The default is None.
    ctrl : QuantumBool or Qubit, optional
        An optional control qubit. If provided, the addition is only applied
        when the control qubit is in the ``|1>`` state. The default is None.

    Raises
    ------
    AttributeError
        If ``ctrl`` is not a QuantumBool or Qubit. The ``custom_control``
        decorator inspects the control qubit before the function body runs.
    TypeError
        If carry in or carry out is not of type QuantumBool or Qubit in static
        mode.
    TypeError
        If the first argument is classical but not an integer, binary string or
        BigInteger in static mode.
    ValueError
        If the first argument is a ``str`` that is not a valid binary string.
    ValueError
        If the second argument is not a quantum register, i.e. if ``b`` is not a
        QuantumVariable, DynamicQubitArray or a non-empty ``list[Qubit]``.
    ValueError
        If the first argument is a ``list`` that does not contain only Qubits.

    Returns
    -------
    None
        The function modifies the second input in place.

    Examples
    --------
    The examples below use ``QuantumFloat`` inputs; any other supported quantum
    type works the same way. In static mode, ``print`` already simulates and shows
    the outcome:

    >>> from qrisp import QuantumFloat, cuccaro_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> cuccaro_adder(a, b)
    >>> print(b)
    {9: 1.0}

    In dynamic mode, the call is wrapped in a :func:`~qrisp.jasp.jaspify`
    function and the result is collapsed with ``measure`` before it is returned:

    >>> from qrisp import QuantumFloat, cuccaro_adder, measure
    >>> from qrisp.jasp import jaspify
    >>> @jaspify
    ... def main():
    ...     a = QuantumFloat(4)
    ...     b = QuantumFloat(4)
    ...     a[:] = 4
    ...     b[:] = 5
    ...     cuccaro_adder(a, b)
    ...     return measure(b)
    >>> main()
    Array(9., dtype=float64)

    """
    # The second argument is required to be a (non-empty) quantum register
    if not _is_quantum_register(b) or (isinstance(b, list) and len(b) == 0):
        raise ValueError(
            "The second argument must be of type QuantumVariable, DynamicQubitArray or a non-empty list[Qubit]."
        )

    # A list that does not contain only Qubits is neither a valid quantum register
    # nor a valid classical input.
    if isinstance(a, list) and not _is_quantum_register(a):
        raise ValueError("If the first argument is a list, it must contain only Qubits.")

    # Normalize and validate the optional arguments before allocating any
    # ancillas or applying gates, so that an invalid argument raises cleanly
    # without leaving qubits or a half-applied circuit behind.
    c_in = _normalize_optional_input_arg(c_in, "c_in")
    c_out = _normalize_optional_input_arg(c_out, "c_out")
    ctrl = _normalize_optional_input_arg(ctrl, "ctrl")

    # convert the classical input to a quantum input
    if not _is_quantum_register(a):
        if isinstance(a, str):
            a = _binary_string_to_int(a)

        # int_encoder reads only the target's bits while tracing. In static mode,
        # validate the integer and truncate here because int_encoder rejects
        # values wider than the register.
        if not check_for_tracing_mode():
            if not isinstance(a, Integral):
                raise TypeError(
                    "a must be of type int, str, BigInteger, QuantumVariable, "
                    f"DynamicQubitArray or list[Qubit], not {type(a)}"
                )
            a = a % (1 << jlen(b))

        # create a quantum variable of the same size as the other quantum input
        q_a = QuantumVariable(jlen(b))

        with conjugate(int_encoder)(q_a, a):
            cuccaro_adder(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)

        # outside the conjugation, q_a is back in the state |0> and the addition has been performed on b
        # delete the temporary quantum variable created for the classical input
        q_a.delete()
        return

    # when the quantum inputs are of unequal length
    # pad the size of the input with the smaller size
    dim_a = jlen(a)
    dim_b = jlen(b)

    # reduce the size of a to the size of b if a is larger than b
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    # create an extension ancilla to change the size of a when it is smaller than b
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    extended_a = a[:] + extension_anc_a[:]
    a = extended_a

    # redefine the dimensions of a and b after the size adjustments
    dim_a = jlen(a)
    dim_b = jlen(b)

    # a single carry ancilla is sufficient for the Cuccaro adder
    ancilla = QuantumVariable(1)

    # seed the carry ancilla with the carry-in value
    _apply_c_in_cx(c_in, ancilla)

    # first maj gate application + iterator maj gate application
    _apply_maj_gates(a, b, ancilla, dim_a)

    # copy carry-out
    _apply_c_out(c_out, a, ctrl)

    # iterator + last uma gate application
    _apply_uma_gates(a, b, ancilla, ctrl, dim_a)

    # clear the carry-in seeding so the ancilla can be deleted
    _apply_c_in_cx(c_in, ancilla)

    # delete the ancilla used for carry bits
    ancilla.delete()

    # delete the extension ancillas when the inputs are of unequal length
    extension_anc_a.delete()
