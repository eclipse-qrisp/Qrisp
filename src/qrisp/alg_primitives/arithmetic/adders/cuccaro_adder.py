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

import jax.numpy as jnp

from qrisp.alg_primitives.arithmetic.adders.adder_utilities import _is_quantum_register
from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, mcx, x
from qrisp.environments import conjugate, custom_control
from qrisp.jasp import DynamicQubitArray, check_for_tracing_mode, jlen, jrange
from qrisp.misc import int_encoder
from qrisp.qtypes import QuantumBool, QuantumFloat


@custom_control
def cuccaro_adder(
    a: int | QuantumVariable | DynamicQubitArray | list,
    b: QuantumVariable | DynamicQubitArray | list,
    c_in: QuantumBool | Qubit | None = None,
    c_out: QuantumBool | Qubit | None = None,
    ctrl: QuantumBool | None = None,
) -> None:
    """In-place adder as introduced in https://arxiv.org/abs/quant-ph/0410184

    This function works in both static and dynamic modes. The allowed inputs are both quantum types or one classical
    type and one quantum type. All :ref:`QuantumTypes <QuantumTypes>` (e.g. QuantumFloat, QuantumBool, QuantumModulus,
    ...), as well as lists of Qubits and DynamicQubitArrays, are supported as quantum inputs. Note that when the first
    input is larger than the second input, the function will perform modulo addition (relative to the size of the second
    input) after the first input is truncated to be the same size as the second input.

    The custom control implementation is based on Theorem 2.12 of https://arxiv.org/abs/2407.20167

    .. note::

        If the first input is quantum and the second classical, the function cannot work as addition is
        performed "in-place" on the second input.


    Parameters
    ----------
    a : int or QuantumVariable or list[Qubit] or DynamicQubitArray
        The value that should be added.
    b : QuantumVariable or list[Qubit] or DynamicQubitArray
        The value that should be modified in the in-place addition.
    c_in : QuantumBool or Qubit, optional
        An optional carry in value. The default is None.
    c_out : QuantumBool or Qubit, optional
        An optional carry out value. The default is None.
    ctrl : QuantumBool, optional
        An optional control qubit. If provided, the addition is only applied
        when the control qubit is in the ``|1>`` state. The default is None.

    Raises
    ------
    TypeError
        If carry in or carry out is not of type QuantumBool or Qubit in static mode.
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

    The examples below show how to use
    :func:`~qrisp.alg_primitives.arithmetic.adders.cuccaro_adder`. Because ``a``
    and ``b`` are generic quantum variables, the adder works with any quantum
    type that can store a value, e.g. ``QuantumFloat``, ``QuantumVariable`` or
    ``QuantumModulus``.

    Static mode with both quantum inputs of equal size:

    >>> from qrisp import QuantumFloat, cuccaro_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> cuccaro_adder(a,b)
    >>> print(b)
    {9: 1.0}

    Static mode with a classical first input:

    >>> from qrisp import QuantumFloat, cuccaro_adder
    >>> b = QuantumFloat(4)
    >>> b[:] = 3
    >>> cuccaro_adder(5, b)
    >>> print(b)
    {8: 1.0}

    If the classical input is larger than the second input, it is truncated
    modulo ``2**len(b)``. Here, the 4-qubit `QuantumFloat` ``b`` can only hold
    values from 0 to 15, so the sum value 16 cannot be represented. Since 16
    wraps around to 0 in this `QuantumFloat`, adding 16 is equivalent to adding
    0:

    >>> b = QuantumFloat(4)
    >>> b[:] = 5
    >>> cuccaro_adder(16, b)
    >>> print(b)
    {5: 1.0}

    Static mode with a quantum first input larger than the second input. The
    first input is truncated to the size of the second input (i.e. addition is
    performed modulo ``2**len(b)``) by slicing off its high-order qubits. No
    qubits are added or removed, so ``a`` keeps its value and size:

    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(2)
    >>> a[:] = 9
    >>> b[:] = 2
    >>> cuccaro_adder(a, b)
    >>> print(a)
    {9: 1.0}
    >>> print(a.size)
    4
    >>> print(b)
    {3: 1.0}

    Static mode with a quantum first input smaller than the second input. The
    first input is temporarily padded with additional (ancilla) qubits. The
    helper ancillas are created in the ``|0>`` state and appended to ``a`` so
    that the adder can process a register as large as ``b``. They are deleted
    once the addition is done, so no extra qubits are left over. ``a`` itself is
    not modified:

    >>> a = QuantumFloat(2)
    >>> b = QuantumFloat(4)
    >>> a[:] = 3
    >>> b[:] = 2
    >>> cuccaro_adder(a, b)
    >>> print(a.size)
    2
    >>> print(b)
    {5: 1.0}

    Lists of :class:`~qrisp.circuit.Qubit` objects are supported as well. The
    slices ``a[:]`` and ``b[:]`` return the qubit registers of ``a`` and ``b``
    as plain lists of Qubits, which are passed to the adder instead of the
    QuantumFloat objects:

    >>> a = QuantumFloat(5)
    >>> b = QuantumFloat(4)
    >>> a[:] = 10
    >>> b[:] = 5
    >>> cuccaro_adder(a[:], b[:])
    >>> print(b)
    {15: 1.0}

    Addition with a carry-in and a carry-out qubit. ``c_in`` is an optional
    carry-in bit, flipped to ``|1>`` here with ``x``, so it adds an extra 1 to
    the sum. ``c_out`` records the overflow: it is set to ``True`` whenever the sum
    does not fit in ``b``. The total sum is 6 + 3 + 1 = 10, which wraps around
    in the 3-qubit ``b``, so ``b`` ends up with 10 mod 8 = 2 and ``c_out``
    holds the overflow:

    >>> from qrisp import QuantumBool, x
    >>> b = QuantumFloat(3)
    >>> b[:] = 6
    >>> c_in = QuantumBool()
    >>> x(c_in[0])
    >>> c_out = QuantumBool()
    >>> cuccaro_adder(3, b, c_in=c_in, c_out=c_out)
    >>> print(b)
    {2: 1.0}
    >>> print(c_out)
    {True: 1.0}

    Controlled addition. ``ctrl`` is an optional control qubit, flipped to ``|1>``
    here with ``x``. When ``ctrl`` is in the ``|1>`` state the addition is
    applied; otherwise ``b`` stays unchanged. Here the sum 5 + 3 = 8 wraps
    around in the 3-qubit ``b``, leaving 8 mod 8 = 0:

    >>> a = QuantumFloat(5)
    >>> b = QuantumFloat(5)
    >>> a[:] = 3
    >>> b[:] = 5
    >>> ctrl = QuantumBool()
    >>> x(ctrl[0])
    >>> cuccaro_adder(a, b, ctrl=ctrl)
    >>> print(b)
    {8: 1.0}

    Dynamic mode (inside a :func:`~qrisp.jasp.jaspify` decorated function):

    The examples above can also be run inside a :func:`~qrisp.jasp.jaspify`
    function. As ``b`` holds the result, ``measure`` is used to read it out. In
    static mode ``print(b)`` already simulates and shows the outcome, so no
    explicit measurement is needed. Inside a jaspified function, however, the
    result is a quantum state that has to be collapsed with ``measure`` before
    it can be returned as a classical value:

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
    >>> result = main()
    >>> result  # result is 9 (4 + 5 = 9)
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

    # convert the classical input to a quantum input
    if not _is_quantum_register(a):
        # truncate the classical value modulo 2**len(b) so that values larger than the
        # target register are handled via modulo addition
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

    max_size = jnp.maximum(dim_a, dim_b)

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

    ancilla = QuantumFloat(max_size)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        elif not check_for_tracing_mode() and not isinstance(c_in, Qubit):
            raise TypeError(f"c_in must be of type QuantumBool or Qubit, not {type(c_in)}")
        cx(c_in, ancilla[0])

    if c_out is not None:
        if isinstance(c_out, QuantumBool):
            ancilla2 = c_out[0]
        elif not check_for_tracing_mode() and not isinstance(c_out, Qubit):
            raise TypeError(f"c_out must be of type QuantumBool or Qubit, not {type(c_out)}")
        else:
            ancilla2 = c_out

    # first maj gate application
    cx(a[0], b[0])
    cx(a[0], ancilla[0])
    mcx([ancilla[0], b[0]], a[0])

    # iterator maj gate application

    for i in jrange(1, dim_a):
        cx(a[i], b[i])
        cx(a[i], a[i - 1])
        mcx([a[i - 1], b[i]], a[i])

    # cnot
    if c_out is not None:
        cx(a[-1], ancilla2)

    if ctrl is None:
        # iterator uma gate application
        for j in jrange(dim_a - 1):
            # reverse the iteration
            i = dim_a - j - 1

            x(b[i])
            cx(a[i - 1], b[i])
            mcx([a[i - 1], b[i]], a[i])
            x(b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        # last uma gate application
        x(b[0])
        cx(ancilla[0], b[0])
        mcx([ancilla[0], b[0]], a[0])
        x(b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])

    else:
        # iterator uma gate application
        for j in jrange(dim_a - 1):
            # reverse the iteration
            i = dim_a - j - 1

            mcx([a[i - 1], b[i]], a[i])
            mcx([ctrl, a[i - 1]], b[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])

        # last uma gate application
        mcx([ancilla[0], b[0]], a[0])
        mcx([ctrl, ancilla[0]], b[0])
        cx(a[0], ancilla[0])
        cx(a[0], b[0])

    if c_in is not None:
        cx(c_in, ancilla[0])

    # delete the ancilla used for carry bits
    ancilla.delete()

    # delete the extension ancillas when the inputs are of unequal length
    extension_anc_a.delete()
