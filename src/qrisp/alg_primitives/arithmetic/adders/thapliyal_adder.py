# ********************************************************************************
# * Copyright (c) 2026 the Qrisp Authors
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

"""Implements the low-level Thapliyal adder circuit."""

from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np

from qrisp.alg_primitives.arithmetic.adders.gidney_adder import gidney_adder
from qrisp.circuit import Qubit
from qrisp.core import QuantumVariable, cx, mcx, p, rx, x
from qrisp.environments import conjugate, control, custom_control, invert
from qrisp.jasp import check_for_tracing_mode, jlen, jrange
from qrisp.misc import int_encoder
from qrisp.qtypes import QuantumBool

# Adder based on https://arxiv.org/abs/1712.02630


def _tr_gate(a: Qubit, b: Qubit, c: Qubit) -> None:
    """TR gate, as proposed in arXiv:1712.02630 (Section 3).

    (A, B, C) -> (A, A^B, A*(~B)^C). Used by the with-input-carry adder design
    (see thapliyal_procedure_with_carry); the no-input-carry design
    (thapliyal_procedure) uses the Peres gate instead (see _peres_gate).
    """
    with control(b):
        rx(-np.pi / 2, c)
    p(-np.pi / 4, b)
    cx(a, b)
    p(np.pi / 4, a)
    with control(a):
        rx(np.pi / 2, c)
    p(np.pi / 4, b)
    with control(b):
        rx(np.pi / 2, c)


def _peres_gate(a: Qubit, b: Qubit, c: Qubit) -> None:
    """Peres gate, as proposed in arXiv:1712.02630 (Section 2.5).

    (A, B, C) -> (A, A^B, A*B^C). Used by the no-input-carry adder design
    (see thapliyal_procedure); the with-input-carry design
    (thapliyal_procedure_with_carry) uses the TR gate instead (see _tr_gate).
    """
    with control(a):
        rx(-np.pi / 2, c)
    p(-np.pi / 4, a)
    with control(b):
        rx(-np.pi / 2, c)
    p(-np.pi / 4, b)
    cx(a, b)
    p(np.pi / 4, a)
    with control(b):
        rx(np.pi / 2, c)


def thapliyal_procedure(qubit_list_a: list[Qubit], qubit_list_b: list[Qubit], output_qubit_z: Qubit) -> None:
    """Apply the 6-step Thapliyal ripple-carry procedure to raw qubit lists.

    This is the *no-input-carry* construction of arXiv:1712.02630 (Section 4,
    "Methodology 1: Reversible Adder Circuit With No Input Carry"): this procedure
    has no carry-in slot, so it cannot accept one. thapliyal_adder uses this when
    c_in is None, and dispatches to thapliyal_procedure_with_carry (the paper's
    Section 5, "Methodology 2") instead when a carry-in is given.

    Uses free-function primitives (cx/mcx/_peres_gate) and jrange so the loop bounds
    may be traced values, making it usable in both static and dynamic (Jasp) mode.
    thapliyal_adder drives this helper, wrapping it with the full
    size-handling / c_in / c_out / ctrl API.

    qubit_list_a and qubit_list_b must have equal length; the loop count is derived
    from qubit_list_a via jlen (works for both static lists and traced sizes), the
    same way cuccaro_adder derives its own register size.

    Descending loops (steps 2 and 4) are rewritten as forward jrange loops with a
    computed index, mirroring the UMA-reversal pattern in cuccaro_adder.

    Step 2's paper range is "i = n-1 downto 1" (note: *not* down to 0), which is
    empty for a 1-qubit register. a_ext folds output_qubit_z into the same index
    space as qubit_list_a (a_ext[n] is output_qubit_z) so this loop can be written
    uniformly and correctly degrades to zero gates for n == 1, rather than
    special-casing the i=n-1 boundary as an unconditional gate outside the loop
    (which previously fired even for n == 1, corrupting output_qubit_z's carry-out
    for that case).

    a_ext is built via qubit_list_a + output_qubit_z (relying on Qubit/DynamicQubitArray's
    own __add__/__radd__), not list(qubit_list_a) + [...]: under Jasp tracing,
    qubit_list_a is a DynamicQubitArray, which defines neither __len__ nor __iter__, so
    Python's list() falls back to probing __getitem__(0), __getitem__(1), ... until an
    IndexError -- which DynamicQubitArray's __getitem__ never raises, making list() loop
    effectively forever (observed: multi-GB, non-terminating even for n == 1).
    """
    n = jlen(qubit_list_a)
    a_ext = qubit_list_a + output_qubit_z

    # Step 1
    for i in jrange(1, n):
        cx(qubit_list_a[i], qubit_list_b[i])

    # Step 2
    for j in jrange(n - 1):
        i = n - 1 - j
        cx(a_ext[i], a_ext[i + 1])

    # Step 3
    for i in jrange(n - 1):
        mcx([qubit_list_a[i], qubit_list_b[i]], qubit_list_a[i + 1])

    # Step 4 (Peres gate, per the paper)
    _peres_gate(qubit_list_a[-1], qubit_list_b[-1], output_qubit_z)

    for j in jrange(n - 1):
        i = n - 2 - j
        _peres_gate(qubit_list_a[i], qubit_list_b[i], qubit_list_a[i + 1])

    # Step 5
    for i in jrange(1, n - 1):
        cx(qubit_list_a[i], qubit_list_a[i + 1])

    # Step 6
    for i in jrange(1, n):
        cx(qubit_list_a[i], qubit_list_b[i])


def thapliyal_procedure_with_carry(
    qubit_list_a: list[Qubit], qubit_list_b: list[Qubit], c_in_qubit: Qubit, output_qubit_z: Qubit
) -> None:
    """Apply the 6-step Thapliyal ripple-carry procedure with a native carry-in.

    This is the *with-input-carry* construction of arXiv:1712.02630 (Section 5,
    "Methodology 2: Reversible Adder Circuit With Input Carry"). Unlike
    thapliyal_procedure (Methodology 1), c_in_qubit is a genuine slot in the ripple
    chain -- the paper's virtual bit position A_{-1} -- so, unlike the
    thapliyal_adder c_in path this replaces, no external carry-in synthesis or extra
    ancilla qubits are needed. c_in_qubit is restored to its original value by the
    end of the procedure, exactly like every bit of qubit_list_a.

    a_ext folds c_in_qubit into the same index space as qubit_list_a (a_ext[k] is
    the paper's A_{k-1}, so a_ext[0] is c_in_qubit and a_ext[k] is
    qubit_list_a[k-1] for k >= 1). Every step below then indexes uniformly through
    a_ext with no boundary-case branching -- in particular this makes step 3's
    Peres gate resolve correctly even for a 1-qubit register, where the paper's
    A_{n-2} position coincides with A_{-1} (i.e. a_ext[n-1] is c_in_qubit when
    n == 1).

    qubit_list_a and qubit_list_b must have equal length; the loop count is derived
    from qubit_list_a via jlen, mirroring thapliyal_procedure.

    a_ext is built via c_in_qubit + qubit_list_a (relying on Qubit/DynamicQubitArray's
    own __add__/__radd__), not [c_in_qubit] + list(qubit_list_a): under Jasp tracing,
    qubit_list_a is a DynamicQubitArray, which defines neither __len__ nor __iter__, so
    Python's list() falls back to probing __getitem__(0), __getitem__(1), ... until an
    IndexError -- which DynamicQubitArray's __getitem__ never raises, making list() loop
    effectively forever (observed: multi-GB, non-terminating even for the smallest case).
    """
    n = jlen(qubit_list_a)
    a_ext = c_in_qubit + qubit_list_a
    b = qubit_list_b

    # Step 1
    for i in jrange(n):
        cx(a_ext[i + 1], b[i])

    # Step 2
    for i in jrange(n):
        cx(a_ext[i + 1], a_ext[i])
    cx(a_ext[n], output_qubit_z)

    # Step 3
    for i in jrange(n - 1):
        mcx([a_ext[i], b[i]], a_ext[i + 1])
    _peres_gate(a_ext[n - 1], b[n - 1], output_qubit_z)

    for i in jrange(n - 1):
        x(b[i])

    # Step 4 (TR gate, per the paper)
    for j in jrange(n - 1):
        i = n - 2 - j
        _tr_gate(a_ext[i], b[i], a_ext[i + 1])

    for i in jrange(n - 1):
        x(b[i])

    # Step 5
    for j in jrange(n):
        i = n - 1 - j
        cx(a_ext[i + 1], a_ext[i])

    # Step 6
    for i in jrange(n):
        cx(a_ext[i + 1], b[i])


def _uncompute_thapliyal_carry(a: list[Qubit], b: QuantumVariable | list[Qubit], carry_qubit: Qubit) -> None:
    """Zeroes carry_qubit, given it currently holds (a > b) as an unsigned comparison.

    carry_qubit currently holds (a > b) as an unsigned, equal-bit-length
    comparison (this is what the ripple procedure leaves behind in its
    output_qubit once a is restored and b holds the final sum). a and b are
    left unmodified.

    Unlike cuccaro_adder's carry-out (a non-destructive tap on a self-restoring wire),
    the Thapliyal output_qubit is a genuinely exposed wire with no built-in
    undo step. This recomputes the same boolean value independently via a scratch
    comparison (same trick as uint_qq_less_than in uint_clifford_t_comparisons.py,
    using gidney_adder as the reversible scratch arithmetic since, unlike
    cuccaro_adder, it accepts a plain qubit-list target) and XORs it into
    carry_qubit to cancel it out.

    """
    comparison_anc = QuantumBool()

    def inv_gidney(x: list[Qubit], y: list[Qubit]) -> None:
        with invert():
            gidney_adder(x, y)

    with conjugate(inv_gidney, allocation_management=False)(a, b[:] + comparison_anc[:]):
        cx(comparison_anc[0], carry_qubit)

    comparison_anc.delete()


def _pad_operand_a(a: list[Qubit], b: QuantumVariable | list[Qubit]) -> tuple[list[Qubit], QuantumVariable]:
    """Truncate/extend a to match the length of b.

    Register sizes come from .size for a QuantumVariable and jlen for a raw
    qubit list (jlen handles both static lists and traced sizes). Returns the
    resized a_qubits together with the extension ancilla (owned by the caller,
    who must delete it once the addition is complete).
    """
    dim_a = a.size if isinstance(a, QuantumVariable) else jlen(a)
    dim_b = b.size if isinstance(b, QuantumVariable) else jlen(b)

    # reduce the size of a to the size of b if a is larger than b
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a_qubits = a[:effective_size_a]

    # create an extension ancilla to change the size of a when it is smaller than b
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    return a_qubits[:] + extension_anc_a[:], extension_anc_a


def _setup_output_qubit(c_out: QuantumBool | Qubit | None) -> tuple[Qubit, QuantumBool | None]:
    """Resolve the c_out argument into an output_qubit, allocating an ancilla if needed.

    Unlike cuccaro_adder's carry-out (a non-destructive tap on a
    self-restoring wire, free when c_out isn't requested), Thapliyal's
    output_qubit is a genuinely exposed wire. When the caller supplies
    c_out, we write directly into their qubit (same zero-overhead behavior as
    cuccaro_adder). Otherwise we still need a private ancilla for the
    procedure to write into, and pay the extra comparator-based uncompute
    cost to zero it before deleting (see _uncompute_thapliyal_carry).
    """
    if c_out is None:
        output_anc = QuantumBool()
        return output_anc[0], output_anc

    if isinstance(c_out, QuantumBool):
        return c_out[0], None
    if not check_for_tracing_mode() and not isinstance(c_out, Qubit):
        raise TypeError(f"c_out must be of type QuantumBool or Qubit, not {type(c_out)}")
    return c_out, None


@custom_control
def thapliyal_adder(
    a: int | QuantumVariable | list,
    b: QuantumVariable | list,
    c_in: QuantumBool | Qubit | None = None,
    c_out: QuantumBool | Qubit | None = None,
    ctrl: QuantumBool | None = None,
) -> None:
    """In-place adder as introduced in https://arxiv.org/abs/1712.02630

    This function works in both static and dynamic modes. Note that when the first
    input is larger than the second input, the function will perform modulo addition
    (relative to the size of the second input) after the first input is truncated to
    be the same size as the second input.

    .. note::

        If the first input is quantum and the second classical, the function cannot work as addition is
        performed "in-place" on the second input.


    Parameters
    ----------
    a : int, QuantumVariable or list[Qubit]
        The value that should be added.
    b : QuantumVariable or list[Qubit]
        The value that should be modified in the in-place addition.
    c_in : QuantumBool or Qubit, optional
        An optional carry in value. The default is None.
    c_out : QuantumBool or Qubit, optional
        An optional carry out value. The default is None.
    ctrl : QuantumBool, optional
        An optional control value; when given, the addition is only performed if
        ctrl is in the |1> state. The default is None.

    Raises
    ------
    TypeError
        If carry in or carry out is not of type QuantumBool or Qubit in static mode.
    ValueError
        If the inputs are not valid quantum or classical types.

    Returns
    -------
    None
        The function modifies the second input in place.

    Examples
    --------
    Static mode with both quantum inputs:

    >>> from qrisp import QuantumFloat, thapliyal_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> thapliyal_adder(a,b)
    >>> print(b)
    {9: 1.0}

    """
    # convert the classical input to a quantum input. A raw list[Qubit] counts as
    # a quantum register (this is how inpl_add drives the adder), so only genuine
    # classical scalars fall through to the encoder branch below.
    if not isinstance(a, (QuantumVariable, list)):
        # create a QuantumFloat of the same size as the other quantum input
        q_a = b.duplicate()

        with conjugate(int_encoder)(q_a, a):
            thapliyal_adder(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)

        # outside the conjugation, q_a is back in the state |0> and the addition has been performed on b
        # delete the temporary quantum variable created for the classical input
        q_a.delete()
        return

    if not isinstance(b, (QuantumVariable, list)):
        raise ValueError("The second argument must be of type QuantumVariable.")

    # when the inputs are of unequal length, pad the size of the input with the smaller size
    a_qubits, extension_anc_a = _pad_operand_a(a, b)

    output_qubit, output_anc = _setup_output_qubit(c_out)

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        elif not check_for_tracing_mode() and not isinstance(c_in, Qubit):
            raise TypeError(f"c_in must be of type QuantumBool or Qubit, not {type(c_in)}")

    # TODO: naive full-control fallback - controls every gate in the procedure
    # instead of exploiting a MAJ/UMA-style split like cuccaro_adder does. Correct,
    # not yet gate-optimal.
    ctrl_env = nullcontext() if ctrl is None else control(ctrl)
    with ctrl_env:
        if c_in is not None:
            thapliyal_procedure_with_carry(a_qubits, b, c_in, output_qubit)
        else:
            thapliyal_procedure(a_qubits, b, output_qubit)

        if output_anc is not None:
            # output_qubit now holds the carry-out (a > b); since it's not
            # exposed via c_out, zero it back out so it can be deleted (see
            # _uncompute_thapliyal_carry docstring).
            _uncompute_thapliyal_carry(a_qubits, b, output_qubit)

    if output_anc is not None:
        output_anc.delete()

    # delete the extension ancillas when the inputs are of unequal length
    extension_anc_a.delete()
