"""********************************************************************************
* Copyright (c) 2026 the Qrisp Authors
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


def _tr_gate(a, b, c):
    # TR gate (arXiv:1712.02630): (A, B, C) -> (A, A^B, A*(~B)^C).
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


def _peres_gate(a, b, c):
    # Peres gate (arXiv:1712.02630): (A, B, C) -> (A, A^B, A*B^C). Step 4 of the
    # adder calls for the Peres gate, which is the inverse of the TR gate (the paper
    # states the two are inverses of each other), so we invert _tr_gate here.
    with invert():
        _tr_gate(a, b, c)


def _uncompute_thapliyal_carry(a, b, carry_qubit):
    """Zeroes carry_qubit, given carry_qubit currently holds (a > b) as an unsigned,
    equal-bit-length comparison (this is what the ripple procedure leaves behind in
    its output_qubit once a is restored and b holds the final sum). a and b are left
    unmodified.

    Unlike cuccaro_adder's carry-out (a non-destructive tap on a self-restoring wire),
    the Thapliyal output_qubit is a genuinely exposed wire with no built-in
    undo step. This recomputes the same boolean value independently via a scratch
    comparison (same trick as uint_qq_less_than in uint_clifford_t_comparisons.py,
    using gidney_adder as the reversible scratch arithmetic since, unlike
    cuccaro_adder, it accepts a plain qubit-list target) and XORs it into
    carry_qubit to cancel it out.

    """
    comparison_anc = QuantumBool()

    def inv_gidney(x, y):
        with invert():
            gidney_adder(x, y)

    with conjugate(inv_gidney, allocation_management=False)(a, b[:] + comparison_anc[:]):
        cx(comparison_anc[0], carry_qubit)

    comparison_anc.delete()


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

    # when the inputs are of unequal length
    # pad the size of the input with the smaller size. Register sizes come from
    # .size for a QuantumVariable and jlen for a raw qubit list (jlen handles both
    # static lists and traced sizes).
    dim_a = a.size if isinstance(a, QuantumVariable) else jlen(a)
    dim_b = b.size if isinstance(b, QuantumVariable) else jlen(b)

    # reduce the size of a to the size of b if a is larger than b
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a_qubits = a[:effective_size_a]

    # create an extension ancilla to change the size of a when it is smaller than b
    extension_size = jnp.maximum(0, dim_b - dim_a)
    extension_anc_a = QuantumVariable(extension_size)
    a_qubits = a_qubits[:] + extension_anc_a[:]

    if c_out is not None:
        if isinstance(c_out, QuantumBool):
            output_qubit = c_out[0]
        elif not check_for_tracing_mode() and not isinstance(c_out, Qubit):
            raise TypeError(f"c_out must be of type QuantumBool or Qubit, not {type(c_out)}")
        else:
            output_qubit = c_out
        output_anc = None
    else:
        # Unlike cuccaro_adder's carry-out (a non-destructive tap on a
        # self-restoring wire, free when c_out isn't requested), Thapliyal's
        # output_qubit is a genuinely exposed wire. When the caller supplies
        # c_out, we write directly into their qubit (same zero-overhead behavior as
        # cuccaro_adder). Otherwise we still need a private ancilla for the
        # procedure to write into, and pay the extra comparator-based uncompute
        # cost to zero it before deleting (see _uncompute_thapliyal_carry).
        output_anc = QuantumBool()
        output_qubit = output_anc[0]

    if c_in is not None:
        if isinstance(c_in, QuantumBool):
            c_in = c_in[0]
        elif not check_for_tracing_mode() and not isinstance(c_in, Qubit):
            raise TypeError(f"c_in must be of type QuantumBool or Qubit, not {type(c_in)}")

        # Thapliyal's algorithm has no native c_in slot (unlike cuccaro_adder's
        # ancilla[0], which plays the role of a fictitious a[-1] in a uniform MAJ
        # chain). Instead, prepend a virtual bit position -1 to both a and b: a
        # constant |1> ancilla for a[-1] and a copy of c_in for b[-1]. Since
        # AND(1, c_in) = c_in, this injects exactly a carry-in of c_in into the
        # real registers (verified numerically against all (a, b, c_in)
        # combinations for small register sizes).
        v_a0 = QuantumVariable(1)
        v_b0 = QuantumVariable(1)
        a_qubits = v_a0[:] + a_qubits[:]
        b = v_b0[:] + b[:]

    # TODO: naive full-control fallback - controls every gate in both steps below
    # instead of exploiting a MAJ/UMA-style split like cuccaro_adder does. Correct,
    # not yet gate-optimal.
    #
    # The v_a0/v_b0 prep and restore gates (when c_in is given) must live inside
    # the same ctrl-conditioned scope as the procedure itself: if ctrl is off, the
    # procedure never touches them, so a restore sequence designed to undo the
    # procedure's effect would leave them non-zero (verified: produces "faulty
    # uncomputation" warnings if pulled out of this scope). The conditional context
    # manager keeps that scope identical whether or not the addition is controlled.
    ctrl_env = nullcontext() if ctrl is None else control(ctrl)
    with ctrl_env:
        if c_in is not None:
            x(v_a0[0])
            cx(c_in, v_b0[0])

        # --- 6-step Thapliyal ripple (arXiv:1712.02630) ---
        # a_qubits and b have equal length; the loop count is derived from a_qubits
        # via jlen (works for both static lists and traced sizes), the same way
        # cuccaro_adder derives its own register size. Descending loops (steps 2 and
        # 4) are rewritten as forward jrange loops with a computed index, mirroring
        # the UMA-reversal pattern in cuccaro_adder.
        n = jlen(a_qubits)

        # Step 1
        for i in jrange(1, n):
            cx(a_qubits[i], b[i])

        # Step 2
        cx(a_qubits[-1], output_qubit)

        for j in jrange(n - 2):
            i = n - 2 - j
            cx(a_qubits[i], a_qubits[i + 1])

        # Step 3
        for i in jrange(n - 1):
            mcx([a_qubits[i], b[i]], a_qubits[i + 1])

        # Step 4 (Peres gate, per the paper)
        _peres_gate(a_qubits[-1], b[-1], output_qubit)

        for j in jrange(n - 1):
            i = n - 2 - j
            _peres_gate(a_qubits[i], b[i], a_qubits[i + 1])

        # Step 5
        for i in jrange(1, n - 1):
            cx(a_qubits[i], a_qubits[i + 1])

        # Step 6
        for i in jrange(1, n):
            cx(a_qubits[i], b[i])

        if output_anc is not None:
            # output_qubit now holds the carry-out (a > b); since it's not
            # exposed via c_out, zero it back out so it can be deleted (see
            # _uncompute_thapliyal_carry docstring).
            _uncompute_thapliyal_carry(a_qubits, b, output_qubit)

        if c_in is not None:
            x(v_a0[0])
            x(v_b0[0])
            cx(c_in, v_b0[0])

    if output_anc is not None:
        output_anc.delete()

    if c_in is not None:
        v_a0.delete()
        v_b0.delete()

    # delete the extension ancillas when the inputs are of unequal length
    extension_anc_a.delete()
