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

from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx, merge
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.circuit import fast_append
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def _extract_boolean_digit(integer, digit):
    """Return bit ``digit`` from an integer-like value as a JAX boolean.

    In tracing mode ``integer`` can be a symbolic big-integer representation, so this
    helper keeps the bit extraction logic in one place for both static and Jasp paths.

    Parameters
    ----------
    integer : int, jnp.ndarray, or BigInteger
        The value from which a single bit is extracted.  Can be a concrete
        Python/NumPy int, a JAX scalar tracer, or a symbolic ``BigInteger``
        used during Jasp tracing.
    digit : int or jnp.ndarray
        Zero-based index of the bit to extract (0 = LSB).

    Returns
    -------
    jnp.bool_
        ``True`` when bit ``digit`` of ``integer`` is 1, ``False`` otherwise.
    """
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
        BigInteger,
    )

    # during JAX tracing, integers are represented as BigInteger objects that
    # store their bits symbolically – use the dedicated accessor in that case
    if isinstance(integer, BigInteger):
        return jnp.bool_(integer.get_bit(digit))
    # for concrete ints and JAX scalars, shift the target bit into the LSB
    # position and mask it, then cast to a JAX boolean so both paths return
    # the same dtype
    return jnp.bool_(integer >> digit & 1)


def _apply_gidney_adder_gates(
    n,
    tracing,
    qs,
    b_qbs,
    ctrl_qb=None,
    a_int=None,
    a_qbs=None,
    c_in_qb=None,
    c_out_qb=None,
):
    """Run the Gidney carry ladder for semi-classical and quantum-quantum modes.

    Parameters
    ----------
    n : int or jnp.ndarray
        Width (number of qubits) of the operands being added.
    tracing : bool
        ``True`` when running inside a JAX/Jasp tracing context.
    qs : QuantumSession or None
        The quantum session that owns the qubits.  ``None`` during tracing.
    b_qbs : list of Qubit
        Target qubit list that is modified in-place.
    ctrl_qb : Qubit or None
        Optional external control qubit.
    a_int : int, jnp.ndarray, BigInteger, or None
        Classical addend for the semi-classical mode.
    a_qbs : list of Qubit or None
        Quantum addend qubit list for the quantum-quantum mode.
    c_in_qb : Qubit or None
        Optional carry-in qubit for the quantum-quantum mode.
    c_out_qb : Qubit or None
        Optional carry-out qubit for the quantum-quantum mode.
    """
    # Carry-ladder structure:
    # forward  -> compute carries from LSB towards MSB
    # midpoint -> apply MSB update using the last carry
    # backward -> uncompute carries and restore ancillas
    # jrange works in both tracing and static modes, so loop code stays shared.
    is_quantum_mode = a_qbs is not None
    is_controlled = ctrl_qb is not None

    # when the operand is only one bit wide there are no carries to propagate,
    # so the entire ladder is skipped – the control() environment guards this
    with control(n > 1):
        # allocate n-1 ancilla qubits, one per carry position between adjacent
        # bit pairs (bit i and bit i+1 share ancilla i)
        gidney_anc = (
            QuantumVariable(n - 1, name="gidney_anc*")
            if tracing
            else QuantumVariable(n - 1, name="gidney_anc*", qs=qs)
        )

        # init: seed the carry chain with carry_0 = a[0] AND b[0]
        if not is_quantum_mode:

            def first_forward_carry():
                if ctrl_qb is None:
                    cx(b_qbs[0], gidney_anc[0])
                else:
                    mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney")

            _apply_if_classical_bit_set(a_int, 0, tracing, first_forward_carry)
        elif c_in_qb is not None:
            # Carry-in trick: inject c_in directly at i=0 without extending a.
            cx(c_in_qb, b_qbs[0])
            cx(c_in_qb, a_qbs[0])
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")
            cx(c_in_qb, gidney_anc[0])
        else:
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")

        # forward pass – for each subsequent bit position i (1 .. n-2), use the
        # incoming carry and the current a/b bits to produce the next carry
        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], b_qbs[i])

            if not is_quantum_mode:

                def toggle_forward_anc():
                    if ctrl_qb is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl_qb, gidney_anc[i - 1])

                _apply_if_classical_bit_set(a_int, i, tracing, toggle_forward_anc)
                mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")
                _apply_if_classical_bit_set(a_int, i, tracing, toggle_forward_anc)
            else:
                cx(gidney_anc[i - 1], a_qbs[i])
                mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney")

            cx(gidney_anc[i - 1], gidney_anc[i])

        # midpoint – the last carry (ancilla[n-2]) together with the MSBs
        # determines the MSB of the sum; after this the backward pass begins
        if not is_quantum_mode:
            cx(gidney_anc[n - 2], b_qbs[n - 1])
        else:
            if is_controlled:
                mcx([ctrl_qb, gidney_anc[n - 2]], b_qbs[n - 1])
                if c_out_qb is None:
                    mcx([ctrl_qb, a_qbs[n - 1]], b_qbs[n - 1])
            else:
                cx(gidney_anc[n - 2], b_qbs[n - 1])
                if c_out_qb is None:
                    cx(a_qbs[n - 1], b_qbs[n - 1])

        # backward pass – walk back from the MSB to bit 1, reversing the
        # carry computation so each ancilla is restored to |0> while
        # simultaneously XOR-ing the sum bits into the b register
        for j in jrange(n - 2):
            i = n - j - 2
            cx(gidney_anc[i - 1], gidney_anc[i])

            if not is_quantum_mode:
                if is_controlled:

                    def toggle_backward_ctrl():
                        cx(ctrl_qb, gidney_anc[i - 1])

                    _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_ctrl)
                    mcx(
                        [gidney_anc[i - 1], b_qbs[i]],
                        gidney_anc[i],
                        method="gidney_inv",
                    )
                    _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_ctrl)
                else:

                    def toggle_backward_anc():
                        x(gidney_anc[i - 1])

                    _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_anc)
                    mcx(
                        [gidney_anc[i - 1], b_qbs[i]],
                        gidney_anc[i],
                        method="gidney_inv",
                    )
                    _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_anc)
            else:
                mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney_inv")
                if is_controlled:
                    mcx([ctrl_qb, a_qbs[i]], b_qbs[i])
                    cx(gidney_anc[i - 1], a_qbs[i])
                    cx(gidney_anc[i - 1], b_qbs[i])
                else:
                    cx(gidney_anc[i - 1], a_qbs[i])
                    cx(a_qbs[i], b_qbs[i])

        # final: erase carry_0 (mirror of init)
        if not is_quantum_mode:

            def first_backward_carry():
                if ctrl_qb is None:
                    cx(b_qbs[0], gidney_anc[0])
                else:
                    mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney_inv")

            _apply_if_classical_bit_set(a_int, 0, tracing, first_backward_carry)
        elif c_in_qb is not None:
            cx(c_in_qb, gidney_anc[0])
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")

            # In controlled mode, convert the unconditional c_in toggle on b[0]
            # into a controlled one while keeping the same gate family.
            if is_controlled:
                lsb_ctrl_anc = (
                    QuantumBool(name="gidney_lsb_ctrl_anc*")
                    if tracing
                    else QuantumBool(name="gidney_lsb_ctrl_anc*", qs=qs)
                )
                mcx([ctrl_qb, c_in_qb], lsb_ctrl_anc[0], method="gidney")
                cx(lsb_ctrl_anc[0], b_qbs[0])
                mcx([ctrl_qb, c_in_qb], lsb_ctrl_anc[0], method="gidney_inv")
                cx(c_in_qb, b_qbs[0])
                lsb_ctrl_anc.delete()

            cx(c_in_qb, a_qbs[0])
        else:
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")

        # all ancillas are now back in |0> and can be safely deallocated
        gidney_anc.delete(verify=False)


def _apply_if_classical_bit_set(a_int, bit_index, tracing, op):
    """Execute ``op`` if classical bit ``bit_index`` of ``a_int`` is one.

    In tracing mode the condition must be expressed with a quantum control
    context; in static mode we can use a direct Python bit test.

    Parameters
    ----------
    a_int : int, jnp.ndarray, or BigInteger
        The classical integer whose bits are tested.
    bit_index : int or jnp.ndarray
        Zero-based index of the bit to test (0 = LSB).
    tracing : bool
        ``True`` when running inside a JAX/Jasp tracing context.
    op : callable
        Zero-argument callable that emits quantum gates.  It is invoked
        exactly once if the tested bit is 1, and not at all otherwise.
    """
    if tracing:
        # during JAX tracing the value of a_int is not yet known at Python
        # level, so the branch must be expressed as a quantum-controlled
        # environment that will be resolved when the traced program runs
        with control(_extract_boolean_digit(a_int, bit_index)):
            op()
    elif (int(a_int) >> bit_index) & 1:
        # at normal (non-tracing) time the integer is concrete, so a plain
        # Python if-check on the relevant bit suffices
        op()


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.  Works in
    both standard static and dynamic modes.

    Parameters
    ----------
    a : int, str, jnp.ndarray, BigInteger, QuantumVariable, DynamicQubitArray, or list
        The addend.  When ``a`` is a classical value (int, binary string,
        JAX scalar, or ``BigInteger``), the semi-classical path is taken
        and no quantum encoding of ``a`` is created.  When ``a`` is a
        quantum register or qubit list, the quantum-quantum path is used.
    b : QuantumVariable or DynamicQubitArray
        The target register that is updated in-place: ``b ← b + a``.
    c_in : QuantumBool, Qubit, or None
        Optional single-qubit carry-in.  When provided, the addition
        becomes ``b ← b + a + c_in``.
    c_out : QuantumBool, Qubit, or None
        Optional single-qubit carry-out.  When provided, this qubit
        receives the overflow carry of the addition.
    ctrl : Qubit or None
        Optional control qubit.  When provided, the entire addition is
        executed only if ``ctrl`` is in state |1⟩ (controlled addition).

    Examples
    --------

    >>> from qrisp import QuantumFloat, gidney_adder
    >>> a = QuantumFloat(4)
    >>> b = QuantumFloat(4)
    >>> a[:] = 4
    >>> b[:] = 5
    >>> gidney_adder(a, b)
    >>> print(b)
    {9: 1.0}
    """
    # ---- Normalize Optional Qubit Inputs ----
    c_in_qb = c_in[0] if isinstance(c_in, QuantumBool) else c_in
    c_out_qb = c_out[0] if isinstance(c_out, QuantumBool) else c_out
    is_tracing = check_for_tracing_mode()

    # ---- Semi-Classical Path (classical a, quantum b) ----
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        # accept binary strings like "1101" (LSB-first) and convert to int
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        # determine execution context: JAX tracing requires symbolic
        # operations; static mode can use concrete Python values directly
        is_concrete_int = isinstance(a, (int, np.integer))
        # a JAX tracer that is 0-dimensional (scalar) behaves like an int
        # but cannot be inspected at trace time
        is_scalar_like = getattr(a, "ndim", None) == 0
        # BigInteger objects expose individual bits via .get_bit()
        is_biginteger_like = hasattr(a, "get_bit")

        # any of the above qualifies as a semi-classical source – the value
        # provides classical bit information but is not a qubit array
        if is_concrete_int or (is_tracing and (is_scalar_like or is_biginteger_like)):
            # truncate classical a to the width of b so higher bits don't produce spurious
            # gates (addition is mod 2^n)
            if is_concrete_int:
                a = int(a) % (1 << jlen(b))

            # in JAX tracing mode, wrap the integer in a JAX array so
            # subsequent bit-shift operations stay within the JAX graph
            if is_tracing:
                if isinstance(a, int):
                    a = jnp.array(a, dtype=jnp.int64)
            else:
                # in static mode every qubit involved must belong to the same
                # quantum session – merge them before emitting any gates
                # have to worry about qs due to the QCLA adder
                qs_list = [b[0].qs()]
                for qb in [c_in, c_out, ctrl]:
                    if qb is not None:
                        if isinstance(qb, QuantumBool):
                            qb = qb[0]
                        qs_list.append(qb.qs())
                merge(qs_list)

            # build a flat target list (works in tracing and static modes)
            b_qbs = b[:]

            # Encode carry-in by prepending c_in to b and setting the extra
            # classical LSB of a to 1.
            if c_in_qb is not None:
                b_qbs = [c_in_qb] + b_qbs
                a = (a << 1) + 1

            # carry-out is modelled by appending a fresh qubit to b; the
            # ladder will naturally propagate the overflow carry into it
            if c_out_qb is not None:
                b_qbs = b_qbs + [c_out_qb]

            # n is the total width of the (possibly extended) target register
            n = jlen(b_qbs)

            # when the target register is a single qubit, the carry ladder
            # degenerates (no carries to propagate) – just conditionally flip that qubit
            if (not is_tracing) and n == 1:

                def single_bit_update():
                    if ctrl is None:
                        x(b_qbs[0])
                    else:
                        cx(ctrl, b_qbs[0])

                _apply_if_classical_bit_set(
                    a, 0, is_tracing, single_bit_update
                )
                return

            # fast_append batches gate emissions for better performance in
            # static mode; in tracing mode no batching is needed
            ctx = fast_append(3) if not is_tracing else nullcontext()
            with ctx:
                # run the three-phase carry ladder (forward, mid, backward)
                _apply_gidney_adder_gates(
                    n=n,
                    tracing=is_tracing,
                    qs=None if is_tracing else b_qbs[0].qs(),
                    b_qbs=b_qbs,
                    ctrl_qb=ctrl,
                    a_int=a,
                )

                # after the carry chain has been fully uncomputed, each bit of
                # a still needs to be XOR-ed onto the corresponding b qubit to
                # complete the addition (the ladder only handled the carries)
                for i in jrange(n):
                    # the carry-in qubit was prepended at position 0 and is
                    # already accounted for by the ladder – skip it
                    if c_in is not None and i == 0:
                        continue

                    def final_sum_update():
                        if ctrl is not None:
                            cx(ctrl, b_qbs[i])
                        else:
                            x(b_qbs[i])

                    # only emit the XOR gate when the corresponding
                    # classical bit of a is 1
                    _apply_if_classical_bit_set(
                        a, i, is_tracing, final_sum_update
                    )

            return

    # ---- Quantum-Quantum Path ----
    # The ladder requires matching widths, so truncate/pad a to b.
    qs = None if is_tracing else b[0].qs()

    # if a is wider than b, extra high bits cannot affect the (mod 2^n)
    # result – drop them.  if a is narrower, pad with |0> ancillas below
    dim_a = jlen(a)
    dim_b = jlen(b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    # extend a with |0> ancillas up to b width
    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = (
        QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs)
        if qs is not None
        else QuantumVariable(extension_size, name="gidney_a_ext*")
    )
    # build flat qubit lists so indexing works uniformly for both paths
    a = a[:] + a_ext[:]
    b = b[:]

    # include optional carry-out in the target carry chain
    if c_out_qb is not None:
        b = b + [c_out_qb]

    # ladder width follows the active target chain
    n = jlen(b)

    # run the three-phase carry ladder for the quantum-quantum case
    _apply_gidney_adder_gates(
        n=n,
        tracing=is_tracing,
        qs=qs,
        b_qbs=b,
        ctrl_qb=ctrl,
        a_qbs=a,
        c_in_qb=c_in_qb,
        c_out_qb=c_out_qb,
    )

    # Ladder handles carries and upper-bit sums; apply the LSB a[0] contribution.
    if ctrl is not None:
        mcx([ctrl, a[0]], b[0])
    else:
        cx(a[0], b[0])

    # release the zero-padding ancillas that extended a to match b's width
    a_ext.delete(verify=False)
