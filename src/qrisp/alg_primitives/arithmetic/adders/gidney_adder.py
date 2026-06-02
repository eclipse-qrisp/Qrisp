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



import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def _validate_gidney_adder_inputs(a, b):
    """Validate that ``(a, b)`` is a supported input pair.

    Returns
    -------
    a_is_quantum : bool
        Whether ``a`` is a quantum register.

    Raises
    ------
    ValueError
        If the pair is not classical-quantum or quantum-quantum.
    """
    # b must be a quantum register (QuantumVariable, DynamicQubitArray,
    # or non-empty list of qubit-like objects).
    b_is_quantum = isinstance(b, (QuantumVariable, DynamicQubitArray)) or (
        isinstance(b, list) and len(b) > 0
        and all(callable(getattr(qb, "qs", None)) for qb in b)
    )
    # a is quantum if it passes the same check or is an empty list
    # (which the quantum path treats as a zero-size register).
    a_is_quantum = isinstance(a, (QuantumVariable, DynamicQubitArray)) or (
        isinstance(a, list) and len(a) > 0
        and all(callable(getattr(qb, "qs", None)) for qb in a)
    ) or (isinstance(a, list) and len(a) == 0)

    # Classical a is valid if it is a concrete int / binary string / numpy
    # integer, or if we are in tracing mode and a is an integer JAX scalar
    # or BigInteger (which carries its own bit-access protocol).
    is_concrete_int = isinstance(a, (int, np.integer, str))
    # Reject non-integer JAX scalars (e.g. float32) — bit extraction
    # on a float would produce silently wrong results.
    is_scalar_like = getattr(a, "ndim", None) == 0 and jnp.issubdtype(
        getattr(a, "dtype", None), jnp.integer
    )
    is_biginteger_like = hasattr(a, "get_bit")
    is_valid_classical = is_concrete_int or (
        check_for_tracing_mode() and (is_scalar_like or is_biginteger_like)
    )
    if not (b_is_quantum and (a_is_quantum or is_valid_classical)):
        raise ValueError(
            "gidney_adder expects inputs to be either classical-quantum "
            "(classical a, quantum b) or quantum-quantum (quantum a, quantum b)."
        )
    return a_is_quantum


def _extract_bit(a_int, digit_index, a_int_is_bigint):
    """Extract one bit from a classical scalar as a JAX boolean.

    Parameters
    ----------
    a_int : int, jnp.ndarray scalar, or BigInteger
        Classical value whose bit is queried.
    digit_index : int
        Zero-based bit index to read (little-endian convention).
    a_int_is_bigint : bool
        If ``True``, read the bit through ``a_int.get_bit``.
        If ``False``, use shift-and-mask extraction.

    Examples
    --------
    >>> bool(_extract_bit(0b1010, 1, False))
    True
    >>> bool(_extract_bit(0b1010, 0, False))
    False
    """
    if a_int_is_bigint:
        return jnp.bool_(a_int.get_bit(digit_index))
    return jnp.bool_((a_int >> digit_index) & 1)


def _apply_x_bit(target, ctrl=None):
    """Apply X to ``target``, optionally controlled by ``ctrl``.

    Parameters
    ----------
    target : Qubit
        The qubit to flip.
    ctrl : Qubit or None
        Optional control qubit. When provided, a CNOT is applied
        instead of a plain X.
    """
    if ctrl is not None:
        cx(ctrl, target)
    else:
        x(target)


def _apply_classical_carry_chain(gidney_anc, b_qbs, n, a_int, a_int_is_bigint, ctrl):
    """Apply carry-chain phases for classical input values.

    Implements the Gidney AND-based carry network (arXiv:1709.06648)
    for the case where *a* is known classically.  Bits of ``a_int``
    are extracted inline and used as classical controls.

    Parameters
    ----------
    gidney_anc : QuantumVariable
        Carry ancilla register with ``n - 1`` qubits.
    b_qbs : list
        Target qubits participating in the carry chain.
    n : int
        Effective carry-chain width, including optional carry wires.
    a_int : int, jnp.ndarray scalar, or BigInteger 
        Classical input representation used for bit extraction.
    a_int_is_bigint : bool
        Indicates whether ``a_int`` must be read via ``get_bit``.
    ctrl : Qubit or None
        Optional outer control qubit.
    """
    # Bit-0 carry setup: if a[0] == 1, seed the carry into the first ancilla qubit:
    #   gidney_anc[0] ^= b[0] (or controlled by ctrl).
    a_bit0 = _extract_bit(a_int, 0, a_int_is_bigint)
    with control(a_bit0):
        if ctrl is not None:
            mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney")
        else:
            cx(b_qbs[0], gidney_anc[0])

    # Forward sweep i = 1 .. n-2.
    # At each position i:
    #   1) Bring the previous carry onto b[i]:  b[i] ^= carry[i-1]
    #   2) If a[i] == 1, toggle the carry input:  carry[i-1] ^= 1
    #   3) Compute the new carry:  carry[i] ^= carry[i-1] & b[i]
    #      (using a Gidney AND gate)
    #   4) Restore a[i] toggle (uncompute step 2)
    #   5) Propagate:  carry[i] ^= carry[i-1]
    for j in jrange(n - 2):
        i = j + 1
        cx(gidney_anc[i - 1], b_qbs[i])
        # _extract_bit is called twice for the same index i — once here and
        # once below for the uncomputation.  This is harmless: in JAX tracing
        # the duplicate is collapsed, and in static mode both calls return the
        # same constant.  The alternative (caching the bit) would add complexity
        # for zero benefit.
        bit_i = _extract_bit(a_int, i, a_int_is_bigint)
        with control(bit_i):
            _apply_x_bit(gidney_anc[i - 1], ctrl)
        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")
        with control(bit_i):
            _apply_x_bit(gidney_anc[i - 1], ctrl)
        cx(gidney_anc[i - 1], gidney_anc[i])

    # Final carry-out: propagate the last carry from position n-2 to b[n-1] (MSB).
    cx(gidney_anc[n - 2], b_qbs[n - 1])

    # Reverse sweep i = n-2 .. 1.
    # Uncompute the carry ancillas in reverse order while writing the
    # sum bits into b.  Each iteration reverses one step of the forward sweep.
    for j in jrange(n - 2):
        i = n - j - 2
        cx(gidney_anc[i - 1], gidney_anc[i])
        # Duplicate _extract_bit call for the same index i (see forward sweep
        # comment above — the double call is intentional and harmless).
        bit_i = _extract_bit(a_int, i, a_int_is_bigint)
        with control(bit_i):
            _apply_x_bit(gidney_anc[i - 1], ctrl)
        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
        with control(bit_i):
            _apply_x_bit(gidney_anc[i - 1], ctrl)

    # Cleanup of bit-0 carry setup.
    with control(a_bit0):
        if ctrl is not None:
            mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney_inv")
        else:
            cx(b_qbs[0], gidney_anc[0])


def _apply_quantum_carry_chain(gidney_anc, a_qbs, b_qbs, n, c_in_qb, c_out_qb, ctrl):
    """Apply carry-chain phases for quantum input registers.

    Implements the Gidney AND-based carry network (arXiv:1709.06648)
    for the case where *a* is a quantum register.  Unlike the classical
    path, the Toffoli-and-uncompute pattern must be fully reversible.

    Parameters
    ----------
    gidney_anc : QuantumVariable
        Carry ancilla register with ``n - 1`` qubits.
    a_qbs : list
        Input-a qubits (including any extension ancillas).
    b_qbs : list
        Target qubits participating in the carry chain.
    n : int
        Effective carry-chain width, including optional carry wires.
    c_in_qb : Qubit or None
        Optional carry-in qubit.
    c_out_qb : Qubit or None
        Optional carry-out qubit.
    ctrl : Qubit or None
        Optional outer control qubit.
    """
    # Bit-0 carry setup (with optional carry-in).
    # The carry-in is folded into the first carry via a conjugation trick:
    #   c_in flips both a[0] and b[0], then a Gidney AND computes
    #   carry[0] = a[0] & b[0], and c_in is XORed onto carry[0].
    # Without carry-in:  carry[0] ^= a[0] & b[0]
    if c_in_qb is not None:
        cx(c_in_qb, b_qbs[0])
        cx(c_in_qb, a_qbs[0])
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")
        cx(c_in_qb, gidney_anc[0])
    else:
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")

    # Forward sweep i = 1 .. n-2.
    # At each position:
    #   1) Bring previous carry onto b[i] and a[i]
    #   2) Compute carry[i] = (a[i] XOR carry[i-1]) & (b[i] XOR carry[i-1])
    #      via a single Gidney AND on the already-encoded inputs.
    #   3) Propagate: carry[i] ^= carry[i-1] (this completes the full
    #      carry recurrence: carry[i] = a[i]&b[i] ^ carry[i-1]&(a[i]^b[i]))
    for j in jrange(n - 2):
        i = j + 1
        # ctrl is not referenced here — the @custom_control decorator's
        # CustomControlEnvironment auto-prepends the control qubit to every
        # instruction at compile time (see custom_control_environment.py:325).
        # The reverse sweep below does reference ctrl explicitly because those
        # mcx gates need it as a second control, not just an outer guard.
        cx(gidney_anc[i - 1], b_qbs[i])
        cx(gidney_anc[i - 1], a_qbs[i])
        mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney")
        cx(gidney_anc[i - 1], gidney_anc[i])

    # MSB carry-out and sum.
    # The carry from position n-2 toggles b[n-1] (the sum's MSB).
    # If no explicit c_out is requested, a[n-1] also contributes
    # to b[n-1] (the full addition b += a).
    if ctrl is not None:
        mcx([ctrl, gidney_anc[n - 2]], b_qbs[n - 1], method="gidney")
        if c_out_qb is None:
            mcx([ctrl, a_qbs[n - 1]], b_qbs[n - 1], method="gidney")
    else:
        cx(gidney_anc[n - 2], b_qbs[n - 1])
        if c_out_qb is None:
            cx(a_qbs[n - 1], b_qbs[n - 1])

    # Reverse sweep i = n-2 .. 1.
    # Walk backward, uncomputing carry ancillas.  At each step the
    # contributed sum bits are written into b (and a is restored).
    for j in jrange(n - 2):
        i = n - j - 2
        cx(gidney_anc[i - 1], gidney_anc[i])
        mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney_inv")
        if ctrl is not None:
            # ctrl is already in the qubit list, so CustomControlEnvironment
            # takes the targeting_control=True path (no double-add).
            mcx([ctrl, a_qbs[i]], b_qbs[i], method="gidney")
            cx(gidney_anc[i - 1], a_qbs[i])
            cx(gidney_anc[i - 1], b_qbs[i])
        else:
            cx(gidney_anc[i - 1], a_qbs[i])
            cx(a_qbs[i], b_qbs[i])

    # Cleanup of bit-0 carry setup.
    if c_in_qb is not None:
        # Uncompute the carry with the carry-in contribution.
        cx(c_in_qb, gidney_anc[0])
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")

        # When an outer control is present, the carry-in handling needs
        # an extra ancilla to keep the c_in / b[0] interaction clean.
        #
        # note: lsb_ctrl_anc is allocated and deleted entirely within this
        # nested block (inside c_in_qb AND ctrl guards).  If this block is
        # ever restructured, make sure the ancilla is still deleted before
        # the enclosing c_in_qb scope ends.
        if ctrl is not None:
            lsb_ctrl_anc = QuantumBool(name="gidney_lsb_ctrl_anc*")
            mcx([ctrl, c_in_qb], lsb_ctrl_anc[0], method="gidney")
            cx(lsb_ctrl_anc[0], b_qbs[0])
            mcx([ctrl, c_in_qb], lsb_ctrl_anc[0], method="gidney_inv")
            cx(c_in_qb, b_qbs[0])
            lsb_ctrl_anc.delete()

        # Restore the a[0] input that was flipped during carry-in setup.
        cx(c_in_qb, a_qbs[0])
    else:
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.  Works in
    both standard static and dynamic modes.

    Parameters
    ----------
    a : int, str, jnp.ndarray, BigInteger, QuantumVariable, DynamicQubitArray, or list
        The input value ``a``.  When ``a`` is a classical value (int, binary string,
        JAX scalar, or ``BigInteger``), the semi-classical path is taken
        and no quantum encoding of ``a`` is created.  When ``a`` is a
        quantum register or qubit list, the quantum-quantum path is used.
        If a list, must contain only qubit-like objects (elements with a callable
        ``qs`` attribute); the list may be empty for quantum-quantum mode.
        Binary strings are little-endian: ``"10"`` means bit 0 = 1, bit 1 = 0
        (decimal value 1).
    b : QuantumVariable, DynamicQubitArray, or list
        The target register that is updated in-place: :math:`b \\leftarrow b + a`.
        If a list, must be a non-empty list of qubit-like objects
        (elements with a callable ``qs`` attribute).
    c_in : QuantumBool, Qubit, or None
        Optional single-qubit carry-in.  When provided, the addition
        becomes :math:`b \\leftarrow b + a + c_{\\text{in}}`.
    c_out : QuantumBool, Qubit, or None
        Optional single-qubit carry-out.  When provided, this qubit
        receives the overflow carry of the addition.
    ctrl : Qubit or None
        Optional control qubit.  When provided, the entire addition is
        executed only if ``ctrl`` is in state :math:`\ket{1}` (controlled addition).

    Raises
    ------
    ValueError
        If inputs do not match one of the supported combinations:
        classical-quantum (classical ``a``, quantum ``b``) or
        quantum-quantum (quantum ``a``, quantum ``b``).

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
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    a_is_quantum = _validate_gidney_adder_inputs(a, b)

    # Normalise QuantumBool wrappers to raw qubits for downstream code.
    c_in_qb = c_in[0] if isinstance(c_in, QuantumBool) else c_in
    c_out_qb = c_out[0] if isinstance(c_out, QuantumBool) else c_out

    # Semi-classical path (classical a, quantum b).
    if not a_is_quantum:
        # Binary strings are little-endian; convert to int.
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        is_concrete_int = isinstance(a, (int, np.integer))

        # Clamp to the target register width (mod 2^len(b)).
        # BigInteger and JAX-scalar paths skip this: bit extraction
        # naturally ignores high bits, so the result is functionally
        # identical.  Skipping the clamp avoids eager evaluation of
        # (1 << jlen(b)) on traced values.
        if is_concrete_int:
            a = int(a) % (1 << jlen(b))

        # In tracing mode promote plain Python ints to JAX arrays so
        # shift/mask operations are traced rather than eagerly evaluated.
        if check_for_tracing_mode() and isinstance(a, int):
            a = jnp.array(a, dtype=jnp.int64)

        # No register-size matching needed here: a is a classical integer,
        # so its effective size is determined by n (the number of b qubits
        # we iterate over via bit extraction).  There is no quantum register
        # to truncate or pad, unlike the fully quantum path below.
        #
        # Build the working register:
        #   - Prepending c_in as an extra LSB handles carry-in without
        #     special-casing the carry chain (a is shifted left by 1 and
        #     the LSB is set to 1 so that the chain "absorbs" c_in).
        #   - Appending c_out as an extra MSB captures overflow.
        b_qbs = b[:]
        if c_in_qb is not None:
            b_qbs = [c_in_qb] + b_qbs
            # Shift a left by 1 and set LSB to 1 so that the carry chain
            # treats c_in as a constant-1 addend at position 0 while the
            # original a bits move up one slot.  This mutated a is used
            # downstream (final XOR loop) — the shift is effectively undone
            # by the adjusted start offset.
            # BigInteger supports both << and + (see jasp_bigintiger.py).
            a = (a << 1) + 1
        if c_out_qb is not None:
            b_qbs = b_qbs + [c_out_qb]

        n = jlen(b_qbs)
        a_int = a
        a_int_is_bigint = isinstance(a_int, BigInteger)

        # Carry chain (only meaningful when n >= 2).
        with control(n > 1):
            gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")
            _apply_classical_carry_chain(
                gidney_anc, b_qbs, n, a_int, a_int_is_bigint, ctrl,
            )
            gidney_anc.delete()

        # Final sum: XOR the remaining bits of a into b.
        # Bits 0 .. start-1 were already accounted for by the carry chain
        # setup (bit 0 is skipped when c_in was folded in).
        start = 1 if c_in_qb is not None else 0
        for i in jrange(start, n):
            bit_i = _extract_bit(a_int, i, a_int_is_bigint)
            with control(bit_i):
                _apply_x_bit(b_qbs[i], ctrl)
        return

    # Quantum path (quantum a, quantum b).
    # Both a and b are quantum registers that must be made the same
    # effective size so the carry chain can operate on matching bit
    # positions.

    # If a is wider than b, truncate it: adding extra high bits to b
    # is impossible since b's register is fixed, and those extra a bits
    # would wrap around modulo 2**len(b) anyway during in-place addition.
    dim_a = jlen(a)
    dim_b = jlen(b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    # If a is narrower than b, pad with |0> ancillas so index arithmetic
    # stays uniform.  These contribute zero to the sum and are deleted
    # at the end (no net effect).
    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*")

    a_qbs = a[:] + a_ext[:]
    b_qbs = b[:]
    if c_out_qb is not None:
        b_qbs = b_qbs + [c_out_qb]

    n = jlen(b_qbs)

    # Carry chain (only meaningful when n >= 2).
    with control(n > 1):
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")
        _apply_quantum_carry_chain(
            gidney_anc, a_qbs, b_qbs, n, c_in_qb, c_out_qb, ctrl
        )
        gidney_anc.delete()

    # Final LSB sum: b[0] ^= a[0].
    # The carry chain handles positions 1 .. n-1; the least-significant
    # bit is the only remaining XOR not yet performed.
    if ctrl is not None:
        mcx([ctrl, a_qbs[0]], b_qbs[0], method="gidney")
    else:
        cx(a_qbs[0], b_qbs[0])

    a_ext.delete()

