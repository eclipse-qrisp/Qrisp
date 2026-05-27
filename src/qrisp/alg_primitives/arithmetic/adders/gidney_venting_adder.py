from __future__ import annotations

import jax.numpy as jnp
import qrisp
from qrisp.environments import control, custom_control
from qrisp.jasp import check_for_tracing_mode




def _extract_bit(a_int, digit_index, a_int_is_bigint):
    """Extract one bit from an integer at a specific position. 
    Returns 1 if the bit at that position is set, 0 otherwise.

    Parameters
    ----------
    a_int : int, jnp.ndarray scalar, or BigInteger
        Classical value whose bit is queried.
    digit_index : int
        Zero-based bit index to read (little-endian convention, index 0 = LSB).
    a_int_is_bigint : bool
        If ``True``, read the bit through ``a_int.get_bit(digit_index)``.
        If ``False``, use ``(a_int >> digit_index) & 1``.

    Returns
    -------
    jnp.bool_
        The extracted bit.

    Notes
    -----
    The function exists so that callers don't need to repeat the
    ``if a_int_is_bigint … else …`` pattern at every callsite and to
    guarantee the result is a JAX scalar (``jnp.bool_``) rather than a
    plain Python ``bool`` — JAX cannot trace through Python scalars
    inside compiled loops.
    """
    if a_int_is_bigint:
        return jnp.bool_(a_int.get_bit(digit_index))
    return jnp.bool_((a_int >> digit_index) & 1)





def bit_inverted_mcx(
    parity_check_ctrl: qrisp.Qubit | qrisp.QuantumVariable,
    simple_ctrl: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
    b: bool | jnp.bool_,
    ctrl: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> None:
    """Fig. 1 left: Toffoli with one inverted (Z⊕b) control and one normal control. 
    Done by flipping the parity-check qubit, running a Toffoli gate, then flipping it back.

    - When b is 0: flip the target qubit if both the parity-check qubit and the simple control qubit are in the one
      state. This is a normal MCX gate controlled on 11.
    - When b is 1: flip the target qubit if the parity-check qubit is in the zero state and the simple control qubit is
      in the one state. This is an MCX gate controlled on 01, i.e. one inverted control and one normal control.

    Circuit diagram — two X gates conditioned on classical b wrap a standard MCX::

        parity_check_ctrl: ───[X^b]──•──[X^b]──
                                     │
        simple_ctrl: ────────────────•─────────
                                     │
        target: ─────────────────────⊕─────────

    Used in the carry chain when the classical addend bit decides whether the control fires on zero or one.
    """
    with control(b):
        if ctrl is None:
            qrisp.x(parity_check_ctrl)
        else:
            qrisp.cx(ctrl, parity_check_ctrl)

    # MCX stays 2-control — ctrl is NOT added as a third control because
    # ctrl conditions the parity flips above, not the MCX itself.
    # When ctrl=|0⟩ and b=1, the CX flips are no-ops → MCX sees un-flipped
    # parity (= same as b=0) → d_i=1 effectively becomes d_i=0.
    qrisp.mcx([parity_check_ctrl, simple_ctrl], target)

    with control(b):
        if ctrl is None:
            qrisp.x(parity_check_ctrl)
        else:
            qrisp.cx(ctrl, parity_check_ctrl)



def zz_mcx(
    z0_left: qrisp.Qubit | qrisp.QuantumVariable,
    z1_left: qrisp.Qubit | qrisp.QuantumVariable,
    control: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
) -> None:
    """Fig. 1 middle: Toffoli with one normal control and one ZZ-parity control.

    Flips the target qubit if the two Z-box qubits are different from each other and the control qubit is in the one state.

    The ZZ pair (z0_left, z1_left) acts as a single control that fires when
    the two qubits disagree.  z1_left is temporarily modified by a CX and
    restored, so both Z-box qubits are returned to their original state.

    Circuit diagram — three columns (CX, MCX, CX)::

        z0_left:   ──•──────────────────────•──
                     │                      │
        z1_left:   ──⊕─────•────────────────⊕──
                           │
        control: ──────────•────────────────────
                           │
        target: ───────────⊕────────────────────

    Used in the carry XOR chains to check whether two qubits disagree instead
    of checking if they are in the one state.
    """
    qrisp.cx(z0_left, z1_left)           # collect parity into z1_left
    qrisp.mcx([z1_left, control], target)
    qrisp.cx(z0_left, z1_left)           # restore z1_left


def zz_zz_mcx(
    z_left: qrisp.Qubit | qrisp.QuantumVariable,
    z_left_right: qrisp.Qubit | qrisp.QuantumVariable,
    z_right: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
) -> None:
    """Fig. 1 right: Toffoli controlled on AND of two ZZ-parity checks.

    Flips the target qubit if the two Z qubits on the left are different
    from each other AND the two Z qubits on the right are different from
    each other. The middle qubit (z_left_right) belongs to both pairs.

    Circuit diagram — five columns (CX, CX, MCX, CX, CX)::

        z_left:       ──⊕───────────•───────────⊕──
                        │           │           │
        z_left_right: ──•─────•───────────•─────•──
                              │     │     │
        z_right:      ────────⊕─────•─────⊕────────
                                    │
        target:       ──────────────⊕────────────────

    All three Z qubits are restored after the operation.

    Used in the carry XOR block to check two conditions at once: whether the
    incoming carry and the target bit disagree with their neighbours.
    """
    qrisp.cx(z_left_right, z_left)       # z_left    ← left parity
    qrisp.cx(z_left_right, z_right)      # z_right   ← right parity
    qrisp.mcx([z_left, z_right], target)
    qrisp.cx(z_left_right, z_right)      # restore z_right
    qrisp.cx(z_left_right, z_left)       # restore z_left


def bit_inverted_zz_zz_mcx(
    parity_ctrl1: qrisp.Qubit | qrisp.QuantumVariable,
    parity_ctrl2: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
    b: bool | jnp.bool_,
    ctrl: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> None:
    """MCX with two inverted controls. Flips the parity qubits before and after the
    Toffoli so the control condition depends on the classical bit b.

    - When b is 0: flip the target qubit if both parity controls are in the one state (MCX on 11).
    - When b is 1: flip the target qubit if both parity controls are in the zero state (MCX on 00).

    Circuit diagram — two X gates on each control wrap a standard MCX::

        parity_ctrl1: ──[X^b]──•──[X^b]──
                               │
        parity_ctrl2: ──[X^b]──•──[X^b]──
                               │
        target:      ──────────⊕──────────

    This gate is not explicitly defined as a standalone circuit in the paper
    but appears as a building block of the carry-xor block in Fig. 3.
    
    Used in the carry XOR block for the first slot, where the classical addend decides whether to flip both controls.
    """
    with control(b):
        if ctrl is None:
            qrisp.x(parity_ctrl1)
            qrisp.x(parity_ctrl2)
        else:
            qrisp.cx(ctrl, parity_ctrl1)
            qrisp.cx(ctrl, parity_ctrl2)

    # MCX stays 2-control — same reasoning as bit_inverted_mcx.
    # ctrl conditions the parity flips above, not the MCX itself.
    qrisp.mcx([parity_ctrl1, parity_ctrl2], target)
    
    with control(b):
        if ctrl is None:
            qrisp.x(parity_ctrl1)
            qrisp.x(parity_ctrl2)
        else:
            qrisp.cx(ctrl, parity_ctrl1)
            qrisp.cx(ctrl, parity_ctrl2)

def carry_venting_adder(
    d: int,
    target: list[qrisp.Qubit] | qrisp.QuantumVariable,
    ancilla: qrisp.QuantumVariable,
    c_in: qrisp.Qubit | qrisp.QuantumVariable | None = None,
    carry_xor_target: qrisp.QuantumVariable | None = None,
    a_int_is_bigint: bool = False,
    ctrl: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> int:
    """Fig. 2 — carry-venting CQ in-place adder.

    target ← (target + d + c_in) mod 2^n

    Instead of propagating carries across all bits with many Toffoli gates,
    each carry is measured in the X-basis (vented) as soon as it is no longer
    needed.  Measurement results are packed into an integer bitmask (ventmask)
    and used later for phase correction.  This reduces Toffoli depth from O(n)
    to O(1).

    Two provided clean ancillae alternate roles: one holds the current carry,
    the other receives the next carry.  After each bit the current-carry
    ancilla is vented and reset, becoming available two steps later.

    When *carry_xor_target* is provided, each carry is also copied into the
    corresponding dirty workspace qubit via CX before being vented.  This
    implements the combined first carry-xor and carry vernting pass used
    by dirty_ancillae_adder (Fig. 4) at no extra gate cost.

    Parameters
    ----------
    target : list[qrisp.Qubit] | qrisp.QuantumVariable
        Target register, little-endian (index 0 = LSB). Modified in place.
    d : int
        Classical addend (compile-time constant).
    ancilla : qrisp.QuantumVariable
        2-qubit clean ancilla register for the streaming carry chain.
    c_in : qrisp.Qubit | qrisp.QuantumVariable | None
        Quantum carry-in.  ``None`` is treated as ``|0⟩``.
    carry_xor_target : qrisp.QuantumVariable | None
        Optional dirty workspace register used for the fused carry-xor pass
        (Fig. 4 in the paper).  The carry into bit i is XORed into
        ``carry_xor_target[i-1]``.
    a_int_is_bigint : bool
        If True, read bits from *d* via ``d.get_bit(i)`` (BigInteger).
        If False (default), use shift-and-mask extraction.

    Returns
    -------
    int
        Vented carry measurement bitmask.
    """
    from qrisp.jasp import jrange, jlen, q_fori_loop

    num_qubits = jlen(target)

    clean_anc = ancilla

    if c_in is None:
        carry_in = None
    else:
        try:
            carry_in = c_in[0]
        except (TypeError, AttributeError, IndexError):
            carry_in = c_in

    # Bitmask that records the measurement outcome of each vented carry.
    # The k-th bit of ventmask stores the vent measurement at position k.
    ventmask = jnp.zeros((), dtype=jnp.int64)

    # Initial setup: X^{d_i} on all target bits
    # Flip each target bit if the corresponding classical addend bit is 1.
    # This bakes the classical addend into the target register so the carry
    # chain only needs the register and carry-in to compute the sum.
    for i in jrange(num_qubits):
        d_i = _extract_bit(d, i, a_int_is_bigint)
        with control(d_i):
            if ctrl is None:
                qrisp.x(target[i])
            else:
                qrisp.cx(ctrl, target[i])

    # First building block (bits 0 and 1) that have to be handled before the loop
    # The remaining bits are handled by a loop that applies the same gate
    # pattern to each bit position.
    d0 = _extract_bit(d, 0, a_int_is_bigint)
    d1 = _extract_bit(d, 1, a_int_is_bigint)

    if carry_in is not None:
        bit_inverted_mcx(carry_in, target[0], clean_anc[1], d0, ctrl=ctrl)
        qrisp.cx(carry_in, target[0])
    else:
        bit_inverted_mcx(clean_anc[0], target[0], clean_anc[1], d0, ctrl=ctrl)

    # X on clean_anc[1] when d0=1: corrects the carry for the flipped
    # target[0] bit so the carry chain still computes the right value.
    with control(d0):
        if ctrl is None:
            qrisp.x(clean_anc[1])
        else:
            qrisp.cx(ctrl, clean_anc[1])

    bit_inverted_mcx(clean_anc[1], target[1], clean_anc[0], d1, ctrl=ctrl)
    qrisp.cx(clean_anc[1], target[1])

    # Fused first carry-xor: write carry at clean_anc[1] into carry_xor_target[0].
    # clean_anc[1] holds the carry into bit 1 at this point (computed by the
    # first building block above, before it gets vented).  A single CX copies
    # it to carry_xor_target at no extra Toffoli cost.
    if carry_xor_target is not None:
        qrisp.cx(clean_anc[1], carry_xor_target[0])

    # Vent clean_anc[1] (carry into bit 1) — X-basis measurement
    qrisp.h(clean_anc[1])
    m0 = qrisp.measure(clean_anc[1])
    qrisp.reset(clean_anc[1])
    # Record the vent measurement at bit position 0 of the ventmask
    ventmask = ventmask + (m0.astype(jnp.int64) << 0)

    # Normal repeated blocks (bits 2 .. n-3)
    # Each iteration handles one bit: holds the current carry, computes the
    # next carry, then vents and resets.  The two ancilla qubits alternate so
    # each gets reused every other step.
    #
    # Before computing the next carry, we flip the current carry qubit if the
    # previous addend bit was 1.  This corrects for the effect of the initial
    # X gates on the carry computation.

    def process_middle_bit(j, val):
        target, clean_anc, ventmask = val
        i = j + 2
        d_i = _extract_bit(d, i, a_int_is_bigint)
        d_prev = _extract_bit(d, i - 1, a_int_is_bigint)

        cur = clean_anc[i % 2]
        nxt = clean_anc[(i + 1) % 2]

        with control(d_prev):
            if ctrl is None:
                qrisp.x(cur)
            else:
                qrisp.cx(ctrl, cur)

        bit_inverted_mcx(cur, target[i], nxt, d_i, ctrl=ctrl)
        qrisp.cx(cur, target[i])

        if carry_xor_target is not None:
            qrisp.cx(cur, carry_xor_target[i - 1])

        qrisp.h(cur)
        m_i = qrisp.measure(cur)
        qrisp.reset(cur)
        # Record the vent measurement at bit position (j+1) of the ventmask
        ventmask = ventmask + (m_i.astype(jnp.int64) << (j + 1))

        return target, clean_anc, ventmask

    num_main = jnp.maximum(0, num_qubits - 4)
    target, clean_anc, ventmask = q_fori_loop(0, num_main, process_middle_bit, (target, clean_anc, ventmask))

    # Write the final carry into the most significant bit
    # The carry out of the second-to-last bit becomes the most significant
    # bit of the result.  It is not vented (no follow-up bit needs it).

    # Three cases:
    # 2 qubits: nothing to do (bits 0 and 1 handled by the first block)
    # 3 qubits: the carry is already computed in clean_anc[0]; just CX it into target[2], then vent clean_anc[0].
    # 4+ qubits: use the full gate sequence to compute and write the final carry into the top bit.
    #
    # q_cond is used because ventmask is an integer that is modified inside
    # the branch and read outside.
    def write_final_carry_to_msb(target, clean_anc, ventmask):
        last_i = num_qubits - 2
        d_last = _extract_bit(d, last_i, a_int_is_bigint)
        cur = clean_anc[num_main % 2]              # carry lives in the alternator that didn't just vent

        # Correction: X^{d_{last_i-1}} on cur to account for d's effect on
        # the previous step's carry computation.
        d_correction = _extract_bit(d, jnp.maximum(1, num_qubits - 3), a_int_is_bigint)
        with control(d_correction):
            if ctrl is None:
                qrisp.x(cur)
            else:
                qrisp.cx(ctrl, cur)

        # For n=3, the carry into bit 2 is already in clean_anc[0] (computed by the
        # first block).  Just CX it into target[2] — no MCX needed.
        with control(num_qubits == 3):
            qrisp.cx(cur, target[last_i + 1])

        # Full MCX-based last block: write the carry into target[n-1].
        with control(num_qubits > 3):
            bit_inverted_mcx(cur, target[last_i], target[num_qubits - 1], d_last, ctrl=ctrl)
            qrisp.cx(cur, target[last_i])
            # Fused carry-xor: write the last carry into carry_xor_target[last_i-1].
            # NOTE: For n=3 this write is intentionally NOT done here (it's
            # inside the n>3 branch).  For n=3, the first block already wrote
            # to carry_xor_target[0], and a second write to the same slot would create a
            # double-write that the phase correction cannot undo.
            if carry_xor_target is not None:
                qrisp.cx(cur, carry_xor_target[last_i - 1])

        # Vent (common to both paths) — X-basis measurement
        qrisp.h(cur)
        m_last = qrisp.measure(cur)
        qrisp.reset(cur)
        # Record the vent measurement at bit position (1 + num_main) of the ventmask
        ventmask = ventmask + (m_last.astype(jnp.int64) << (1 + num_main))

        # Final X^{d_last} on target[n-1] (only for n>=4 — for n=3, target[2]
        # was never X^{d} in the initial setup because d_last = d_2).
        with control(num_qubits > 3):
            with control(d_last):
                if ctrl is None:
                    qrisp.x(target[num_qubits - 1])
                else:
                    qrisp.cx(ctrl, target[num_qubits - 1])

        return target, clean_anc, ventmask

    def skip_final_block(target, clean_anc, ventmask):
        return target, clean_anc, ventmask

    from qrisp.jasp.program_control.prefix_control import q_cond
    target, clean_anc, ventmask = q_cond(num_qubits > 2, write_final_carry_to_msb, skip_final_block,
                                   target, clean_anc, ventmask)

    return ventmask


def carry_xor_block(
    d: int,
    dirty_ancillas: list[qrisp.Qubit] | qrisp.QuantumVariable,
    target: list[qrisp.Qubit] | qrisp.QuantumVariable,
    c_in: qrisp.Qubit | qrisp.QuantumVariable | None = None,
    a_int_is_bigint: bool = False,
    ctrl: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> None:
    """Fig. 3 — carry-XOR block: second pass of the two-pass phase correction.

    The target register holds the sum result.  The dirty workspace already
    contains carries from the first pass (fused into carry_venting_adder).
    Recompute the carries from the sum and XOR them into the dirty workspace.
    After this block the dirty workspace should return to all zeros (up to
    Z-phase corrections from surrounding CZ gates).

    The computation has four phases:
      1. Reverse order — walk dirty slots from most significant bit to
         least significant bit, computing the carry into each position
         and XORing it in.
      2. Flip each dirty slot if the corresponding addend bit is 1.
      3. If there is a carry-in, handle the first dirty slot using a
         parity-checking gate.
      4. Forward order — walk dirty slots from least significant bit to
         most significant bit, computing the carry into each position
         and XORing it in.

    Steps 3-4 use parity-checking gates because the dirty workspace stores
    carries differently from the clean ancilla chain.

    Parameters
    ----------
    a_int_is_bigint : bool
        If True, read bits from *d* via ``d.get_bit(i)`` (BigInteger).
        If False (default), use shift-and-mask extraction.
    """
    from qrisp.jasp import jrange, jlen

    num_target_qubits = jlen(target)
    num_dirty_qubits = jlen(dirty_ancillas)

    if c_in is None:
        carry_in = None
    else:
        try:
            carry_in = c_in[0]
        except (TypeError, AttributeError, IndexError):
            carry_in = c_in

    # Reverse chain
    # Walk the dirty slots from most significant bit to least significant bit. 
    # For each slot, combine the
    # corresponding target bit and the previous dirty slot (as carry-in) to
    # recompute the carry, then flip the current dirty slot if the addend
    # bit is 1.  This propagates carries downward from high to low bits.
    for j in jrange(num_dirty_qubits - 1):
        i = (num_dirty_qubits - 1) - j
        bit_i = _extract_bit(d, i, a_int_is_bigint)
        bit_inverted_mcx(target[i], dirty_ancillas[i - 1], dirty_ancillas[i], bit_i, ctrl=ctrl)

    # Flip each dirty slot if the addend bit is 1
    for i in jrange(num_dirty_qubits):
        bit_i = _extract_bit(d, i, a_int_is_bigint)
        with control(bit_i):
            if ctrl is None:
                qrisp.x(dirty_ancillas[i])
            else:
                qrisp.cx(ctrl, dirty_ancillas[i])

    if carry_in is not None:
        d0 = _extract_bit(d, 0, a_int_is_bigint)
        bit_inverted_zz_zz_mcx(carry_in, target[0], dirty_ancillas[0], d0, ctrl=ctrl)

    # Forward chain
    # Walk dirty slots from least significant bit to most significant bit.
    for j in jrange(num_dirty_qubits - 1):
        i = j + 1
        bit_i = _extract_bit(d, i, a_int_is_bigint)
        bit_inverted_zz_zz_mcx(dirty_ancillas[i - 1], target[i], dirty_ancillas[i], bit_i, ctrl=ctrl)





def dirty_ancillae_adder(
    d: int,
    target: list[qrisp.Qubit] | qrisp.QuantumVariable,
    dirty_ancillas: list[qrisp.Qubit] | qrisp.QuantumVariable,
    ancilla: list[qrisp.Qubit] | qrisp.QuantumVariable,
    c_in: qrisp.Qubit | qrisp.QuantumVariable | None = None,
    a_int_is_bigint: bool = False,
    ctrl: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> int:
    """Fig. 4 — dirty-ancilla CQ in-place adder.

    target ← (target + d + c_in) mod 2^n

    Uses 2 clean ancillae for the streaming carry chain plus n-2 dirty
    workspace qubits that are restored to their original state.

    Algorithm (two passes):
      1. Vented addition (carry_venting_adder) with fused carry-xor pass.
         Each carry is also written into the dirty workspace as it is vented.
         After this, target contains the sum and dirty contains the carries.
      2. Phase correction — a sandwich of X, CZ, carry_xor_block, CZ, X
          that fixes the Z-phase from the vented measurements and returns
          the dirty qubits to all zeros.

     Parameters
    ----------
    target : list[qrisp.Qubit] | qrisp.QuantumVariable
        Target register, little-endian (index 0 = LSB). Modified in place.
    d : int
        Classical addend (compile-time constant).
    dirty_ancillas : list[qrisp.Qubit] | qrisp.QuantumVariable
        n-2 dirty workspace qubits.
    ancilla : list[qrisp.Qubit] | qrisp.QuantumVariable
        2-qubit clean ancilla register for the streaming carry chain.
    c_in : qrisp.Qubit | qrisp.QuantumVariable | None
        Quantum carry-in.  ``None`` is treated as ``|0⟩``.
    a_int_is_bigint : bool
        If True, read bits from *d* via ``d.get_bit(i)`` (BigInteger).
        If False (default), use shift-and-mask extraction.

    Returns
    -------
    int
        Vented carry measurement bitmask.  Bit k corresponds to the k-th vent.
    """
    from qrisp.jasp import jrange, jlen

    num_targets = jlen(target)

    dirty_qubits = dirty_ancillas
    num_dirty = jlen(dirty_qubits)

    # Step 1: Vented addition with fused first carry-xor
    ventmask = carry_venting_adder(
        d, target, ancilla=ancilla, c_in=c_in, carry_xor_target=dirty_qubits, a_int_is_bigint=a_int_is_bigint, ctrl=ctrl)


    # Step 2: Phase correction
    for i in jrange(num_targets):
        qrisp.x(target[i])

    # CZ pass 1: correct Z-phases using ventmask
    for k in jrange(num_dirty):
        with control((ventmask >> k) & 1):
            qrisp.z(dirty_qubits[k])

    # Second carry-xor pass: recompute carries against the flipped sum
    carry_xor_block(d, dirty_qubits, target[:-1], c_in, a_int_is_bigint=a_int_is_bigint, ctrl=ctrl)

    # CZ pass 2: correct Z-phases again
    for k in jrange(num_dirty):
        with control((ventmask >> k) & 1):
            qrisp.z(dirty_qubits[k])

    # Restore target
    for i in jrange(num_targets):
        qrisp.x(target[i])

    return ventmask


@custom_control
def gidney_cq_venting_adder(
    d: int,
    target: qrisp.QuantumVariable | list[qrisp.Qubit],
    c_in: qrisp.Qubit | qrisp.QuantumVariable | None = None,
    a_int_is_bigint: bool = False,
    ctrl: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> int:
    """Fig. 5 — borrowing/splitting CQ in-place adder (3 clean ancillae).

    Heavily adapted from the reference implementation released with
    https://arxiv.org/pdf/2507.23079 at https://zenodo.org/records/15866587.

    target ← (target + d + c_in) mod 2^n

    Uses 3 clean ancillae allocated internally (1 carry_mid + 2 streaming
    carry) and no external dirty workspace — the target register is split
    in half, and each half serves as dirty storage for the other half's
    carry chain.

    Algorithm:
      1. Swap carry_mid into the target register at the split point.
      2. Vented addition on the bottom half using the first 2 ancillae.
      3. Swap carry_mid back out — it now holds the carry into the top half.
      4. Dirty-ancilla addition on the top half, using the bottom half as
         dirty workspace (carry_mid as carry-in).
      5. Measure the overflow carry (carry_mid) in the X-basis.
      6. Phase correction for the bottom half using two carry_xor_block
         passes (the fused first pass was consumed by step 4).

    Requires n ≥ 3 (the splitting degenerates for n < 3).

    Parameters
    ----------
    target : qrisp.QuantumVariable | list[qrisp.Qubit]
        Target register, little-endian (index 0 = LSB). Modified in place.
    d : int
        Classical addend (compile-time constant).
    c_in : qrisp.Qubit | qrisp.QuantumVariable | None
        Quantum carry-in. None is treated as |0⟩.
    a_int_is_bigint : bool
        If True, read bits from *d* via ``d.get_bit(i)`` (BigInteger).
        If False (default), use shift-and-mask extraction.

    Returns
    -------
    int
        Vented carry measurement bitmask.  The k-th bit is the X-basis
        measurement outcome of the k-th vented carry.
    """
    from qrisp.jasp import jrange, jlen

    if isinstance(d, (qrisp.QuantumVariable, list)):
        raise TypeError("The first argument must be a classical integer.")
    if not isinstance(target, (qrisp.QuantumVariable, list)):
        raise TypeError("The second argument must be a QuantumVariable or list of Qubits.")

    if not check_for_tracing_mode():
        raise RuntimeError("The Gidney Classical-Quantum adder does not work in standard python execution mode.")

    num_targets = jlen(target)

    # Truncate the classical addend if it has more bits than the target register.
    # Zero-extension for addends narrower than the target is already the default
    # (bits beyond the value read as 0 in _extract_bit).
    if not a_int_is_bigint:
        d = d & ((1 << num_targets) - 1)

    h = (num_targets - 1) >> 1  # bottom half size (needs num_targets >= 3 for a meaningful split)

    # initialize 3 clean ancillae
    # clean_anc[0] = carry_mid   — captures carry from bottom into top half
    # clean_anc[1:] = anc_clean2 — 2 streaming carry ancillae shared by both halves
    clean_anc = qrisp.QuantumVariable(3, name="gidney_anc*")
    carry_mid = clean_anc[0]
    anc_clean2 = clean_anc[1:]

    # Split classical addend: d_lo is the lower h bits, d_hi is the upper bits
    d_lo = d & ((1 << h) - 1)
    d_hi = d >> h

    # Step 1: Place carry_mid into target[h].
    # Three CNOTs implement a SWAP so that carry_venting_adder on the
    # h+1-bit slice target[:h+1] writes the carry out of bit h-1 into
    # carry_mid (as its final carry).  Gidney's code does a Python-level
    # reference swap, but JASP's dynamic-length QuantumVariable requires
    # a quantum gate.
    qrisp.cx(carry_mid, target[h])
    qrisp.cx(target[h], carry_mid)
    qrisp.cx(carry_mid, target[h])

    # Step 2: Vented addition on bottom half
    ventmask_lo = carry_venting_adder(d_lo, target[:h + 1], anc_clean2, c_in=c_in, a_int_is_bigint=a_int_is_bigint, ctrl=ctrl)

    # Step 3: Swap carry_mid back out of target[h].
    # target[h] is restored to its original qubit and carry_mid now
    # holds the carry into the top half.
    qrisp.cx(carry_mid, target[h])
    qrisp.cx(target[h], carry_mid)
    qrisp.cx(carry_mid, target[h])

    # Step 4: Top half addition using dirty ancillas.
    # target[h:] has n-h bits (top half).  carry_mid is the carry-in.
    # target[:max(1, n-h-2)] (bottom n-h-2 bits, at least 1) are borrowed as
    # dirty workspace — they will be restored by dirty_ancillae_adder's internal
    # phase correction.
    # The same anc_clean2 is passed as the clean ancilla register, so the
    # total clean ancilla count is just 3 (shared between both halves).
    # NOTE: n-h-2 may be less than h (e.g. n=5, h=2: n-h-2=1).  We use
    # n-h-2 dirty qubits (at least 1) because dirty_ancillae_adder expects
    # exactly num_targets - 2 dirty qubits.  The remaining bottom bit(s) stay
    # untouched and keep their value from the bottom-half sum.
    dirty_ancillae_adder(
        d_hi, target[h:],
        dirty_ancillas=target[:jnp.maximum(1, num_targets - h - 2)],
        ancilla=anc_clean2,
        c_in=carry_mid,
        a_int_is_bigint=a_int_is_bigint,
        ctrl=ctrl,
    )

    # Step 5: MX measurement on carry_mid.
    # The carry_mid qubit still holds the carry out of the bottom half.
    # We measure it in the X-basis (H then Z-measurement).
    qrisp.h(carry_mid)
    m_carry_mid = qrisp.measure(carry_mid)
    qrisp.reset(carry_mid)

    # Combine ventmask_lo with m_carry_mid to get the full ventmask for the
    # entire register.  m_carry_mid is the vent measurement for the carry
    # between the two halves, which corresponds to bit position h in the ventmask.
    # ventmask_lo contains the vent measurements for the bottom half, which correspond
    # to bit positions 0 through h-1 in the ventmask. 
    # The top half has no vents of its own, so its bits in the ventmask are zero.
    ventmask_lo_mask = (1 << (h - 1)) - 1
    full_ventmask = ((ventmask_lo >> 1) & ventmask_lo_mask) | (
        m_carry_mid.astype(jnp.int64) << (h - 1)
    )

    # Step 6: Phase correction for bottom half vents

    workspace = target[h:2 * h]
    bottom = target[:h]

    # 6a — Complement bottom half: NOT(bottom)
    for i in jrange(h):
        qrisp.x(bottom[i])

    # 6b — CZ(workspace[k], vent[k+1])
    for k in jrange(h):
        with control((full_ventmask >> k) & 1):
            qrisp.z(workspace[k])

    # 6c — First carry_xor pass (explicit, no fused pass for bottom half)
    carry_xor_block(d_lo, workspace, bottom, c_in, a_int_is_bigint=a_int_is_bigint, ctrl=ctrl)

    # 6d — CZ(workspace[k], vent[k+1]) again
    for k in jrange(h):
        with control((full_ventmask >> k) & 1):
            qrisp.z(workspace[k])

    # 6e — Second carry_xor pass
    carry_xor_block(d_lo, workspace, bottom, c_in, a_int_is_bigint=a_int_is_bigint, ctrl=ctrl)

    # 6f — Restore bottom half: NOT cancels the complement from 6a
    for i in jrange(h):
        qrisp.x(bottom[i])

    clean_anc.delete()

    return full_ventmask
