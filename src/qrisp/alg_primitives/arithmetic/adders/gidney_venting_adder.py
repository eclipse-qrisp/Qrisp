from __future__ import annotations

import jax.numpy as jnp
from typing import TYPE_CHECKING
from qrisp import QuantumVariable, x, cx, mcx, h, measure, reset, z
from qrisp.environments import control, custom_control
from qrisp.jasp import check_for_tracing_mode
from qrisp.qtypes import QuantumBool

from qrisp.alg_primitives.arithmetic.adders.gidney_adder import _extract_bit

if TYPE_CHECKING:
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
        BigInteger,
    )


def bit_inverted_mcx(
    parity_check_ctrl: QuantumVariable,
    simple_ctrl: QuantumVariable,
    target: QuantumVariable,
    b: bool,
    ctrl: QuantumVariable | None = None,
) -> None:
    """Fig. 1 left: Toffoli with one inverted (Z⊕b) control and one normal control.
    Done by flipping the parity-check qubit, running a Toffoli gate, then flipping it back.

    - When b is 0: flip the target qubit if both the parity-check qubit and the simple control qubit are in the one
      state. This is a normal MCX gate controlled on 11.
    - When b is 1: flip the target qubit if the parity-check qubit is in the zero state and the simple control qubit is
      in the one state. This is an MCX gate controlled on 01, i.e. one inverted control and one normal control.

    .. warning::

       When *ctrl* is provided, it only gates the X-flips on *parity_check_ctrl*, **not** the MCX itself.
       If *ctrl* = |0⟩, the X-flips are suppressed (same as *b* = 0), but the MCX still executes and can
       flip *target* whenever the resulting input state satisfies its control condition.  This function
       does **not** guarantee a no-op for *ctrl* = |0⟩ — callers must ensure that when *ctrl* = |0⟩,
       the combination of *parity_check_ctrl* and *simple_ctrl* never causes the MCX to fire.

    Circuit diagram — two X gates conditioned on classical b wrap a standard MCX:

    .. code-block:: text

        parity_check_ctrl: ───[X^b]──•──[X^b]──
                                     │
        simple_ctrl: ────────────────•─────────
                                     │
        target: ─────────────────────X─────────

    Used in the carry chain when the classical addend bit decides whether the control fires on zero or one.
    """
    with control(b):
        if ctrl is None:
            x(parity_check_ctrl)
        else:
            cx(ctrl, parity_check_ctrl)

    # MCX stays 2-control — ctrl is NOT added as a third control because
    # ctrl conditions the parity flips above, not the MCX itself.
    # When ctrl=|0⟩ and b=1, the CX flips are no-ops → MCX sees un-flipped
    # parity (= same as b=0) → d_i=1 effectively becomes d_i=0.
    mcx([parity_check_ctrl, simple_ctrl], target)

    with control(b):
        if ctrl is None:
            x(parity_check_ctrl)
        else:
            cx(ctrl, parity_check_ctrl)


def zz_mcx(
    z0_left: QuantumVariable,
    z1_left: QuantumVariable,
    control: QuantumVariable,
    target: QuantumVariable,
) -> None:
    """Fig. 1 middle: Toffoli with one normal control and one ZZ-parity control.

    Flips the target qubit if the two Z-box qubits are different from each other and the control qubit is in the one state.

    The ZZ pair (z0_left, z1_left) acts as a single control that fires when
    the two qubits disagree.  z1_left is temporarily modified by a CX and
    restored, so both Z-box qubits are returned to their original state.

    Circuit diagram — three columns (CX, MCX, CX):

    .. code-block:: text

        z0_left:   ──•──────────────────────•──
                     │                      │
        z1_left:   ──X─────•────────────────X──
                           │
        control: ──────────•────────────────────
                           │
        target: ───────────X────────────────────

    Used in the carry XOR chains to check whether two qubits disagree instead
    of checking if they are in the one state.
    """
    cx(z0_left, z1_left)  # collect parity into z1_left
    mcx([z1_left, control], target)
    cx(z0_left, z1_left)  # restore z1_left


def zz_zz_mcx(
    z_left: QuantumVariable,
    z_left_right: QuantumVariable,
    z_right: QuantumVariable,
    target: QuantumVariable,
) -> None:
    """Fig. 1 right: Toffoli controlled on AND of two ZZ-parity checks.

    Flips the target qubit if the two Z qubits on the left are different
    from each other AND the two Z qubits on the right are different from
    each other. The middle qubit (z_left_right) belongs to both pairs.

    Circuit diagram — five columns (CX, CX, MCX, CX, CX):

    .. code-block:: text

        z_left:       ──X───────────•───────────X──
                        │           │           │
        z_left_right: ──•─────•───────────•─────•──
                              │     │     │
        z_right:      ────────X─────•─────X────────
                                    │
        target:       ──────────────X────────────────

    All three Z qubits are restored after the operation.

    Used in the carry XOR block to check two conditions at once: whether the
    incoming carry and the target bit disagree with their neighbours.
    """
    cx(z_left_right, z_left)  # z_left    ← left parity
    cx(z_left_right, z_right)  # z_right   ← right parity
    mcx([z_left, z_right], target)
    cx(z_left_right, z_right)  # restore z_right
    cx(z_left_right, z_left)  # restore z_left


def bit_inverted_zz_zz_mcx(
    parity_ctrl1: QuantumVariable,
    parity_ctrl2: QuantumVariable,
    target: QuantumVariable,
    b: bool,
    ctrl: QuantumVariable | None = None,
) -> None:
    """MCX with two inverted controls. Flips the parity qubits before and after the
    Toffoli so the control condition depends on the classical bit b.

    - When b is 0: flip the target qubit if both parity controls are in the one state (MCX on 11).
    - When b is 1: flip the target qubit if both parity controls are in the zero state (MCX on 00).

    .. warning::

       When *ctrl* is provided, it only gates the X-flips on *parity_ctrl1* and *parity_ctrl2*,
       **not** the MCX itself.  If *ctrl* = |0⟩, the X-flips are suppressed (same as *b* = 0),
       but the MCX still executes and can flip *target* whenever the resulting input state
       satisfies its control condition.  This function does **not** guarantee a no-op for
       *ctrl* = |0⟩ — callers must ensure that when *ctrl* = |0⟩, the combination of
       *parity_ctrl1* and *parity_ctrl2* never causes the MCX to fire.

    Circuit diagram — two X gates on each control wrap a standard MCX:

    .. code-block:: text

        parity_ctrl1: ──[X^b]──•──[X^b]──
                               │
        parity_ctrl2: ──[X^b]──•──[X^b]──
                               │
        target:      ──────────X──────────

    This gate is not explicitly defined as a standalone circuit in the paper
    but appears as a building block of the carry-xor block in Fig. 3.

    Used in the carry XOR block for the first slot, where the classical addend decides whether to flip both controls.
    """
    with control(b):
        if ctrl is None:
            x(parity_ctrl1)
            x(parity_ctrl2)
        else:
            cx(ctrl, parity_ctrl1)
            cx(ctrl, parity_ctrl2)

    # MCX stays 2-control — same reasoning as bit_inverted_mcx.
    # ctrl conditions the parity flips above, not the MCX itself.
    mcx([parity_ctrl1, parity_ctrl2], target)

    with control(b):
        if ctrl is None:
            x(parity_ctrl1)
            x(parity_ctrl2)
        else:
            cx(ctrl, parity_ctrl1)
            cx(ctrl, parity_ctrl2)


def carry_venting_adder(
    d: int | BigInteger,
    target: QuantumVariable,
    ancilla: QuantumVariable,
    c_in: QuantumVariable | None = None,
    carry_xor_target: QuantumVariable | None = None,
    ctrl: QuantumVariable | None = None,
) -> tuple[int, int]:
    r"""Fig. 2 — carry-venting CQ in-place adder.

    :math:`\text{target} \rightarrow (\text{target} + d + c_{\text{in}}) \bmod 2^n`

    Instead of propagating carries across all bits with many Toffoli gates,
    each carry is measured in the X-basis (vented) as soon as it is no longer
    needed.  Measurement results are packed into an integer bitmask (ventmask)
    and used later for phase correction.  This reduces Toffoli depth from O(n)
    to O(1).

    Two provided clean ancillae alternate roles: one holds the current carry,
    the other receives the next carry.  After each bit the current carry
    ancilla is vented and reset, becoming available two steps later.

    When *carry_xor_target* is provided, each carry is also copied into the
    corresponding dirty workspace qubit via CX before being vented.  This
    implements the combined first carry-xor and carry vernting pass used
    by dirty_ancillae_adder (Fig. 4) at no extra gate cost.

    Parameters
    ----------
    target : QuantumVariable
        Target register, little-endian (index 0 = LSB). Modified in place.
    d : int
        Classical addend (compile-time constant).
    ancilla : QuantumVariable
        2-qubit clean ancilla register for the streaming carry chain.
    c_in : QuantumBool | None
        Quantum carry-in.  ``None`` is treated as ``|0⟩``.
    carry_xor_target : QuantumVariable | None
        Optional dirty workspace register used for the fused carry-xor pass
        (Fig. 4 in the paper).  The carry into bit i is XORed into
        ``carry_xor_target[i-1]``.

    Returns
    -------
    ventmask : int
        Vented carry measurement bitmask.  Bit k corresponds to the k-th vent.
    num_vents : int
        Number of vents written to the bitmask (helps callers reconstruct
        the vent layout without knowing internal bit-indexing details).
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

    # Counter for carry_xor_target writes — advances on each write so slots
    # are always filled consecutively, eliminating index-collision bugs
    # (e.g. n=3 where the first-block write and final-block write would
    # both target index 0 without this counter).
    carry_xor_cnt = 0

    # Initial setup: X^{d_i} on all target bits
    # Flip each target bit if the corresponding classical addend bit is 1.
    # This bakes the classical addend into the target register so the carry
    # chain only needs the register and carry-in to compute the sum.
    for i in jrange(num_qubits):
        d_i = _extract_bit(d, i)
        with control(d_i):
            if ctrl is None:
                x(target[i])
            else:
                cx(ctrl, target[i])

    # First building block (bits 0 and 1) that have to be handled before the loop
    # The remaining bits are handled by a loop that applies the same gate
    # pattern to each bit position.
    d0 = _extract_bit(d, 0)
    d1 = _extract_bit(d, 1)

    if carry_in is not None:
        bit_inverted_mcx(carry_in, target[0], clean_anc[1], d0, ctrl=ctrl)
        cx(carry_in, target[0])
    else:
        bit_inverted_mcx(clean_anc[0], target[0], clean_anc[1], d0, ctrl=ctrl)

    # X on clean_anc[1] when d0=1: corrects the carry for the flipped
    # target[0] bit so the carry chain still computes the right value.
    with control(d0):
        if ctrl is None:
            x(clean_anc[1])
        else:
            cx(ctrl, clean_anc[1])

    bit_inverted_mcx(clean_anc[1], target[1], clean_anc[0], d1, ctrl=ctrl)
    cx(clean_anc[1], target[1])

    # Fused first carry-xor: write carry at clean_anc[1] into carry_xor_target.
    # clean_anc[1] holds the carry into bit 1 at this point (computed by the
    # first building block above, before it gets vented).  A single CX copies
    # it to carry_xor_target at no extra Toffoli cost.
    if carry_xor_target is not None:
        cx(clean_anc[1], carry_xor_target[carry_xor_cnt])
        carry_xor_cnt = carry_xor_cnt + 1

    # Vent clean_anc[1] (carry into bit 1) — X-basis measurement
    h(clean_anc[1])
    m0 = measure(clean_anc[1])
    reset(clean_anc[1])
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
        target, clean_anc, ventmask, carry_xor_cnt = val
        i = j + 2
        d_i = _extract_bit(d, i)
        d_prev = _extract_bit(d, i - 1)

        current_carry = clean_anc[i % 2]
        next_carry = clean_anc[(i + 1) % 2]

        with control(d_prev):
            if ctrl is None:
                x(current_carry)
            else:
                cx(ctrl, current_carry)

        bit_inverted_mcx(current_carry, target[i], next_carry, d_i, ctrl=ctrl)
        cx(current_carry, target[i])

        if carry_xor_target is not None:
            cx(current_carry, carry_xor_target[carry_xor_cnt])
            carry_xor_cnt = carry_xor_cnt + 1

        h(current_carry)
        m_i = measure(current_carry)
        reset(current_carry)
        # Record the vent measurement at bit position (j+1) of the ventmask
        ventmask = ventmask + (m_i.astype(jnp.int64) << (j + 1))

        return target, clean_anc, ventmask, carry_xor_cnt

    num_main = jnp.maximum(0, num_qubits - 4)
    target, clean_anc, ventmask, carry_xor_cnt = q_fori_loop(
        0, num_main, process_middle_bit, (target, clean_anc, ventmask, carry_xor_cnt)
    )

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
    def write_final_carry_to_msb(target, clean_anc, ventmask, carry_xor_cnt):
        last_i = num_qubits - 2
        d_last = _extract_bit(d, last_i)
        current_carry = clean_anc[
            num_main % 2
        ]  # carry lives in the alternator that didn't just vent

        # Correction: X^{d_{last_i-1}} on current_carry to account for d's effect on
        # the previous step's carry computation.
        d_correction = _extract_bit(d, jnp.maximum(1, num_qubits - 3))
        with control(d_correction):
            if ctrl is None:
                x(current_carry)
            else:
                cx(ctrl, current_carry)

        # For n=3, the carry into bit 2 is already in clean_anc[0] (computed by the
        # first block).  Just CX it into target[2] — no MCX needed.
        with control(num_qubits == 3):
            cx(current_carry, target[last_i + 1])

        # Full MCX-based last block: write the carry into target[n-1].
        with control(num_qubits > 3):
            bit_inverted_mcx(
                current_carry, target[last_i], target[num_qubits - 1], d_last, ctrl=ctrl
            )
            cx(current_carry, target[last_i])

        # Fused carry-xor: write the last carry into the next unwritten slot.
        # Guared by counter vs. dirty count so it naturally skips when all
        # slots are already filled (e.g. n=3 where the first block already
        # wrote the only slot).  No need for a num_qubits > 3 guard.
        if carry_xor_target is not None:
            still_needed = carry_xor_cnt < jlen(carry_xor_target)
            with control(still_needed):
                cx(current_carry, carry_xor_target[carry_xor_cnt])
            carry_xor_cnt = carry_xor_cnt + still_needed.astype(jnp.int64)
        else:
            pass

        # Vent (common to both paths) — X-basis measurement
        h(current_carry)
        m_last = measure(current_carry)
        reset(current_carry)
        # Record the vent measurement at bit position (1 + num_main) of the ventmask
        ventmask = ventmask + (m_last.astype(jnp.int64) << (1 + num_main))

        # Final X^{d_last} on target[n-1] (only for n>=4 — for n=3, target[2]
        # was never X^{d} in the initial setup because d_last = d_2).
        with control(num_qubits > 3):
            with control(d_last):
                if ctrl is None:
                    x(target[num_qubits - 1])
                else:
                    cx(ctrl, target[num_qubits - 1])

        return target, clean_anc, ventmask, carry_xor_cnt

    def skip_final_block(target, clean_anc, ventmask, carry_xor_cnt):
        return target, clean_anc, ventmask, carry_xor_cnt

    from qrisp.jasp.program_control.prefix_control import q_cond

    target, clean_anc, ventmask, carry_xor_cnt = q_cond(
        num_qubits > 2,
        write_final_carry_to_msb,
        skip_final_block,
        target,
        clean_anc,
        ventmask,
        carry_xor_cnt,
    )

    # Count the number of vents for the caller (used in ventmask recombination).
    # When num_qubits <= 2 the final block is skipped (q_cond with num_qubits > 2),
    # so only m0 from the first block is vented → 1 vent.
    # When num_qubits >= 3 all three sources fire: m0 + middle + m_last.
    num_vents = 1 + jnp.where(num_qubits > 2, 1 + jnp.maximum(0, num_qubits - 4), 0)

    return ventmask, num_vents


def carry_xor_block(
    d: int | BigInteger,
    dirty_ancillas: QuantumVariable,
    target: QuantumVariable,
    c_in: QuantumVariable | None = None,
    ctrl: QuantumVariable | None = None,
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
    """
    from qrisp.jasp import jrange, jlen

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
        bit_i = _extract_bit(d, i)
        bit_inverted_mcx(
            target[i], dirty_ancillas[i - 1], dirty_ancillas[i], bit_i, ctrl=ctrl
        )

    # Flip each dirty slot if the addend bit is 1
    for i in jrange(num_dirty_qubits):
        bit_i = _extract_bit(d, i)
        with control(bit_i):
            if ctrl is None:
                x(dirty_ancillas[i])
            else:
                cx(ctrl, dirty_ancillas[i])

    if carry_in is not None:
        d0 = _extract_bit(d, 0)
        bit_inverted_zz_zz_mcx(carry_in, target[0], dirty_ancillas[0], d0, ctrl=ctrl)

    # Forward chain
    # Walk dirty slots from least significant bit to most significant bit.
    for j in jrange(num_dirty_qubits - 1):
        i = j + 1
        bit_i = _extract_bit(d, i)
        bit_inverted_zz_zz_mcx(
            dirty_ancillas[i - 1], target[i], dirty_ancillas[i], bit_i, ctrl=ctrl
        )


def dirty_ancillae_adder(
    d: int | BigInteger,
    target: QuantumVariable,
    dirty_ancillas: QuantumVariable,
    ancilla: QuantumVariable,
    c_in: QuantumVariable | None = None,
    ctrl: QuantumVariable | None = None,
) -> int:
    r"""Fig. 4 — dirty-ancilla CQ in-place adder.

    :math:`\text{target} \rightarrow (\text{target} + d + c_{\text{in}}) \bmod 2^n`

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
    target : QuantumVariable
        Target register, little-endian (index 0 = LSB). Modified in place.
    d : int
        Classical addend (compile-time constant).
    dirty_ancillas : QuantumVariable
        n-2 dirty workspace qubits.
    ancilla : QuantumVariable
        2-qubit clean ancilla register for the streaming carry chain.
    c_in : QuantumBool | None
        Quantum carry-in.  ``None`` is treated as ``|0⟩``.

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
    ventmask, _ = carry_venting_adder(
        d, target, ancilla=ancilla, c_in=c_in, carry_xor_target=dirty_qubits, ctrl=ctrl
    )

    # Step 2: Phase correction
    for i in jrange(num_targets):
        x(target[i])

    # CZ pass 1: correct Z-phases using ventmask
    for k in jrange(num_dirty):
        with control((ventmask >> k) & 1):
            z(dirty_qubits[k])

    # Second carry-xor pass: recompute carries against the flipped sum
    carry_xor_block(d, dirty_qubits, target[:-1], c_in, ctrl=ctrl)

    # CZ pass 2: correct Z-phases again
    for k in jrange(num_dirty):
        with control((ventmask >> k) & 1):
            z(dirty_qubits[k])

    # Restore target
    for i in jrange(num_targets):
        x(target[i])

    return ventmask


@custom_control
def gidney_cq_venting_adder(
    d: int | BigInteger,
    target: QuantumVariable,
    c_in: QuantumBool | None = None,
    ctrl: QuantumVariable | QuantumBool | None = None,
) -> None:
    r"""In-place classical-quantum adder using Gidney's carry-venting technique.

    Adds a classical integer :math:`d` and an optional carry-in :math:`c_{\text{in}}`
    to a quantum register, storing the result back in the same register.

    Uses 3 clean ancillae (allocated internally) and no external dirty workspace.
    The target register is split in half — each half serves as dirty storage for
    the other half's carry chain — keeping the total qubit count to ``n + 3``.

    Heavily adapted from the reference implementation released with
    `arXiv:2507.23079 <https://arxiv.org/abs/2507.23079>`_ at https://zenodo.org/records/15866587.

    .. warning::

       This function requires dynamic (JASP) mode and does **not** work in static
       circuit generation, because it relies on mid-circuit measurements (venting)
       whose outcomes determine subsequent operations.

    Parameters
    ----------
    target : QuantumVariable
        Target register, little-endian (index 0 = LSB). Modified in place.
    d : int or BigInteger
        Classical addend (compile-time constant).
    c_in : QuantumBool | None
        Quantum carry-in. None is treated as :math:`\ket{0}`.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``d`` is a :class:`QuantumVariable` or list (must be a classical integer).
    TypeError
        If ``target`` is not a :class:`QuantumVariable`.
    RuntimeError
        If called outside dynamic (JASP) tracing mode.

    Examples
    --------
    Basic addition in dynamic (JASP) mode::

        from qrisp import QuantumFloat, measure
        from qrisp.jasp import jaspify

        @jaspify
        def main():
            target = QuantumFloat(5)
            target[:] = 2
            gidney_cq_venting_adder(7, target)
            return measure(target)

        result = main()
        # result is 9 (2 + 7 = 9)

    Addition with a carry-in qubit::

        @jaspify
        def main():
            target = QuantumFloat(4)
            target[:] = 5
            c_in = QuantumFloat(1)
            c_in[:] = 1
            gidney_cq_venting_adder(2, target, c_in=c_in[0])
            return measure(target)

        result = main()
        # result is 8 (5 + 2 + carry_in = 8)

    Controlled addition (zero extra Toffoli cost)::

        from qrisp import control, QuantumBool

        @jaspify
        def main():
            target = QuantumFloat(5)
            target[:] = 10
            ctrl = QuantumBool()
            ctrl[:] = 1
            with control(ctrl):
                gidney_cq_venting_adder(5, target)
            return measure(target)

        result = main()
        # result is 15 (10 + 5 = 15)
        # when ctrl is not provided, target stays unchanged
    """
    from qrisp.jasp import jrange, jlen

    if isinstance(d, (QuantumVariable, list)):
        raise TypeError("The first argument must be a classical integer.")
    if not isinstance(target, QuantumVariable):
        raise TypeError("The second argument must be a QuantumVariable.")

    if not check_for_tracing_mode():
        raise RuntimeError(
            "The Gidney Classical-Quantum adder does not work in standard python execution mode."
        )

    num_targets = jlen(target)

    n_half = (
        num_targets - 1
    ) >> 1  # bottom half size (needs num_targets >= 3 for a meaningful split)

    # initialize 3 clean ancillae
    clean_anc = QuantumVariable(3, name="gidney_anc*")
    clean0 = clean_anc[0]  # mid-carry ancilla (clean0 in paper notation)
    # clean1, clean2 (streaming-carry ancillae) are not bound individually —
    # sub-functions receive them as a traceable slice: clean_anc[1:]

    # Split classical addend: d_lo is the lower n_half bits, d_hi is the upper bits
    d_lo = d & ((1 << n_half) - 1)
    d_hi = d >> n_half

    # Step 1: Place clean0 into target[n_half].
    # Three CNOTs implement a SWAP so that carry_venting_adder on the
    # (n_half+1)-bit slice target[:n_half+1] writes the carry out of bit n_half-1 into
    # clean0 (as its final carry).  Gidney's code does a Python-level
    # reference swap, but JASP's dynamic-length QuantumVariable requires
    # a quantum gate.
    cx(clean0, target[n_half])
    cx(target[n_half], clean0)
    cx(clean0, target[n_half])

    # Step 2: Vented addition on bottom half
    ventmask_lo, _ = carry_venting_adder(
        d_lo, target[: n_half + 1], clean_anc[1:], c_in=c_in, ctrl=ctrl
    )

    # Step 3: Swap clean0 back out of target[n_half].
    # target[n_half] is restored to its original qubit and clean0 now
    # holds the carry into the top half.
    cx(clean0, target[n_half])
    cx(target[n_half], clean0)
    cx(clean0, target[n_half])

    # Step 4: Top half addition using dirty ancillas.
    # target[n_half:] has n-n_half bits (top half).  clean0 is the carry-in.
    # target[:max(1, n-n_half-2)] (bottom n-n_half-2 bits, at least 1) are borrowed as
    # dirty workspace — they will be restored by dirty_ancillae_adder's internal
    # phase correction.
    # The same [clean1, clean2] is reused as the clean ancilla register, so the
    # total clean ancilla count is just 3 (shared between both halves).
    # note: n-n_half-2 may be less than n_half (e.g. n=5, n_half=2: n-n_half-2=1).  We use
    # n-n_half-2 dirty qubits (at least 1) because dirty_ancillae_adder expects
    # exactly num_targets - 2 dirty qubits.  The remaining bottom bit(s) stay
    # untouched and keep their value from the bottom-half sum.
    dirty_ancillae_adder(
        d_hi,
        target[n_half:],
        dirty_ancillas=target[: jnp.maximum(1, num_targets - n_half - 2)],
        ancilla=clean_anc[1:],
        c_in=clean0,
        ctrl=ctrl,
    )

    # Step 5: MX measurement on clean0.
    # The clean0 qubit still holds the carry out of the bottom half.
    # We measure it in the X-basis (H then Z-measurement).
    h(clean0)
    m_clean0 = measure(clean0)
    reset(clean0)

    # Combine ventmask_lo with m_clean0 to get the full ventmask for the
    # entire register.
    #
    # ventmask_lo bits 0..n_half-2 hold carries into bits 1..n_half-1 (from the
    # bottom-half adder).  m_clean0 is the carry into bit n_half.  These are
    # packed contiguously so that bit k of full_ventmask corresponds to the
    # carry into bit k+1, matching the dirty-workspace slots 0..n_half-1 used in
    # the phase correction (reference _add_3_clean.py line 123).
    ventmask_lo_mask = (1 << (n_half - 1)) - 1
    full_ventmask = (ventmask_lo & ventmask_lo_mask) | (
        m_clean0.astype(jnp.int64) << (n_half - 1)
    )

    # Step 6: Phase correction for bottom half vents

    workspace = target[n_half : 2 * n_half]
    bottom = target[:n_half]

    # 6a — Complement bottom half: NOT(bottom)
    for i in jrange(n_half):
        x(bottom[i])

    # 6b — CZ(workspace[k], vent[k+1])
    for k in jrange(n_half):
        with control((full_ventmask >> k) & 1):
            z(workspace[k])

    # 6c — First carry_xor pass (explicit, no fused pass for bottom half)
    carry_xor_block(d_lo, workspace, bottom, c_in, ctrl=ctrl)

    # 6d — CZ(workspace[k], vent[k+1]) again
    for k in jrange(n_half):
        with control((full_ventmask >> k) & 1):
            z(workspace[k])

    # 6e — Second carry_xor pass
    carry_xor_block(d_lo, workspace, bottom, c_in, ctrl=ctrl)

    # 6f — Restore bottom half: NOT cancels the complement from 6a
    for i in jrange(n_half):
        x(bottom[i])

    clean_anc.delete()

    return None
