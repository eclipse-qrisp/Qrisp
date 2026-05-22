from __future__ import annotations

import qrisp
from qrisp.environments import control
from qrisp.jasp import check_for_tracing_mode


def bit_inverted_mcx(
    parity_check_ctrl: qrisp.Qubit | qrisp.QuantumVariable,
    simple_ctrl: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
    b,
) -> None:
    """Figure 1 left: Toffoli with two controls, one inverted by classical bit b.

    parity_check_ctrl has the Z⊕b box and is implemented as an explicit X^b sandwich:
    apply X to parity_check_ctrl iff b is True, do a standard MCX, then undo X.
    simple_ctrl is a normal control (fires on |1⟩).

    Args:
        parity_check_ctrl: Control qubit with the Z⊕b box.
        simple_ctrl: Normal control qubit (fires on |1⟩).
        target: Target qubit for the Toffoli gate.
        b: Classical bit determining whether parity_check_ctrl is inverted (can be int, bool, or traced).
    """
    # b can be a traced value from JASP, so pass it directly to control()
    with control(b):
        qrisp.x(parity_check_ctrl)

    qrisp.mcx([parity_check_ctrl, simple_ctrl], target)

    with control(b):
        qrisp.x(parity_check_ctrl)


 
 
def zz_mcx(
    z0_left: qrisp.Qubit | qrisp.QuantumVariable,
    z1_left: qrisp.Qubit | qrisp.QuantumVariable,
    control: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
) -> None:
    """Figure 1 middle: Toffoli with one normal control and one ZZ parity control.

    z0_left, z1_left are the two Z-box qubits whose parity acts as one control.
    control is a normal control (fires on |1⟩).

    Args:
        z0_left: First Z-box qubit of the ZZ parity pair.
        z1_left: Second Z-box qubit of the ZZ parity pair.
        control: Normal control qubit (fires on |1⟩).
        target: Target qubit for the Toffoli gate.
    """
    qrisp.cx(z0_left, z1_left)
    qrisp.mcx([z1_left, control], target)
    qrisp.cx(z0_left, z1_left)

 
 
def zz_zz_mcx(
    z_left: qrisp.Qubit | qrisp.QuantumVariable,
    z_left_right: qrisp.Qubit | qrisp.QuantumVariable,
    z_right: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
) -> None:
    """Figure 1 right: Toffoli controlled on AND of two ZZ parity checks.

    Left pair:  (z_left, z_left_right) — Z box on left side.
    Right pair: (z_left_right, z_right) — Z box on right side.
    z_left_right is shared between both pairs.

    Args:
        z_left: Left qubit of the left ZZ parity pair.
        z_left_right: Shared middle qubit (right of left pair, left of right pair).
        z_right: Right qubit of the right ZZ parity pair.
        target: Target qubit for the Toffoli gate.
    """
    qrisp.cx(z_left_right, z_left)    # z_left now holds z_left ⊕ z_left_right (left parity)
    qrisp.cx(z_left_right, z_right)    # z_right now holds z_left_right ⊕ z_right (right parity)
    qrisp.mcx([z_left, z_right], target)
    qrisp.cx(z_left_right, z_right)    # restore z_right
    qrisp.cx(z_left_right, z_left)    # restore z_left


def bit_inverted_zz_zz_mcx(
    parity_ctrl1: qrisp.Qubit | qrisp.QuantumVariable,
    parity_ctrl2: qrisp.Qubit | qrisp.QuantumVariable,
    target: qrisp.Qubit | qrisp.QuantumVariable,
    b,
) -> None:
    """Inverted MCX with two direct parity controls.

    If b == 1, apply X to both controls before and after the MCX.

    Args:
        parity_ctrl1: First parity control qubit.
        parity_ctrl2: Second parity control qubit.
        target: Target qubit for the MCX gate.
        b: Classical bit determining whether the controls are inverted (can be int, bool, or traced).
    """
    # b can be a traced value from JASP, so pass it directly to control()
    with control(b):
        qrisp.x(parity_ctrl1)
        qrisp.x(parity_ctrl2)

    qrisp.mcx([parity_ctrl1, parity_ctrl2], target)

    with control(b):
        qrisp.x(parity_ctrl1)
        qrisp.x(parity_ctrl2)


def carry_venting_adder(
    target: list[qrisp.Qubit] | qrisp.QuantumVariable,
    d: int,
    c_in: qrisp.Qubit | qrisp.QuantumVariable,
) -> list:
    """Fig. 2 of https://arxiv.org/pdf/2507.23079 — carry-venting classical-quantum in-place adder.

    Performs

        target <- (target + d + c_in) mod 2^n

    using a streamed carry process with two clean ancillas and mid-circuit X-basis
    venting of obsolete carries. Returns the vented measurement bits in the order
    they are produced.

    Parameters
    ----------
    target : list[qrisp.Qubit] | qrisp.QuantumVariable
        Target register in little-endian bit order (index 0 is the least-significant bit).
        This register is modified in place.
    d : int
        Classical addend (Python int, not traced).
    c_in : qrisp.Qubit | qrisp.QuantumVariable
        Quantum carry-in. Supported forms are:

        - ``qrisp.Qubit``: quantum carry-in qubit.
        - size-1 ``qrisp.QuantumVariable``: quantum carry-in qubit container.

    Returns
    -------
    list
        List of vented carry measurement results, ordered by production during the
        streamed carry process.
    """
    from qrisp.jasp import jrange, jlen

    n = jlen(target)

    anc = qrisp.QuantumVariable(2, name="carry_vent_anc*")

    if c_in is None:
        raise ValueError("c_in must be provided")
    if isinstance(c_in, qrisp.QuantumVariable):
        carry_in = c_in[0]
    elif isinstance(c_in, qrisp.Qubit):
        carry_in = c_in
    else:
        raise TypeError("c_in must be a Qubit or size-1 QuantumVariable")

    vented = []

    # ── Initial setup: X^d_i on ALL target bits ──
    for i in jrange(n):
        d_i = (d >> i) & 1
        with control(d_i):
            qrisp.x(target[i])

    # ── First building block (bits 0 and 1) ──
    #
    # Bit 0: carry_in → anc[1]
    #   bit_inverted_mcx(carry_in, target[0], anc[1], d0)
    #   CX(carry_in, target[0])
    #   X^d_0 on anc[1]
    #
    # Bit 1: anc[1] → anc[0]
    #   bit_inverted_mcx(anc[1], target[1], anc[0], d1)
    #   CX(anc[1], target[1])
    #   VENT anc[1]

    d0 = (d >> 0) & 1
    d1 = (d >> 1) & 1

    bit_inverted_mcx(carry_in, target[0], anc[1], d0)
    qrisp.cx(carry_in, target[0])
    with control(d0):
        qrisp.x(anc[1])

    bit_inverted_mcx(anc[1], target[1], anc[0], d1)
    qrisp.cx(anc[1], target[1])

    qrisp.h(anc[1])
    vented.append(qrisp.measure(anc[1]))
    qrisp.reset(anc[1])

    # ── Normal repeated blocks (bits 2 .. n-3) ──
    #
    # For bit i (where i = j + 2):
    #   cur = anc[i % 2]           (holds current carry)
    #   nxt = anc[(i + 1) % 2]    (receives next carry)
    #
    #   X^d_{i-1} on cur
    #   bit_inverted_mcx(cur, target[i], nxt, d_i)
    #   CX(cur, target[i])
    #   VENT cur

    for j in jrange(n - 4):
        i = j + 2
        d_i = (d >> i) & 1
        d_prev = (d >> (i - 1)) & 1

        cur = anc[i % 2]
        nxt = anc[(i + 1) % 2]

        with control(d_prev):
            qrisp.x(cur)

        bit_inverted_mcx(cur, target[i], nxt, d_i)
        qrisp.cx(cur, target[i])

        qrisp.h(cur)
        vented.append(qrisp.measure(cur))
        qrisp.reset(cur)

    # ── Last block (bit n-2 → target[n-1]) ──
    #
    # Three regimes:
    #   n = 2  → no-op (first block already covers all bits)
    #   n = 3  → simplified: carry c_2 already in anc[0]; just correct, CX to
    #            target[2], and vent anc[0].
    #   n >= 4 → full MCX-based last block writing into target[n-1].

    import jax.numpy as jnp

    with control(n > 2):
        last_i = n - 2
        d_last = (d >> last_i) & 1

        num_normal = jnp.maximum(0, n - 4)
        cur = anc[num_normal % 2]

        # Correction common to both paths
        d_correction = (d >> jnp.maximum(1, n - 3)) & 1
        with control(d_correction):
            qrisp.x(cur)

        # ── n == 3: simplified (just CX carry to last target, no MCX) ──
        with control(n == 3):
            qrisp.cx(cur, target[last_i + 1])

        # ── n >= 4: full MCX-based last block ──
        with control(n > 3):
            bit_inverted_mcx(cur, target[last_i], target[n - 1], d_last)
            qrisp.cx(cur, target[last_i])

        # Vent (common to both paths, outside conditional branches)
        qrisp.h(cur)
        vented.append(qrisp.measure(cur))
        qrisp.reset(cur)

        # Final X on target[n-1] (only needed for n >= 4)
        with control(n > 3):
            with control(d_last):
                qrisp.x(target[n - 1])

    return vented

def carry_xor_block(
    target: list[qrisp.Qubit] | qrisp.QuantumVariable,
    dirty_ancillas: list[qrisp.Qubit] | qrisp.QuantumVariable,
    d: int,
    c_in: qrisp.Qubit | qrisp.QuantumVariable | None = None,
) -> None:
    """Fig. 3 of https://arxiv.org/pdf/2507.23079 — carry-XOR block using dirty ancillas.

    This implements the requested first pass:
    1. Reverse-chain inverted bit-MCX operations from the last target bit down to
       the second target bit.
    2. Conditional X on each dirty ancilla when the corresponding d_i bit is 1.

    If a carry-in is provided, an additional first two-parity-control step is applied
    on the first dirty ancilla.
    """
    from qrisp.jasp import jrange, jlen

    n = jlen(target)

    # Optional carry-in normalization.
    if c_in is None:
        carry_in = None
    elif isinstance(c_in, qrisp.QuantumVariable):
        carry_in = c_in[0]
    elif isinstance(c_in, qrisp.Qubit):
        carry_in = c_in
    else:
        raise TypeError("c_in must be None, a Qubit, or a size-1 QuantumVariable")

    # Reverse-chain inverted bit-MCX:
    # i = n-1, n-2, ..., 1
    # parity control = target[i]
    # simple control = dirty[i-1]
    # target         = dirty[i]
    for j in jrange(n - 1):
        i = n - 1 - j
        bit_i = (d >> i) & 1
        bit_inverted_mcx(target[i], dirty_ancillas[i - 1], dirty_ancillas[i], bit_i)

    # Apply X to each dirty ancilla conditioned on d_i.
    for i in jrange(n):
        bit_i = (d >> i) & 1
        with control(bit_i):
            qrisp.x(dirty_ancillas[i])

    # Two-parity-control step on the first dirty ancilla (placed after conditioned X gates).
    # Applied only when an incoming carry line is provided.
    if carry_in is not None:
        d0 = (d >> 0) & 1
        bit_inverted_zz_zz_mcx(carry_in, target[0], dirty_ancillas[0], d0)

    # Forward chain:
    # parity controls: dirty_bits[i-1], target_bits[i]
    # target: dirty_bits[i]
    # conditioned on d_i, for i = 1..n-1
    for j in jrange(n - 1):
        i = j + 1
        bit_i = (d >> i) & 1
        bit_inverted_zz_zz_mcx(dirty_ancillas[i - 1], target[i], dirty_ancillas[i], bit_i)


def gidney_cq_venting_adder(
    target: qrisp.QuantumVariable | list[qrisp.Qubit],
    d: int,
    c_in: qrisp.Qubit | qrisp.QuantumVariable | None = None,
    ancilla: qrisp.QuantumVariable | None = None,
    dirty_ancillas: qrisp.QuantumVariable | list[qrisp.Qubit] | None = None,
    use_dirty_ancillas: bool = False,
) -> list | None:
    if not check_for_tracing_mode():
        raise RuntimeError("gidney_cq_venting_adder requires JASP dynamic mode")

    if use_dirty_ancillas:
        if dirty_ancillas is None:
            raise ValueError("dirty_ancillas must be provided when use_dirty_ancillas is True")
        return carry_xor_block(target, dirty_ancillas, d, c_in)
    else:
        return carry_venting_adder(target, d, c_in)