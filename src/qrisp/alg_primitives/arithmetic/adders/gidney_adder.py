import jax.numpy as jnp
import numpy as np

from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control, conjugate
from qrisp.misc import int_encoder
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on `Gidney 2018 <https://arxiv.org/abs/1709.06648>`__. The circuit
    has a V-shaped structure:

    - A forward **left branch** builds up carry information using Gidney
      logical-AND gates (each costing 4 T gates).
    - A **tip** propagates the final carry into the most significant
      result bit.
    - An inverse **right branch** uncomputes the carry ancillae using
      measurement-based uncomputation (0 T gates each).

    The total T-count is 4*(n-1) where n is the number of bit positions,
    matching Figure 1 of the original paper.

    Carry-in and carry-out are supported by extending the input arrays:
    a |1> ancilla is prepended for carry-in, and a |0> ancilla is
    appended for carry-out. This turns carries into regular addition
    positions, keeping the V-shape logic uniform.

    Works in both standard Python (static) and jasp tracing modes.

    Parameters
    ----------
    a : int, str, QuantumVariable, DynamicQubitArray, or list[Qubit]
        The value to add. Integers and little-endian bit-strings are
        supported as classical inputs.
    b : QuantumVariable or list[Qubit]
        The target register, modified in place: ``b += a``.
    c_in : Qubit or QuantumBool, optional
        Carry-in qubit.
    c_out : Qubit or QuantumBool, optional
        Carry-out qubit (set to |1> on overflow).
    ctrl : Qubit, optional
        Control qubit (injected by ``@custom_control``).
    """

    # ==================================================================
    # Classical input handling
    # ==================================================================
    # When `a` is not a quantum type, we encode it into a temporary
    # quantum register using conjugation: int_encoder loads the value
    # before the addition and unloads it afterward, ensuring the
    # temporary register returns to |0> for clean deletion.
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):

        # Little-endian bit-string conversion:
        # "011" means qubit 0 = LSB = 0, qubit 1 = 1, qubit 2 = 1
        # Reverse to get big-endian "110" for Python's int("110", 2) = 6
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        # Reduce mod 2^len(b) since addition is in-place on b;
        # any bits of `a` beyond b's width don't affect the result.
        if not check_for_tracing_mode() and isinstance(a, (int, np.integer)):
            a = int(a) % (1 << len(b))

        qs = None if check_for_tracing_mode() else b[0].qs()
        q_a = QuantumVariable(jlen(b), qs=qs)

        with conjugate(int_encoder)(q_a, a):
            gidney_adder(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return

    # ==================================================================
    # Quantum input: equalize lengths of a and b
    # ==================================================================
    # If len(a) > len(b), truncate a (extra high bits can't affect b).
    # If len(a) < len(b), pad a with |0> qubits so both have the same
    # length. This padding register (a_ext) is deleted at the end.
    qs = None if check_for_tracing_mode() else b[0].qs()

    dim_a = jlen(a)
    dim_b = jlen(b)

    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs)

    a = a[:] + a_ext[:]
    b = b[:]

    # ==================================================================
    # Carry-in handling
    # ==================================================================
    # Prepend a |1> ancilla to `a` and the carry-in qubit to `b`.
    # This creates an extra addition position at the bottom where
    # a=1 and b=c_in. Their sum (1 + c_in) naturally produces a
    # carry of c_in into the next position, which is exactly the
    # carry-in behavior we want.
    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        one_anc = QuantumBool(name="gidney_cin_one*", qs=qs)
        x(one_anc[0])
        a = one_anc[:] + a
        b = c_in_qbs + b

    # ==================================================================
    # Carry-out handling
    # ==================================================================
    # Append a |0> ancilla to `a` and the carry-out qubit to `b`.
    # This creates an extra addition position at the top that captures
    # any overflow carry. Since a=0 at this position, b's carry-out
    # qubit will be flipped only if the addition produces a carry
    # from the most significant bit.
    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        zero_anc = QuantumBool(name="gidney_cout_zero*", qs=qs)
        a = a + zero_anc[:]
        b = b + c_out_qbs

    n = jlen(a)

    # ==================================================================
    # V-shaped addition circuit
    # ==================================================================
    # The circuit only needs the V-shape when n > 1 (more than one bit).
    # For n == 1, a single CX gate at the end suffices.
    with control(n > 1):
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*", qs=qs)

        # --- Left branch: compute carries ----------------------------
        #
        # Build up carry information from LSB to MSB. Each iteration
        # computes carry[i] = (a[i] AND b[i]) XOR carry[i-1] using a
        # Gidney logical-AND gate (4 T gates, stored in gidney_anc[i]).
        #
        # Before the Toffoli at position i, CX gates propagate the
        # previous carry into a[i] and b[i] so the Toffoli computes
        # the correct carry for the next position.

        # First Toffoli: no previous carry to propagate
        mcx([a[0], b[0]], gidney_anc[0], method="gidney")

        # Remaining positions: propagate carry, then compute new carry
        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], a[i])
            cx(gidney_anc[i - 1], b[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])

        # --- Tip: write the MSB of the result ------------------------
        #
        # The most significant result bit b[n-1] gets the final carry
        # (from gidney_anc[n-2]) and the XOR of a[n-1] and b[n-1].
        # In the controlled case, both contributions are gated on ctrl.
        if ctrl is not None:
            mcx([ctrl, gidney_anc[n - 2]], b[n - 1])
            mcx([ctrl, a[n - 1]], b[n - 1])
        else:
            cx(gidney_anc[n - 2], b[n - 1])
            cx(a[n - 1], b[n - 1])

        # --- Right branch: uncompute carries and write sum bits ------
        #
        # Walk back from MSB to LSB, undoing the carry computation.
        # Each iteration:
        #   1. Undo the carry propagation: cx(gidney_anc[i-1], gidney_anc[i])
        #   2. Uncompute the Toffoli using measurement-based
        #      uncomputation (gidney_inv, 0 T gates)
        #   3. Write the sum bit into b[i]:
        #      - Uncontrolled: restore a[i] then cx(a[i], b[i])
        #      - Controlled: gate the sum write on ctrl
        for j in jrange(n - 2):
            i = n - j - 2
            cx(gidney_anc[i - 1], gidney_anc[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney_inv")

            if ctrl is not None:
                mcx([ctrl, a[i]], b[i])
                cx(gidney_anc[i - 1], a[i])
                cx(gidney_anc[i - 1], b[i])
            else:
                cx(gidney_anc[i - 1], a[i])
                cx(a[i], b[i])

        # Uncompute the first Toffoli (no carry propagation to undo)
        mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")

        gidney_anc.delete(verify=False)

    # ==================================================================
    # Final sum bit for position 0
    # ==================================================================
    # The V-shape handles positions 1..n-1. Position 0's sum bit
    # (b[0] ^= a[0]) is written here. When carry-in is used, this
    # position is occupied by the carry-in ancillae, so no CX is
    # needed — the carry-in contribution is already incorporated.
    if c_in is None:
        if ctrl is not None:
            mcx([ctrl, a[0]], b[0])
        else:
            cx(a[0], b[0])

    # ==================================================================
    # Clean up carry ancillae
    # ==================================================================
    # Restore and delete the carry-in ancilla (flip back from |1> to |0>)
    if one_anc is not None:
        x(one_anc[0])
        one_anc.delete()

    # Delete the carry-out ancilla (already in |0> since a was |0> there)
    if zero_anc is not None:
        zero_anc.delete()

    # Delete the padding register used to equalize a and b lengths
    a_ext.delete()