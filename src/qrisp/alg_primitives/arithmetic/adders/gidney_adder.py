import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx, merge
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def _is_qubit_list(value, allow_empty=False):
    """Return True when ``value`` is a list of qubit-like objects."""
    if not isinstance(value, list):
        return False
    if not allow_empty and len(value) == 0:
        return False
    return all(callable(getattr(qb, "qs", None)) for qb in value)


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

    The ``a_int_is_bigint=True`` path calls ``get_bit`` on ``a_int``:

    >>> class _DummyBigInt:
    ...     def __init__(self, bits):
    ...         self.bits = bits
    ...     def get_bit(self, i):
    ...         return self.bits[i]
    >>> bool(_extract_bit(_DummyBigInt([0, 1, 0]), 1, True))
    True
    """
    if a_int_is_bigint:
        return jnp.bool_(a_int.get_bit(digit_index))
    return jnp.bool_((a_int >> digit_index) & 1)


def _apply_classical_carry_chain(
    gidney_anc,
    b_qbs,
    n,
    a_int,
    a_bits,
    a_bit0,
    a_int_is_bigint,
    ctrl,
):
    """Apply carry-chain phases for classical input values.

    Parameters
    ----------
    gidney_anc : QuantumVariable
        Carry ancilla register with ``n - 1`` qubits.
    b_qbs : list
        Target qubits participating in the carry chain.
    n : int
        Effective carry-chain width, including optional carry wires.
    a_int : int, jnp.ndarray scalar, or BigInteger
        Classical input representation used for traced bit extraction.
    a_bits : list[int] or None
        Precomputed static bits of ``a_int`` in non-tracing mode.
    a_bit0 : bool or jnp.bool_
        Least-significant bit of ``a_int`` for traced control flow.
    a_int_is_bigint : bool
        Indicates whether ``a_int`` must be read via ``get_bit``.
    ctrl : Qubit or None
        Optional outer control qubit.
    """
    if check_for_tracing_mode():
        with control(a_bit0):
            if ctrl is not None:
                mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney")
            else:
                cx(b_qbs[0], gidney_anc[0])
    elif a_bits[0]:
        if ctrl is not None:
            mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney")
        else:
            cx(b_qbs[0], gidney_anc[0])

    if check_for_tracing_mode():
        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], b_qbs[i])

            bit_i = _extract_bit(a_int, i, a_int_is_bigint)
            with control(bit_i):
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")

            with control(bit_i):
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

            cx(gidney_anc[i - 1], gidney_anc[i])
    else:
        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], b_qbs[i])
            bit_i = a_bits[i]

            if bit_i:
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")

            if bit_i:
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

            cx(gidney_anc[i - 1], gidney_anc[i])

    cx(gidney_anc[n - 2], b_qbs[n - 1])

    if check_for_tracing_mode():
        for j in jrange(n - 2):
            i = n - j - 2
            cx(gidney_anc[i - 1], gidney_anc[i])

            bit_i = _extract_bit(a_int, i, a_int_is_bigint)
            with control(bit_i):
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")

            with control(bit_i):
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])
    else:
        for j in jrange(n - 2):
            i = n - j - 2
            cx(gidney_anc[i - 1], gidney_anc[i])
            bit_i = a_bits[i]

            if bit_i:
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")

            if bit_i:
                if ctrl is not None:
                    cx(ctrl, gidney_anc[i - 1])
                else:
                    x(gidney_anc[i - 1])

    if check_for_tracing_mode():
        with control(a_bit0):
            if ctrl is not None:
                mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney_inv")
            else:
                cx(b_qbs[0], gidney_anc[0])
    elif a_bits[0]:
        if ctrl is not None:
            mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney_inv")
        else:
            cx(b_qbs[0], gidney_anc[0])


def _apply_quantum_carry_chain(gidney_anc, a_qbs, b_qbs, n, c_in_qb, c_out_qb, ctrl, qs):
    """Apply carry-chain phases for quantum input registers.

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
    qs : QuantumSession or None
        Session used for auxiliary allocations in non-tracing mode.
    """
    if c_in_qb is not None:
        # Conjugation trick: encode c_in into a/b before the Gidney AND, then unmask on anc.
        cx(c_in_qb, b_qbs[0])
        cx(c_in_qb, a_qbs[0])
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")
        cx(c_in_qb, gidney_anc[0])
    else:
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")

    for j in jrange(n - 2):
        i = j + 1
        # Move carry dependency forward one position in the ripple.
        cx(gidney_anc[i - 1], b_qbs[i])
        cx(gidney_anc[i - 1], a_qbs[i])
        mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney")
        cx(gidney_anc[i - 1], gidney_anc[i])

    if ctrl is not None:
        mcx([ctrl, gidney_anc[n - 2]], b_qbs[n - 1])
        if c_out_qb is None:
            mcx([ctrl, a_qbs[n - 1]], b_qbs[n - 1])
    else:
        cx(gidney_anc[n - 2], b_qbs[n - 1])
        if c_out_qb is None:
            cx(a_qbs[n - 1], b_qbs[n - 1])

    for j in jrange(n - 2):
        i = n - j - 2
        # Reverse sweep uncomputes carry garbage while preserving the sum in b.
        cx(gidney_anc[i - 1], gidney_anc[i])
        mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney_inv")
        if ctrl is not None:
            mcx([ctrl, a_qbs[i]], b_qbs[i])
            cx(gidney_anc[i - 1], a_qbs[i])
            cx(gidney_anc[i - 1], b_qbs[i])
        else:
            cx(gidney_anc[i - 1], a_qbs[i])
            cx(a_qbs[i], b_qbs[i])

    if c_in_qb is not None:
        cx(c_in_qb, gidney_anc[0])
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")

        if ctrl is not None:
            lsb_ctrl_anc = (
                QuantumBool(name="gidney_lsb_ctrl_anc*")
                if check_for_tracing_mode()
                else QuantumBool(name="gidney_lsb_ctrl_anc*", qs=qs)
            )
            # This ancilla keeps c_in handling equivalent under an outer control.
            mcx([ctrl, c_in_qb], lsb_ctrl_anc[0], method="gidney")
            cx(lsb_ctrl_anc[0], b_qbs[0])
            mcx([ctrl, c_in_qb], lsb_ctrl_anc[0], method="gidney_inv")
            cx(c_in_qb, b_qbs[0])
            lsb_ctrl_anc.delete()

        cx(c_in_qb, a_qbs[0])
    else:
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")


def _apply_classical_final_sum(b_qbs, n, c_in_qb, a_int, a_bits, a_int_is_bigint, ctrl):
    """Apply final sum update for classical input values.

    In the Gidney adder of arXiv:1709.06648, the carry chain computes and then
    uncomputes carry information, but it does not by itself write every sum bit.
    This final step performs the remaining bitwise XOR updates on ``b`` so the
    register holds ``b <- b + a`` (or ``b <- b + a + c_in`` when carry-in is
    used). For classical ``a``, this is done by toggling ``b[i]`` exactly when
    bit ``i`` of ``a`` is 1. When ``c_in_qb`` is present, index 0 is skipped
    because the least-significant contribution was already incorporated earlier
    in the carry construction.

    Parameters
    ----------
    b_qbs : list
        Target qubits that receive the final bitwise updates.
    n : int
        Effective width of the active target region.
    c_in_qb : Qubit or None
        Optional carry-in qubit; bit 0 is skipped when present.
    a_int : int, jnp.ndarray scalar, or BigInteger
        Classical input representation used for traced bit extraction.
    a_bits : list[int] or None
        Precomputed static bits of ``a_int`` in non-tracing mode.
    a_int_is_bigint : bool
        Indicates whether ``a_int`` must be read via ``get_bit``.
    ctrl : Qubit or None
        Optional outer control qubit.
    """
    if check_for_tracing_mode():
        for i in jrange(n):
            if c_in_qb is not None and i == 0:
                continue

            bit_i = _extract_bit(a_int, i, a_int_is_bigint)
            with control(bit_i):
                if ctrl is not None:
                    cx(ctrl, b_qbs[i])
                else:
                    x(b_qbs[i])
        return

    if ctrl is not None:
        for i in jrange(n):
            if c_in_qb is not None and i == 0:
                continue

            bit_i = a_bits[i]
            if bit_i:
                cx(ctrl, b_qbs[i])
    else:
        for i in jrange(n):
            if c_in_qb is not None and i == 0:
                continue

            bit_i = a_bits[i]
            if bit_i:
                x(b_qbs[i])


def _apply_quantum_final_sum(a_qbs, b_qbs, ctrl):
    """Apply final sum update for quantum input registers.

    In the Gidney adder of arXiv:1709.06648, after carry computation and
    uncomputation, one remaining sum write is required at the least-significant
    position. This helper performs that final XOR write, ``b[0] ^= a[0]`` (or
    the controlled variant), completing the in-place map ``b <- b + a``.

    Parameters
    ----------
    a_qbs : list
        Input-a qubits used for the least-significant sum update.
    b_qbs : list
        Target qubits that receive the update.
    ctrl : Qubit or None
        Optional outer control qubit.
    """
    if ctrl is not None:
        mcx([ctrl, a_qbs[0]], b_qbs[0])
    else:
        cx(a_qbs[0], b_qbs[0])


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
    b : QuantumVariable, DynamicQubitArray, or list
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
    # Add BigInteger compatibility for traced modulus arithmetic (jasp_modulus).
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    # Validate input combination: allowed pairs are
    # (classical a, quantum b) and (quantum a, quantum b).
    invalid_input_pair_msg = (
        "gidney_adder expects inputs to be either classical-quantum "
        "(classical a, quantum b) or quantum-quantum (quantum a, quantum b)."
    )

    b_is_quantum_obj = isinstance(b, (QuantumVariable, DynamicQubitArray))
    a_is_quantum_obj = isinstance(a, (QuantumVariable, DynamicQubitArray))
    b_is_qubit_list = _is_qubit_list(b, allow_empty=False)
    a_is_qubit_list = _is_qubit_list(a, allow_empty=True)

    if isinstance(b, list) and not b_is_qubit_list:
        raise ValueError(
            invalid_input_pair_msg
            + " Received invalid list for b; expected a non-empty list of qubits."
        )

    if isinstance(a, list) and not a_is_qubit_list:
        raise ValueError(
            invalid_input_pair_msg
            + " Received invalid list for a; expected a list of qubits."
        )

    b_is_quantum = b_is_quantum_obj or b_is_qubit_list
    a_is_quantum = a_is_quantum_obj or a_is_qubit_list
    is_concrete_int = isinstance(a, (int, np.integer, str))
    is_scalar_like = getattr(a, "ndim", None) == 0
    is_biginteger_like = hasattr(a, "get_bit")
    is_valid_classical = is_concrete_int or (
        check_for_tracing_mode() and (is_scalar_like or is_biginteger_like)
    )
    valid_input_pair = b_is_quantum and (a_is_quantum or is_valid_classical)
    if not valid_input_pair:
        raise ValueError(invalid_input_pair_msg)

    # Normalize optional carry wires so downstream code always sees raw qubits.
    c_in_qb = c_in[0] if isinstance(c_in, QuantumBool) else c_in
    c_out_qb = c_out[0] if isinstance(c_out, QuantumBool) else c_out

    # Variables to decide whether we run the semi-classical or quantum-quantum variant.
    a_int = None
    a_bits = None
    a_bit0 = None
    a_qbs = None
    a_ext = None
    b_qbs = None
    n = None
    qs = None

    # ---- Semi-classical setup (classical a, quantum b) ----
    if not a_is_quantum:
        if isinstance(a, str):
            # Strings are little-endian bit strings in this API.
            a = int(a[::-1], 2) if a else 0

        is_concrete_int = isinstance(a, (int, np.integer))
        is_scalar_like = getattr(a, "ndim", None) == 0
        is_biginteger_like = hasattr(a, "get_bit")
        # In tracing mode we accept scalar-like symbolic values as classical input.
        if is_concrete_int or (check_for_tracing_mode() and (is_scalar_like or is_biginteger_like)):
            if is_concrete_int:
                # Match the target register width for deterministic static behavior.
                a = int(a) % (1 << jlen(b))

            if check_for_tracing_mode() and isinstance(a, int):
                a = jnp.array(a, dtype=jnp.int64)
            elif not check_for_tracing_mode():
                qs_list = [b[0].qs()]
                for qb in [c_in, c_out, ctrl]:
                    if qb is not None:
                        if isinstance(qb, QuantumBool):
                            qb = qb[0]
                        qs_list.append(qb.qs())
                merge(qs_list)

            b_qbs = b[:]
            if c_in_qb is not None:
                # Fold carry-in into bit 0 so the same carry chain can be reused.
                b_qbs = [c_in_qb] + b_qbs
                a = (a << 1) + 1
            if c_out_qb is not None:
                # Append carry-out as an extra MSB slot.
                b_qbs = b_qbs + [c_out_qb]

            n = jlen(b_qbs)
            qs = None if check_for_tracing_mode() else b_qbs[0].qs()
            a_int = a
            if not check_for_tracing_mode():
                a_int_py = int(a_int)
                # Precompute classical input bits for the static branch.
                a_bits = [(a_int_py >> i) & 1 for i in range(n)]

                if n == 1:
                    if a_bits[0]:
                        if ctrl is None:
                            x(b_qbs[0])
                        else:
                            cx(ctrl, b_qbs[0])
                    return
        else:
            # Non-classical input falls through to the full quantum path.
            a_is_quantum = True

    # ---- Quantum setup (quantum a, quantum b) ----
    if a_is_quantum:
        qs = None if check_for_tracing_mode() else b[0].qs()

        dim_a = jlen(a)
        dim_b = jlen(b)
        # Arithmetic is in-place on b, so ignore input-a bits beyond len(b).
        effective_size_a = jnp.minimum(dim_a, dim_b)
        a = a[:effective_size_a]

        extension_size = jnp.maximum(0, dim_b - dim_a)
        # Pad shorter a with |0> ancilla bits so index arithmetic stays uniform.
        a_ext = (
            QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs)
            if qs is not None
            else QuantumVariable(extension_size, name="gidney_a_ext*")
        )

        a_qbs = a[:] + a_ext[:]
        b_qbs = b[:]
        if c_out_qb is not None:
            b_qbs = b_qbs + [c_out_qb]

        n = jlen(b_qbs)

    # BigInteger appears in traced modulus workflows (jasp_modulus).
    a_int_is_bigint = (not a_is_quantum) and check_for_tracing_mode() and isinstance(a_int, BigInteger)
    if (not a_is_quantum) and check_for_tracing_mode():
        a_bit0 = _extract_bit(a_int, 0, a_int_is_bigint)

    # ---- Shared carry chain ----
    with control(n > 1):
        # The carry chain only exists when there are at least two participating bits.
        gidney_anc = (
            QuantumVariable(n - 1, name="gidney_anc*")
            if check_for_tracing_mode()
            else QuantumVariable(n - 1, name="gidney_anc*", qs=qs)
        )

        if a_is_quantum:
            _apply_quantum_carry_chain(gidney_anc, a_qbs, b_qbs, n, c_in_qb, c_out_qb, ctrl, qs)
        else:
            _apply_classical_carry_chain(
                gidney_anc,
                b_qbs,
                n,
                a_int,
                a_bits,
                a_bit0,
                a_int_is_bigint,
                ctrl,
            )

        gidney_anc.delete()

    # ---- Final sum stage ----
    if a_is_quantum:
        # Final LSB sum update happens after carry ancillas are cleaned.
        _apply_quantum_final_sum(a_qbs, b_qbs, ctrl)

        a_ext.delete()
        return

    _apply_classical_final_sum(b_qbs, n, c_in_qb, a_int, a_bits, a_int_is_bigint, ctrl)