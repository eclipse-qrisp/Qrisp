import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx, merge
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def _extract_bit(a_int, digit_index, a_int_is_bigint):
    """Extract a single bit from a classical scalar as a JAX boolean."""
    if a_int_is_bigint:
        return jnp.bool_(a_int.get_bit(digit_index))
    return jnp.bool_((a_int >> digit_index) & 1)


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
    # Keep BigInteger compatibility for traced modulus arithmetic (jasp_modulus).
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    c_in_qb = c_in[0] if isinstance(c_in, QuantumBool) else c_in
    c_out_qb = c_out[0] if isinstance(c_out, QuantumBool) else c_out

    a_is_quantum = isinstance(a, (QuantumVariable, DynamicQubitArray, list))
    a_int = None
    a_int_py = None
    a_bits = None
    a_qbs = None
    a_ext = None

    # ---- Semi-classical setup (classical a, quantum b) ----
    if not a_is_quantum:
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        is_concrete_int = isinstance(a, (int, np.integer))
        is_scalar_like = getattr(a, "ndim", None) == 0
        is_biginteger_like = hasattr(a, "get_bit")
        if is_concrete_int or (check_for_tracing_mode() and (is_scalar_like or is_biginteger_like)):
            if is_concrete_int:
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
                b_qbs = [c_in_qb] + b_qbs
                a = (a << 1) + 1
            if c_out_qb is not None:
                b_qbs = b_qbs + [c_out_qb]

            n = jlen(b_qbs)
            qs = None if check_for_tracing_mode() else b_qbs[0].qs()
            a_int = a
            if not check_for_tracing_mode():
                a_int_py = int(a_int)
                a_bits = [(a_int_py >> i) & 1 for i in range(n)]

            if (not check_for_tracing_mode()) and n == 1:
                if a_bits[0]:
                    if ctrl is None:
                        x(b_qbs[0])
                    else:
                        cx(ctrl, b_qbs[0])
                return
        else:
            a_is_quantum = True

    # ---- Quantum setup (quantum a, quantum b) ----
    if a_is_quantum:
        qs = None if check_for_tracing_mode() else b[0].qs()

        dim_a = jlen(a)
        dim_b = jlen(b)
        effective_size_a = jnp.minimum(dim_a, dim_b)
        a = a[:effective_size_a]

        extension_size = jnp.maximum(0, dim_b - dim_a)
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

    a_int_is_bigint = (not a_is_quantum) and check_for_tracing_mode() and isinstance(a_int, BigInteger)

    # ---- Shared carry ladder ----
    with control(n > 1):
        gidney_anc = (
            QuantumVariable(n - 1, name="gidney_anc*")
            if check_for_tracing_mode()
            else QuantumVariable(n - 1, name="gidney_anc*", qs=qs)
        )

        # Phase 1: initialize carry_0
        if not a_is_quantum:
            # Classical addend path
            if check_for_tracing_mode():
                bit_0 = _extract_bit(a_int, 0, a_int_is_bigint)
                with control(bit_0):
                    if ctrl is not None:
                        mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney")
                    else:
                        cx(b_qbs[0], gidney_anc[0])
            elif a_bits[0]:
                if ctrl is not None:
                    mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney")
                else:
                    cx(b_qbs[0], gidney_anc[0])

        elif c_in_qb is not None:
            # Quantum addend path with carry-in
            cx(c_in_qb, b_qbs[0])
            cx(c_in_qb, a_qbs[0])
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")
            cx(c_in_qb, gidney_anc[0])
        else:
            # Quantum addend path without carry-in
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")

        # Phase 2: forward carry propagation
        if not a_is_quantum:
            # Classical addend path
            if check_for_tracing_mode():
                if ctrl is not None:
                    for j in jrange(n - 2):
                        i = j + 1
                        cx(gidney_anc[i - 1], b_qbs[i])

                        bit_i = _extract_bit(a_int, i, a_int_is_bigint)
                        with control(bit_i):
                            cx(ctrl, gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")

                        with control(bit_i):
                            cx(ctrl, gidney_anc[i - 1])

                        cx(gidney_anc[i - 1], gidney_anc[i])
                else:
                    for j in jrange(n - 2):
                        i = j + 1
                        cx(gidney_anc[i - 1], b_qbs[i])

                        bit_i = _extract_bit(a_int, i, a_int_is_bigint)
                        with control(bit_i):
                            x(gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")

                        with control(bit_i):
                            x(gidney_anc[i - 1])

                        cx(gidney_anc[i - 1], gidney_anc[i])
            else:
                if ctrl is not None:
                    for j in jrange(n - 2):
                        i = j + 1
                        cx(gidney_anc[i - 1], b_qbs[i])
                        bit_i = a_bits[i]

                        if bit_i:
                            cx(ctrl, gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")

                        if bit_i:
                            cx(ctrl, gidney_anc[i - 1])

                        cx(gidney_anc[i - 1], gidney_anc[i])
                else:
                    for j in jrange(n - 2):
                        i = j + 1
                        cx(gidney_anc[i - 1], b_qbs[i])
                        bit_i = _extract_bit(a_int, i, a_int_is_bigint)

                        if bit_i:
                            x(gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")

                        if bit_i:
                            x(gidney_anc[i - 1])

                        cx(gidney_anc[i - 1], gidney_anc[i])
        else:
            # Quantum addend path
            for j in jrange(n - 2):
                i = j + 1
                cx(gidney_anc[i - 1], b_qbs[i])
                cx(gidney_anc[i - 1], a_qbs[i])
                mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])

        # Phase 3: midpoint
        if not a_is_quantum:
            # Classical addend path
            cx(gidney_anc[n - 2], b_qbs[n - 1])
        else:
            # Quantum addend path
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], b_qbs[n - 1])
                if c_out_qb is None:
                    mcx([ctrl, a_qbs[n - 1]], b_qbs[n - 1])
            else:
                cx(gidney_anc[n - 2], b_qbs[n - 1])
                if c_out_qb is None:
                    cx(a_qbs[n - 1], b_qbs[n - 1])

        # Phase 4: backward uncomputation
        if not a_is_quantum:
            # Classical addend path
            if check_for_tracing_mode():
                if ctrl is not None:
                    for j in jrange(n - 2):
                        i = n - j - 2
                        cx(gidney_anc[i - 1], gidney_anc[i])

                        bit_i = _extract_bit(a_int, i, a_int_is_bigint)
                        with control(bit_i):
                            cx(ctrl, gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")

                        with control(bit_i):
                            cx(ctrl, gidney_anc[i - 1])
                else:
                    for j in jrange(n - 2):
                        i = n - j - 2
                        cx(gidney_anc[i - 1], gidney_anc[i])

                        bit_i = _extract_bit(a_int, i, a_int_is_bigint)
                        with control(bit_i):
                            x(gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")

                        with control(bit_i):
                            x(gidney_anc[i - 1])
            else:
                if ctrl is not None:
                    for j in jrange(n - 2):
                        i = n - j - 2
                        cx(gidney_anc[i - 1], gidney_anc[i])
                        bit_i = a_bits[i]

                        if bit_i:
                            cx(ctrl, gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")

                        if bit_i:
                            cx(ctrl, gidney_anc[i - 1])
                else:
                    for j in jrange(n - 2):
                        i = n - j - 2
                        cx(gidney_anc[i - 1], gidney_anc[i])
                        bit_i = a_bits[i]

                        if bit_i:
                            x(gidney_anc[i - 1])

                        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")

                        if bit_i:
                            x(gidney_anc[i - 1])
        else:
            # Quantum addend path
            for j in jrange(n - 2):
                i = n - j - 2
                cx(gidney_anc[i - 1], gidney_anc[i])
                mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney_inv")
                if ctrl is not None:
                    mcx([ctrl, a_qbs[i]], b_qbs[i])
                    cx(gidney_anc[i - 1], a_qbs[i])
                    cx(gidney_anc[i - 1], b_qbs[i])
                else:
                    cx(gidney_anc[i - 1], a_qbs[i])
                    cx(a_qbs[i], b_qbs[i])

        # Phase 5: erase carry_0
        if not a_is_quantum:
            # Classical addend path
            if check_for_tracing_mode():
                bit_0 = _extract_bit(a_int, 0, a_int_is_bigint)
                with control(bit_0):
                    if ctrl is not None:
                        mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney_inv")
                    else:
                        cx(b_qbs[0], gidney_anc[0])
            elif a_bits[0]:
                if ctrl is not None:
                    mcx([ctrl, b_qbs[0]], gidney_anc[0], method="gidney_inv")
                else:
                    cx(b_qbs[0], gidney_anc[0])

        elif c_in_qb is not None:
            # Quantum addend path with carry-in
            cx(c_in_qb, gidney_anc[0])
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")

            if ctrl is not None:
                lsb_ctrl_anc = (
                    QuantumBool(name="gidney_lsb_ctrl_anc*")
                    if check_for_tracing_mode()
                    else QuantumBool(name="gidney_lsb_ctrl_anc*", qs=qs)
                )
                mcx([ctrl, c_in_qb], lsb_ctrl_anc[0], method="gidney")
                cx(lsb_ctrl_anc[0], b_qbs[0])
                mcx([ctrl, c_in_qb], lsb_ctrl_anc[0], method="gidney_inv")
                cx(c_in_qb, b_qbs[0])
                lsb_ctrl_anc.delete()

            cx(c_in_qb, a_qbs[0])
        else:
            # Quantum addend path without carry-in
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")

        gidney_anc.delete(verify=False)

    # ---- Final sum stage ----
    if a_is_quantum:
        # Quantum addend path
        if ctrl is not None:
            mcx([ctrl, a_qbs[0]], b_qbs[0])
        else:
            cx(a_qbs[0], b_qbs[0])

        a_ext.delete(verify=False)
        return

    # Classical addend path
    if check_for_tracing_mode():
        if ctrl is not None:
            for i in jrange(n):
                if c_in_qb is not None and i == 0:
                    continue

                bit_i = _extract_bit(a_int, i, a_int_is_bigint)
                with control(bit_i):
                    cx(ctrl, b_qbs[i])
        else:
            for i in jrange(n):
                if c_in_qb is not None and i == 0:
                    continue

                bit_i = _extract_bit(a_int, i, a_int_is_bigint)
                with control(bit_i):
                    x(b_qbs[i])
        return

    if ctrl is not None:
        for i in jrange(n):
            if c_in_qb is not None and i == 0:
                continue

            bit_i = _extract_bit(a_int, i, a_int_is_bigint)
            if bit_i:
                cx(ctrl, b_qbs[i])
    else:
        for i in jrange(n):
            if c_in_qb is not None and i == 0:
                continue

            bit_i = a_bits[i]
            if bit_i:
                x(b_qbs[i])
