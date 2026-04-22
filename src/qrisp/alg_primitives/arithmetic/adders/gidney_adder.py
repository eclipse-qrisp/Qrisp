import jax.numpy as jnp
import numpy as np

from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def _extract_boolean_digit(integer, digit):
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
    if isinstance(integer, BigInteger):
        return jnp.bool(integer.get_bit(digit))
    return jnp.bool((integer >> digit & 1))


def _session_of(b):
    """Return b's QuantumSession in static mode, or None under tracing."""
    if check_for_tracing_mode():
        return None
    first = b[0]
    return first.qs()


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on https://arxiv.org/abs/1709.06648. Works in both standard
    Python and jasp tracing modes.  Supports both quantum and classical
    values of ``a`` in a single unified code path.
    """

    is_classical_a = not isinstance(a, (QuantumVariable, DynamicQubitArray, list))

    # ------------------------------------------------------------------
    # Pre-processing: normalise *a* and set up *b*, *n*
    # ------------------------------------------------------------------
    a_ext = None
    one_anc = None
    zero_anc = None

    if is_classical_a:
        # Classical a: convert to a JAX-traceable integer
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        if not check_for_tracing_mode() and isinstance(a, (int, np.integer)):
            a = int(a) % (1 << len(b))

        if isinstance(a, int):
            a = jnp.array(a, dtype=jnp.int64)

        b = b[:]
        n = jlen(b)

        # c_in: prepend a classical 1-bit to a and the carry-in qubit to b
        if c_in is not None:
            c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
            a = (a << 1) | 1
            b = c_in_qbs + b
            n = n + 1

        # c_out: a's high bit is implicitly 0; append carry-out qubit to b
        if c_out is not None:
            c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
            b = b + c_out_qbs
            n = n + 1

    else:
        # Quantum a: match dimensions, handle c_in / c_out with ancillae
        qs = _session_of(b)

        def _qv(size, name):
            if qs is not None:
                return QuantumVariable(size, name=name, qs=qs)
            return QuantumVariable(size, name=name)

        def _qbool(name):
            if qs is not None:
                return QuantumBool(name=name, qs=qs)
            return QuantumBool(name=name)

        dim_a = jlen(a)
        dim_b = jlen(b)

        effective_size_a = jnp.minimum(dim_a, dim_b)
        a = a[:effective_size_a]

        extension_size = jnp.maximum(0, dim_b - dim_a)
        a_ext = _qv(extension_size, "gidney_a_ext*")

        a = a[:] + a_ext[:]
        b = b[:]

        if c_in is not None:
            c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
            one_anc = _qbool("gidney_cin_one*")
            x(one_anc[0])
            a = one_anc[:] + a
            b = c_in_qbs + b

        if c_out is not None:
            c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
            zero_anc = _qbool("gidney_cout_zero*")
            a = a + zero_anc[:]
            b = b + c_out_qbs

        n = jlen(a)

    # Helper to allocate ancillae (works for both paths)
    qs_b = _session_of(b)

    def _alloc(size, name):
        if qs_b is not None:
            return QuantumVariable(size, name=name, qs=qs_b)
        return QuantumVariable(size, name=name)

    # ------------------------------------------------------------------
    # V-shape core
    # ------------------------------------------------------------------
    with control(n > 1):
        gidney_anc = _alloc(n - 1, "gidney_anc*")

        # --- Initial Toffoli ---
        if is_classical_a:
            with control(_extract_boolean_digit(a, 0)):
                if ctrl is None:
                    cx(b[0], gidney_anc[0])
                else:
                    mcx([ctrl, b[0]], gidney_anc[0], method="gidney")
        else:
            mcx([a[0], b[0]], gidney_anc[0], method="gidney")

        # --- Left sweep ---
        for j in jrange(n - 2):
            i = j + 1

            if is_classical_a:
                cx(gidney_anc[i - 1], b[i])

                with control(_extract_boolean_digit(a, i)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])

                mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney")

                with control(_extract_boolean_digit(a, i)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])

                cx(gidney_anc[i - 1], gidney_anc[i])
            else:
                cx(gidney_anc[i - 1], a[i])
                cx(gidney_anc[i - 1], b[i])
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])

        # --- Tip of the V ---
        if is_classical_a:
            cx(gidney_anc[n - 2], b[n - 1])
        else:
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], b[n - 1])
                mcx([ctrl, a[n - 1]], b[n - 1])
            else:
                cx(gidney_anc[n - 2], b[n - 1])
                cx(a[n - 1], b[n - 1])

        # --- Right sweep ---
        for j in jrange(n - 2):
            i = n - j - 2

            if is_classical_a:
                cx(gidney_anc[i - 1], gidney_anc[i])

                if ctrl is not None:
                    with control(_extract_boolean_digit(a, i)):
                        cx(ctrl, gidney_anc[i - 1])
                    mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney_inv")
                    with control(_extract_boolean_digit(a, i)):
                        cx(ctrl, gidney_anc[i - 1])
                else:
                    with control(_extract_boolean_digit(a, i)):
                        x(gidney_anc[i - 1])
                    mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney_inv")
                    with control(_extract_boolean_digit(a, i)):
                        x(gidney_anc[i - 1])
            else:
                cx(gidney_anc[i - 1], gidney_anc[i])
                mcx([a[i], b[i]], gidney_anc[i], method="gidney_inv")

                if ctrl is not None:
                    mcx([ctrl, a[i]], b[i])
                    cx(gidney_anc[i - 1], a[i])
                    cx(gidney_anc[i - 1], b[i])
                else:
                    cx(gidney_anc[i - 1], a[i])
                    cx(a[i], b[i])

        # --- Final Toffoli ---
        if is_classical_a:
            with control(_extract_boolean_digit(a, 0)):
                if ctrl is None:
                    cx(b[0], gidney_anc[0])
                else:
                    mcx([ctrl, b[0]], gidney_anc[0], method="gidney_inv")
        else:
            mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")

        gidney_anc.delete()

    # ------------------------------------------------------------------
    # Post-V XOR column
    # ------------------------------------------------------------------
    if is_classical_a:
        xor_start = 1 if c_in is not None else 0
        for i in jrange(xor_start, n):
            with control(_extract_boolean_digit(a, i)):
                if ctrl is not None:
                    cx(ctrl, b[i])
                else:
                    x(b[i])
    else:
        if c_in is None:
            if ctrl is not None:
                mcx([ctrl, a[0]], b[0])
            else:
                cx(a[0], b[0])

    # ------------------------------------------------------------------
    # Cleanup (quantum-a ancillae only)
    # ------------------------------------------------------------------
    if one_anc is not None:
        x(one_anc[0])
        one_anc.delete()

    if zero_anc is not None:
        zero_anc.delete()

    if a_ext is not None:
        a_ext.delete()