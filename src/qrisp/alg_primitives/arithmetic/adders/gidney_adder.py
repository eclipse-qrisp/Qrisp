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

    Based on https://arxiv.org/abs/1709.06648. Works in both standard
    Python and jasp tracing modes.
    """

    # ------------------------------------------------------------------
    # Classical a: encode into a temp register and recurse into the
    # quantum-quantum path.
    # ------------------------------------------------------------------
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):

        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        # Pin q_a to b's QuantumSession in static mode so the recursive
        # call's ancillas live in the scope the caller is tracking.
        if not check_for_tracing_mode():
            qs = b[0].qs()
            q_a = QuantumVariable(jlen(b), qs=qs)
        else:
            q_a = QuantumVariable(jlen(b))

        with conjugate(int_encoder)(q_a, a):
            gidney_adder(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return

    # ------------------------------------------------------------------
    # Resolve b's session once. Every subsequent ancilla allocation
    # passes this as qs= in static mode so all temporaries land on the
    # session the caller is tracking (important for callers like
    # cq_qcla that wrap us in a QuantumEnvironment).
    # ------------------------------------------------------------------
    qs = None if check_for_tracing_mode() else b[0].qs()

    # ------------------------------------------------------------------
    # Size normalization: truncate a if longer than b, pad with zero
    # ancillas if a is shorter. After this block, a and b share width.
    # ------------------------------------------------------------------
    dim_a = jlen(a)
    dim_b = jlen(b)

    # Truncate a to b's width when a is larger (modular addition).
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = (QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs)
             if qs is not None
             else QuantumVariable(extension_size, name="gidney_a_ext*"))

    a = a[:] + a_ext[:]
    b = b[:]

    # ------------------------------------------------------------------
    # c_in: prepend with a fresh |1> ancilla on the a side.
    # ------------------------------------------------------------------
    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        one_anc = (QuantumBool(name="gidney_cin_one*", qs=qs)
                   if qs is not None
                   else QuantumBool(name="gidney_cin_one*"))
        x(one_anc[0])
        a = one_anc[:] + a
        b = c_in_qbs + b

    # ------------------------------------------------------------------
    # c_out: append with a fresh |0> ancilla on the a side.
    # ------------------------------------------------------------------
    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        zero_anc = (QuantumBool(name="gidney_cout_zero*", qs=qs)
                    if qs is not None
                    else QuantumBool(name="gidney_cout_zero*"))
        a = a + zero_anc[:]
        b = b + c_out_qbs

    n = jlen(a)

    # ------------------------------------------------------------------
    # Main V-shaped circuit. Skipped when n == 1; that single-bit case
    # is handled by the top-right CX below.
    # ------------------------------------------------------------------
    with control(n > 1):
        gidney_anc = (QuantumVariable(n - 1, name="gidney_anc*", qs=qs)
                      if qs is not None
                      else QuantumVariable(n - 1, name="gidney_anc*"))

        mcx([a[0], b[0]], gidney_anc[0], method="gidney")

        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], a[i])
            cx(gidney_anc[i - 1], b[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])

        if ctrl is not None:
            mcx([ctrl, gidney_anc[n - 2]], b[n - 1])
            mcx([ctrl, a[n - 1]], b[n - 1])
        else:
            cx(gidney_anc[n - 2], b[n - 1])
            cx(a[n - 1], b[n - 1])

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

        mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")

        gidney_anc.delete()

    # ------------------------------------------------------------------
    # Top-right CX at position 0. Skipped when c_in is set so we don't
    # flip the c_in qubit (a[0] is the |1> one_anc in that case).
    # ------------------------------------------------------------------
    if c_in is None:
        if ctrl is not None:
            mcx([ctrl, a[0]], b[0])
        else:
            cx(a[0], b[0])

    if one_anc is not None:
        x(one_anc[0])
        one_anc.delete()

    if zero_anc is not None:
        zero_anc.delete()

    a_ext.delete()