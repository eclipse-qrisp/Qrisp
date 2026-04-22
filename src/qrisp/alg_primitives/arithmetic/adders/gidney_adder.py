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
    Python and jasp tracing modes. A single circuit body handles both
    the quantum-quantum and classical-quantum cases via an
    ``a_is_quantum`` flag; when ``a`` is classical no QuantumVariable
    is allocated for it. Classical ``a`` must be an ``int``.
    """

    # ------------------------------------------------------------------
    # Classify a as quantum or classical and normalize the classical
    # form into a jnp int.
    # ------------------------------------------------------------------
    if isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        a_is_quantum = True
    else:
        a_is_quantum = False
        if isinstance(a, int):
            a = jnp.array(a, dtype=jnp.int64)

    qs = None if check_for_tracing_mode() else b[0].qs()

    # ------------------------------------------------------------------
    # Size normalization (quantum-a only).
    # ------------------------------------------------------------------
    a_ext = None
    if a_is_quantum:
        dim_a = jlen(a)
        dim_b = jlen(b)
        effective_size_a = jnp.minimum(dim_a, dim_b)
        a = a[:effective_size_a]
        extension_size = jnp.maximum(0, dim_b - dim_a)
        a_ext = (QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs)
                 if qs is not None
                 else QuantumVariable(extension_size, name="gidney_a_ext*"))
        a_qubits = a[:] + a_ext[:]
    else:
        a_qubits = None

    b_eff = b[:]

    # ------------------------------------------------------------------
    # c_in plumbing.
    # ------------------------------------------------------------------
    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        b_eff = c_in_qbs + b_eff
        if a_is_quantum:
            one_anc = (QuantumBool(name="gidney_cin_one*", qs=qs)
                       if qs is not None
                       else QuantumBool(name="gidney_cin_one*"))
            x(one_anc[0])
            a_qubits = one_anc[:] + a_qubits
        else:
            a = (a << 1) | jnp.int64(1)

    # ------------------------------------------------------------------
    # c_out plumbing.
    # ------------------------------------------------------------------
    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        b_eff = b_eff + c_out_qbs
        if a_is_quantum:
            zero_anc = (QuantumBool(name="gidney_cout_zero*", qs=qs)
                        if qs is not None
                        else QuantumBool(name="gidney_cout_zero*"))
            a_qubits = a_qubits + zero_anc[:]
        # Classical case: reads past the MSB of a naturally return 0.

    n = jlen(b_eff)

    # ------------------------------------------------------------------
    # Main V circuit. Structurally identical in both cases; the only
    # differences are:
    #   - q-q: a[i] is a qubit; Toffolis use it as a direct control,
    #     and the left-V does cx(g_{i-1}, a[i]) to mutate it.
    #   - c-q: a[i] is a classical bit; the "mutation" cx(g_{i-1}, a[i])
    #     is skipped, and the Toffoli's a-control is realized by
    #     conjugating with X on g_{i-1} when the classical bit is 1
    #     (which reproduces the effective (a[i] XOR g_{i-1}) control
    #     that the q-q path gets from mutating a[i]).
    # ------------------------------------------------------------------
    with control(n > 1):
        gidney_anc = (QuantumVariable(n - 1, name="gidney_anc*", qs=qs)
                      if qs is not None
                      else QuantumVariable(n - 1, name="gidney_anc*"))

        # --- Initial Toffoli ---------------------------------------
        if a_is_quantum:
            mcx([a_qubits[0], b_eff[0]], gidney_anc[0], method="gidney")
        else:
            with control(jnp.bool((a >> 0) & 1)):
                if ctrl is None:
                    cx(b_eff[0], gidney_anc[0])
                else:
                    mcx([ctrl, b_eff[0]], gidney_anc[0], method="gidney")

        # --- Left V -------------------------------------------------
        for j in jrange(n - 2):
            i = j + 1
            if a_is_quantum:
                cx(gidney_anc[i - 1], a_qubits[i])
                cx(gidney_anc[i - 1], b_eff[i])
                mcx([a_qubits[i], b_eff[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])
            else:
                cx(gidney_anc[i - 1], b_eff[i])
                with control(jnp.bool((a >> i) & 1)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])
                mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                    method="gidney")
                with control(jnp.bool((a >> i) & 1)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])
                cx(gidney_anc[i - 1], gidney_anc[i])

        # --- Tip of V ----------------------------------------------
        if a_is_quantum:
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], b_eff[n - 1])
                mcx([ctrl, a_qubits[n - 1]], b_eff[n - 1])
            else:
                cx(gidney_anc[n - 2], b_eff[n - 1])
                cx(a_qubits[n - 1], b_eff[n - 1])
        else:
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], b_eff[n - 1])
                with control(jnp.bool((a >> (n - 1)) & 1)):
                    cx(ctrl, b_eff[n - 1])
            else:
                cx(gidney_anc[n - 2], b_eff[n - 1])
                with control(jnp.bool((a >> (n - 1)) & 1)):
                    x(b_eff[n - 1])

        # --- Right V -----------------------------------------------
        for j in jrange(n - 2):
            i = n - j - 2
            if a_is_quantum:
                cx(gidney_anc[i - 1], gidney_anc[i])
                mcx([a_qubits[i], b_eff[i]], gidney_anc[i], method="gidney_inv")
                if ctrl is not None:
                    mcx([ctrl, a_qubits[i]], b_eff[i])
                    cx(gidney_anc[i - 1], a_qubits[i])
                    cx(gidney_anc[i - 1], b_eff[i])
                else:
                    cx(gidney_anc[i - 1], a_qubits[i])
                    cx(a_qubits[i], b_eff[i])
            else:
                cx(gidney_anc[i - 1], gidney_anc[i])
                with control(jnp.bool((a >> i) & 1)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])
                mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                    method="gidney_inv")
                with control(jnp.bool((a >> i) & 1)):
                    if ctrl is None:
                        x(gidney_anc[i - 1])
                    else:
                        cx(ctrl, gidney_anc[i - 1])

        # --- Final Toffoli -----------------------------------------
        if a_is_quantum:
            mcx([a_qubits[0], b_eff[0]], gidney_anc[0], method="gidney_inv")
        else:
            with control(jnp.bool((a >> 0) & 1)):
                if ctrl is None:
                    cx(b_eff[0], gidney_anc[0])
                else:
                    mcx([ctrl, b_eff[0]], gidney_anc[0], method="gidney_inv")

        gidney_anc.delete()

    # ------------------------------------------------------------------
    # Final top-right XOR sweep. In q-q this is a single CX at bit 0
    # (the rest of b got its a-contributions from the right-V). In c-q
    # the right-V does not emit b[i] ^= a[i] inline, so we emit the
    # full sweep here, skipping bit 0 when c_in is set (to avoid
    # flipping the c_in qubit).
    # ------------------------------------------------------------------
    if a_is_quantum:
        if c_in is None:
            if ctrl is not None:
                mcx([ctrl, a_qubits[0]], b_eff[0])
            else:
                cx(a_qubits[0], b_eff[0])
    else:
        a_final = (a & ~jnp.int64(1)) if c_in is not None else a
        for i in jrange(n):
            with control(jnp.bool((a_final >> i) & 1)):
                if ctrl is not None:
                    cx(ctrl, b_eff[i])
                else:
                    x(b_eff[i])

    # ------------------------------------------------------------------
    # Cleanup.
    # ------------------------------------------------------------------
    if one_anc is not None:
        x(one_anc[0])
        one_anc.delete()
    if zero_anc is not None:
        zero_anc.delete()
    if a_ext is not None:
        a_ext.delete()