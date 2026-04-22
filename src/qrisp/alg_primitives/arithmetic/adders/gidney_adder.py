import jax.numpy as jnp
import numpy as np

from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control, conjugate
from qrisp.misc import int_encoder
from qrisp.misc.utility import bin_rep
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode
from qrisp import t_depth_indicator


# ======================================================================
# Mode-aware classical-bit conditional execution.
# ======================================================================

def _exec_if_bit(a, i, tracing, a_is_biginteger, func):
    """Execute *func()* only when bit *i* of classical *a* is set.

    In tracing mode this uses a ``control()`` environment.
    In static mode this uses a plain Python ``if`` — no QuantumEnvironment
    wrapper, so the compiler sees a flat gate list.
    """
    if tracing:
        if a_is_biginteger:
            cond_val = a.get_bit(i)
        else:
            cond_val = jnp.bool((a >> i) & 1)
        with control(cond_val):
            func()
    else:
        if a_is_biginteger:
            raise RuntimeError("BigInteger not supported in static mode")
        if i < len(a) and a[i] == "1":
            func()


def _c_inv(c):
    """Invert a single-character bit: '0' <-> '1'."""
    return "0" if c == "1" else "1"


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on https://arxiv.org/abs/1709.06648. Works in both standard
    Python and jasp tracing modes. A single circuit body handles both
    the quantum-quantum and classical-quantum cases via an
    ``a_is_quantum`` flag; when ``a`` is classical no QuantumVariable
    is allocated for it. Classical ``a`` may be an ``int``, a
    little-endian bit-string, or a ``BigInteger``.
    """

    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
        BigInteger,
    )

    # ------------------------------------------------------------------
    # Classify a
    # ------------------------------------------------------------------
    a_is_biginteger = isinstance(a, BigInteger)

    if isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        a_is_quantum = True
    else:
        a_is_quantum = False

    tracing = check_for_tracing_mode()

    # ------------------------------------------------------------------
    # Normalize classical a
    # ------------------------------------------------------------------
    if not a_is_quantum:
        if tracing:
            if isinstance(a, str):
                a = int(a[::-1], 2) if a else 0
            if isinstance(a, int):
                a = jnp.array(a, dtype=jnp.int64)
        else:
            if a_is_biginteger:
                pass
            elif isinstance(a, (int, np.int_)):
                a = int(a)
                a = bin_rep(a % 2 ** len(b), len(b))[::-1]
            elif isinstance(a, str):
                pass
            else:
                raise Exception(
                    f"Tried to call gidney_adder with invalid classical type {type(a)}"
                )

    qs = None if tracing else b[0].qs()

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
    c_in_qb = None
    if c_in is not None:
        if a_is_quantum:
            c_in_qb = c_in[0] if isinstance(c_in, QuantumBool) else c_in
        else:
            c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
            b_eff = c_in_qbs + b_eff
            if tracing:
                if a_is_biginteger:
                    a = (a << 1) | BigInteger.create(1, a.size)
                else:
                    a = (a << 1) | jnp.int64(1)
            else:
                if a_is_biginteger:
                    a = (a << 1) | BigInteger.create(1, a.size)
                else:
                    a = "1" + a

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
        else:
            if not tracing and not a_is_biginteger:
                a = a + "0"

    n = jlen(b_eff) if tracing else len(b_eff)

    # ------------------------------------------------------------------
    # Allocate control ancilla for QQ controlled path.
    # ------------------------------------------------------------------
    ctrl_anc = None
    if a_is_quantum and ctrl is not None:
        ctrl_anc = (QuantumBool(name="gidney_control_anc*", qs=qs)
                    if qs is not None
                    else QuantumBool(name="gidney_control_anc*"))

    # ------------------------------------------------------------------
    # n == 1: trivial case
    # ------------------------------------------------------------------
    _range = jrange if tracing else range

    if tracing:
        with control(n == 1):
            _n1_body(a, b_eff, a_is_quantum, a_is_biginteger, a_qubits,
                     ctrl, ctrl_anc, tracing)
    else:
        if n == 1:
            _n1_body(a, b_eff, a_is_quantum, a_is_biginteger, a_qubits,
                     ctrl, ctrl_anc, tracing)
            if ctrl_anc is not None:
                ctrl_anc.delete()
            if zero_anc is not None:
                zero_anc.delete()
            if a_ext is not None:
                a_ext.delete()
            return

    # ------------------------------------------------------------------
    # Main V circuit (n >= 2)
    # ------------------------------------------------------------------
    if tracing:
        _ctrl_n_gt_1 = control(n > 1)
    else:
        _ctrl_n_gt_1 = _noop()

    with _ctrl_n_gt_1:
        gidney_anc = (QuantumVariable(n - 1, name="gidney_anc*", qs=qs)
                      if qs is not None
                      else QuantumVariable(n - 1, name="gidney_anc*"))

        # --- Initial Toffoli ---
        if a_is_quantum:
            if c_in_qb is not None:
                cx(c_in_qb, b_eff[0])
                cx(c_in_qb, a_qubits[0])
            mcx([a_qubits[0], b_eff[0]], gidney_anc[0], method="gidney")
            if c_in_qb is not None:
                cx(c_in_qb, gidney_anc[0])
        else:
            def _init_tof():
                if ctrl is None:
                    cx(b_eff[0], gidney_anc[0])
                else:
                    mcx([ctrl, b_eff[0]], gidney_anc[0], method="gidney")
            _exec_if_bit(a, 0, tracing, a_is_biginteger, _init_tof)

        # --- Left V ---
        for j in _range(n - 2):
            i = j + 1
            if a_is_quantum:
                cx(gidney_anc[i - 1], a_qubits[i])
                cx(gidney_anc[i - 1], b_eff[i])
                mcx([a_qubits[i], b_eff[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])
            else:
                cx(gidney_anc[i - 1], b_eff[i])
                if tracing:
                    # Tracing mode: use x/cx flip + standard mcx
                    def _left_v_flip(i=i):
                        if ctrl is None:
                            x(gidney_anc[i - 1])
                        else:
                            cx(ctrl, gidney_anc[i - 1])
                    _exec_if_bit(a, i, tracing, a_is_biginteger, _left_v_flip)
                    mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                        method="gidney")
                    _exec_if_bit(a, i, tracing, a_is_biginteger, _left_v_flip)
                else:
                    # Static mode: use ctrl_state for optimal T-depth
                    if ctrl is not None:
                        if a[i] == "1":
                            cx(ctrl, gidney_anc[i - 1])
                        mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                            method="gidney", ctrl_state="11")
                        if a[i] == "1":
                            cx(ctrl, gidney_anc[i - 1])
                    else:
                        mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                            method="gidney", ctrl_state=_c_inv(a[i]) + "1")
                cx(gidney_anc[i - 1], gidney_anc[i])

        # --- Tip of V ---
        if a_is_quantum:
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], b_eff[n - 1])
                mcx([ctrl, a_qubits[n - 1]], ctrl_anc[0], method="gidney")
                cx(ctrl_anc[0], b_eff[n - 1])
                mcx([ctrl, a_qubits[n - 1]], ctrl_anc[0], method="gidney_inv")
            else:
                cx(gidney_anc[n - 2], b_eff[n - 1])
                cx(a_qubits[n - 1], b_eff[n - 1])
        else:
            # CQ tip: only carry propagation. a[n-1] XOR is in the sweep.
            if ctrl is not None:
                mcx([ctrl, gidney_anc[n - 2]], b_eff[n - 1])
            else:
                cx(gidney_anc[n - 2], b_eff[n - 1])

        # --- Right V ---
        for j in _range(n - 2):
            i = n - j - 2
            if a_is_quantum:
                cx(gidney_anc[i - 1], gidney_anc[i])
                mcx([a_qubits[i], b_eff[i]], gidney_anc[i], method="gidney_inv")
                if ctrl is not None:
                    mcx([ctrl, a_qubits[i]], ctrl_anc[0], method="gidney")
                    cx(ctrl_anc[0], b_eff[i])
                    mcx([ctrl, a_qubits[i]], ctrl_anc[0], method="gidney_inv")
                    cx(gidney_anc[i - 1], a_qubits[i])
                    cx(gidney_anc[i - 1], b_eff[i])
                else:
                    cx(gidney_anc[i - 1], a_qubits[i])
                    cx(a_qubits[i], b_eff[i])
            else:
                cx(gidney_anc[i - 1], gidney_anc[i])
                if tracing:
                    # Tracing mode: use x/cx flip + standard mcx
                    def _right_v_flip(i=i):
                        if ctrl is None:
                            x(gidney_anc[i - 1])
                        else:
                            cx(ctrl, gidney_anc[i - 1])
                    _exec_if_bit(a, i, tracing, a_is_biginteger, _right_v_flip)
                    mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                        method="gidney_inv")
                    _exec_if_bit(a, i, tracing, a_is_biginteger, _right_v_flip)
                else:
                    # Static mode: use ctrl_state for optimal T-depth
                    if ctrl is not None:
                        if a[i] == "1":
                            cx(ctrl, gidney_anc[i - 1])
                        mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                            method="gidney", ctrl_state="11")
                        if a[i] == "1":
                            cx(ctrl, gidney_anc[i - 1])
                    else:
                        mcx([gidney_anc[i - 1], b_eff[i]], gidney_anc[i],
                            method="gidney", ctrl_state=_c_inv(a[i]) + "1")

        # --- Final Toffoli ---
        if a_is_quantum:
            if c_in_qb is not None:
                cx(c_in_qb, gidney_anc[0])
            mcx([a_qubits[0], b_eff[0]], gidney_anc[0], method="gidney_inv")
            if c_in_qb is not None:
                cx(c_in_qb, a_qubits[0])
        else:
            def _final_tof():
                if ctrl is None:
                    cx(b_eff[0], gidney_anc[0])
                else:
                    mcx([ctrl, b_eff[0]], gidney_anc[0], method="gidney_inv")
            _exec_if_bit(a, 0, tracing, a_is_biginteger, _final_tof)

        gidney_anc.delete(verify=False)

    # ------------------------------------------------------------------
    # Final XOR sweep
    # ------------------------------------------------------------------
    if a_is_quantum:
        if ctrl is not None:
            mcx([ctrl, a_qubits[0]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b_eff[0])
            mcx([ctrl, a_qubits[0]], ctrl_anc[0], method="gidney_inv")
        else:
            cx(a_qubits[0], b_eff[0])
    else:
        # CQ sweep: ALL n bits (including MSB).
        # Skip bit 0 when c_in is present.
        for i in _range(n):
            def _sweep_step(i=i):
                if ctrl is not None:
                    cx(ctrl, b_eff[i])
                else:
                    x(b_eff[i])

            if c_in is not None:
                if tracing:
                    with control(i > 0):
                        _exec_if_bit(a, i, tracing, a_is_biginteger, _sweep_step)
                else:
                    if i > 0:
                        _exec_if_bit(a, i, tracing, a_is_biginteger, _sweep_step)
            else:
                _exec_if_bit(a, i, tracing, a_is_biginteger, _sweep_step)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if ctrl_anc is not None:
        ctrl_anc.delete()
    if zero_anc is not None:
        zero_anc.delete()
    if a_ext is not None:
        a_ext.delete()


# ======================================================================
# Small helpers
# ======================================================================

class _noop:
    """No-op context manager for static mode where control(n>1) is not needed."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _n1_body(a, b_eff, a_is_quantum, a_is_biginteger, a_qubits,
             ctrl, ctrl_anc, tracing):
    """b[0] ^= a[0] — the trivial single-bit adder."""
    if a_is_quantum:
        if ctrl is not None:
            mcx([ctrl, a_qubits[0]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b_eff[0])
            mcx([ctrl, a_qubits[0]], ctrl_anc[0], method="gidney_inv")
        else:
            cx(a_qubits[0], b_eff[0])
    else:
        def _do_n1():
            if ctrl is not None:
                cx(ctrl, b_eff[0])
            else:
                x(b_eff[0])
        _exec_if_bit(a, 0, tracing, a_is_biginteger, _do_n1)