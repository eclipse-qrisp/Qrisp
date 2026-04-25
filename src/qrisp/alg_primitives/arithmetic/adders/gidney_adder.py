import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx, merge
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control, invert
from qrisp.circuit import fast_append
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.
    Based on https://arxiv.org/abs/1709.06648. Works in both standard
    Python and jasp tracing modes.
    """
    def c_inv(c):
        if c == "0":
            return "1"
        if c == "1":
            return "0"
        raise Exception(f"Invalid control state: {c}")

    def extract_boolean_digit(integer, digit):
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

        if isinstance(integer, BigInteger):
            return jnp.bool_(integer.get_bit(digit))
        return jnp.bool_(integer >> digit & 1)

    def normalize_qubit(qb):
        if isinstance(qb, QuantumBool):
            return qb[0]
        return qb

    def semi_classical_gidney_core(a_int, b_qbs, c_in_qb=None, c_out_qb=None, ctrl_qb=None, tracing=False):
        b_qbs = list(b_qbs)

        def apply_if_bit_set(bit_index, op):
            if tracing:
                with control(extract_boolean_digit(a_int, bit_index)):
                    op()
            elif (int(a_int) >> bit_index) & 1:
                op()

        if c_in_qb is not None:
            c_in_qb = normalize_qubit(c_in_qb)
            b_qbs = [c_in_qb] + b_qbs
            a_int = (a_int << 1) + 1

        if c_out_qb is not None:
            c_out_qb = normalize_qubit(c_out_qb)
            b_qbs = b_qbs + [c_out_qb]

        n = len(b_qbs)

        if n == 1:
            def single_bit_update():
                if ctrl_qb is None:
                    x(b_qbs[0])
                else:
                    cx(ctrl_qb, b_qbs[0])

            apply_if_bit_set(0, single_bit_update)
            return

        range_fn = jrange if tracing else range
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*") if tracing \
            else QuantumVariable(n - 1, name="gidney_anc*", qs=b_qbs[0].qs())

        def first_forward_carry():
            if ctrl_qb is None:
                cx(b_qbs[0], gidney_anc[0])
            else:
                mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney")

        apply_if_bit_set(0, first_forward_carry)

        for j in range_fn(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], b_qbs[i])

            def toggle_forward_anc():
                if ctrl_qb is None:
                    x(gidney_anc[i - 1])
                else:
                    cx(ctrl_qb, gidney_anc[i - 1])

            apply_if_bit_set(i, toggle_forward_anc)
            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")
            apply_if_bit_set(i, toggle_forward_anc)
            cx(gidney_anc[i - 1], gidney_anc[i])

        cx(gidney_anc[n - 2], b_qbs[n - 1])

        for j in range_fn(n - 2):
            i = n - j - 2
            cx(gidney_anc[i - 1], gidney_anc[i])
            if ctrl_qb is not None:

                def toggle_backward_ctrl():
                    cx(ctrl_qb, gidney_anc[i - 1])

                apply_if_bit_set(i, toggle_backward_ctrl)
                mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
                apply_if_bit_set(i, toggle_backward_ctrl)
            else:

                def toggle_backward_anc():
                    x(gidney_anc[i - 1])

                apply_if_bit_set(i, toggle_backward_anc)
                mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
                apply_if_bit_set(i, toggle_backward_anc)

        def first_backward_carry():
            if ctrl_qb is None:
                cx(b_qbs[0], gidney_anc[0])
            else:
                mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney_inv")

        apply_if_bit_set(0, first_backward_carry)

        gidney_anc.delete(verify=False)

        for i in range_fn(n):
            if c_in_qb is not None and i == 0:
                continue

            def final_sum_update():
                if ctrl_qb is not None:
                    cx(ctrl_qb, b_qbs[i])
                else:
                    x(b_qbs[i])

            apply_if_bit_set(i, final_sum_update)

    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        if isinstance(a, (int, np.integer)):
            a = int(a) % (1 << jlen(b))
            tracing_mode = check_for_tracing_mode()

            if tracing_mode:
                if isinstance(a, int):
                    a = jnp.array(a, dtype=jnp.int64)
                semi_classical_gidney_core(a, b, c_in_qb=c_in, c_out_qb=c_out, ctrl_qb=ctrl, tracing=True)
                return

            qs_list = [b[0].qs()]
            for qb in [c_in, c_out, ctrl]:
                if qb is not None:
                    qs_list.append(normalize_qubit(qb).qs())
            merge(qs_list)

            with fast_append(3):
                semi_classical_gidney_core(a, b, c_in_qb=c_in, c_out_qb=c_out, ctrl_qb=ctrl, tracing=False)
            return

    qs = None if check_for_tracing_mode() else b[0].qs()

    dim_a = jlen(a)
    dim_b = jlen(b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]
    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs) if qs is not None \
        else QuantumVariable(extension_size, name="gidney_a_ext*")
    a = a[:] + a_ext[:]
    b = b[:]

    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        one_anc = QuantumBool(name="gidney_cin_one*", qs=qs) if qs is not None \
            else QuantumBool(name="gidney_cin_one*")
        x(one_anc[0])
        a = one_anc[:] + a
        b = c_in_qbs + b

    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        zero_anc = QuantumBool(name="gidney_cout_zero*", qs=qs) if qs is not None \
            else QuantumBool(name="gidney_cout_zero*")
        a = a + zero_anc[:]
        b = b + c_out_qbs

    n = jlen(a)
    with control(n > 1):
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*", qs=qs) if qs is not None \
            else QuantumVariable(n - 1, name="gidney_anc*")
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
        gidney_anc.delete(verify=False)

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

    a_ext.delete(verify=False)
