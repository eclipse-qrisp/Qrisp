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

    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        if isinstance(a, (int, np.integer)):
            a = int(a) % (1 << jlen(b))

            if check_for_tracing_mode():
                b = list(b)
                if isinstance(a, int):
                    a = jnp.array(a, dtype=jnp.int64)

                if c_in is not None:
                    if isinstance(c_in, QuantumBool):
                        c_in = c_in[0]
                    b = [c_in] + b
                    a = (a << 1) + 1

                if c_out is not None:
                    if isinstance(c_out, QuantumBool):
                        c_out = c_out[0]
                    b = b + [c_out]

                n = len(b)
                with control(n > 1):
                    gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")

                    with control(extract_boolean_digit(a, 0)):
                        if ctrl is None:
                            cx(b[0], gidney_anc[0])
                        else:
                            mcx([ctrl, b[0]], gidney_anc[0], method="gidney")

                    for j in jrange(n - 2):
                        i = j + 1
                        cx(gidney_anc[i - 1], b[i])
                        with control(extract_boolean_digit(a, i)):
                            if ctrl is None:
                                x(gidney_anc[i - 1])
                            else:
                                cx(ctrl, gidney_anc[i - 1])
                        mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney")
                        with control(extract_boolean_digit(a, i)):
                            if ctrl is None:
                                x(gidney_anc[i - 1])
                            else:
                                cx(ctrl, gidney_anc[i - 1])
                        cx(gidney_anc[i - 1], gidney_anc[i])

                    cx(gidney_anc[n - 2], b[n - 1])

                    for j in jrange(n - 2):
                        i = n - j - 2
                        cx(gidney_anc[i - 1], gidney_anc[i])
                        if ctrl is not None:
                            with control(extract_boolean_digit(a, i)):
                                cx(ctrl, gidney_anc[i - 1])
                            mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney_inv")
                            with control(extract_boolean_digit(a, i)):
                                cx(ctrl, gidney_anc[i - 1])
                        else:
                            with control(extract_boolean_digit(a, i)):
                                x(gidney_anc[i - 1])
                            mcx([gidney_anc[i - 1], b[i]], gidney_anc[i], method="gidney_inv")
                            with control(extract_boolean_digit(a, i)):
                                x(gidney_anc[i - 1])

                    with control(extract_boolean_digit(a, 0)):
                        if ctrl is None:
                            cx(b[0], gidney_anc[0])
                        else:
                            mcx([ctrl, b[0]], gidney_anc[0], method="gidney_inv")

                    gidney_anc.delete()

                for i in jrange(n):
                    if c_in is not None and i == 0:
                        continue
                    with control(extract_boolean_digit(a, i)):
                        if ctrl is not None:
                            cx(ctrl, b[i])
                        else:
                            x(b[i])
                return

            qs_list = [b[0].qs()]
            for qb in [c_in, c_out, ctrl]:
                if qb is not None:
                    qs_list.append(qb.qs())
            merge(qs_list)

            with fast_append(3):
                a = int(a) % (1 << len(b))
                a = format(a, f"0{len(b)}b")[::-1]

                if c_out is not None:
                    if isinstance(c_out, QuantumBool):
                        c_out = c_out[0]
                    b = list(b) + [c_out]
                    a = a + "0"

                if c_in is not None:
                    if isinstance(c_in, QuantumBool):
                        c_in = c_in[0]
                    b = [c_in] + list(b)
                    a = "1" + a

                if len(b) == 1:
                    if a[0] == "1":
                        if ctrl is None:
                            x(b[0])
                        else:
                            cx(ctrl, b[0])
                    return

                gidney_anc = QuantumVariable(len(b) - 1, name="gidney_anc*", qs=b[0].qs())

                for i in range(len(b) - 1):
                    if i != 0:
                        if ctrl is not None:
                            if a[i] == "1":
                                cx(ctrl, gidney_anc[i - 1])
                            mcx(
                                [gidney_anc[i - 1], b[i]],
                                gidney_anc[i],
                                method="gidney",
                                ctrl_state="11",
                            )
                            if a[i] == "1":
                                cx(ctrl, gidney_anc[i - 1])
                        else:
                            mcx(
                                [gidney_anc[i - 1], b[i]],
                                gidney_anc[i],
                                method="gidney",
                                ctrl_state=c_inv(a[i]) + "1",
                            )
                        cx(gidney_anc[i - 1], gidney_anc[i])
                    else:
                        if a[i] == "1":
                            if ctrl is None:
                                cx(b[i], gidney_anc[i])
                            else:
                                mcx([ctrl, b[i]], gidney_anc[i], method="gidney")

                    if i != len(b) - 2:
                        cx(gidney_anc[i], b[i + 1])

                cx(gidney_anc[-1], b[-1])

                with invert():
                    merge(b[0].qs())

                    for i in range(len(b) - 1):
                        if i != 0:
                            if ctrl is not None:
                                if a[i] == "1":
                                    cx(ctrl, gidney_anc[i - 1])
                                mcx(
                                    [gidney_anc[i - 1], b[i]],
                                    gidney_anc[i],
                                    method="gidney",
                                    ctrl_state="11",
                                )
                                if a[i] == "1":
                                    cx(ctrl, gidney_anc[i - 1])
                            else:
                                mcx(
                                    [gidney_anc[i - 1], b[i]],
                                    gidney_anc[i],
                                    method="gidney",
                                    ctrl_state=c_inv(a[i]) + "1",
                                )
                            cx(gidney_anc[i - 1], gidney_anc[i])
                        else:
                            if a[i] == "1":
                                if ctrl is None:
                                    cx(b[i], gidney_anc[i])
                                else:
                                    mcx([ctrl, b[i]], gidney_anc[i], method="gidney")

                for i in range(len(a)):
                    if a[i] == "1":
                        if c_in is not None and i == 0:
                            continue
                        if ctrl is None:
                            x(b[i])
                        else:
                            cx(ctrl, b[i])

                gidney_anc.delete(verify=False)
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
