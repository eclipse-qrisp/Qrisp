import jax.numpy as jnp
import numpy as np
from jax import jit

from qrisp.core import QuantumVariable, x, cx, mcx, merge, measure
from qrisp.qtypes import QuantumBool
from qrisp.environments import invert, control, custom_control
from qrisp.circuit import fast_append
from qrisp.misc.utility import bin_rep
from qrisp.jasp import jrange, qache, check_for_tracing_mode, DynamicQubitArray


# ============================================================================
# Semi-classical (cq) Gidney adder — static mode
# ============================================================================

def c_inv(c):
    if c == "0":
        return "1"
    elif c == "1":
        return "0"
    else:
        raise Exception


def cq_gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):

    qs_list = [b[0].qs()]

    for qb in [c_in, c_out, ctrl]:
        if qb is not None:
            qs_list.append(qb.qs())

    merge(qs_list)

    with fast_append(3):

        if isinstance(a, (int, np.int_)):
            a = int(a)
            a = bin_rep(a % 2 ** len(b), len(b))[::-1]
        elif not isinstance(a, str):
            raise Exception(
                f"Tried to call semi-classical carry calculator with invalid type {type(a)}"
            )

        if len(a) != len(b):
            raise Exception("Tried to call Gidney adder with inputs of unequal length")

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
                    if "1" == a[i]:
                        cx(ctrl, gidney_anc[i - 1])
                    mcx(
                        [gidney_anc[i - 1], b[i]],
                        gidney_anc[i],
                        method="gidney",
                        ctrl_state="11",
                    )
                    if "1" == a[i]:
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
                        if "1" == a[i]:
                            cx(ctrl, gidney_anc[i - 1])
                        mcx(
                            [gidney_anc[i - 1], b[i]],
                            gidney_anc[i],
                            method="gidney",
                            ctrl_state="11",
                        )
                        if "1" == a[i]:
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


# ============================================================================
# Quantum-quantum (qq) Gidney adder — static mode
# ============================================================================

@custom_control
def qq_gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    if len(a) != len(b):
        raise Exception("Tried to call Gidney adder with inputs of unequal length")

    if c_out is not None:
        if isinstance(c_out, QuantumBool):
            c_out = c_out[0]
        b = list(b) + [c_out]

    if len(b) == 1:
        if ctrl is not None:
            mcx([ctrl, a[0]], b)
        else:
            cx(a[0], b[0])
        if c_in is not None:
            if isinstance(c_in, QuantumBool):
                c_in = c_in[0]
            if ctrl is not None:
                cx(c_in, b[0])
            else:
                mcx([c_in, b[0]], b)
        return

    if ctrl is not None:
        gidney_control_anc = QuantumBool(name="gidney_control_anc*", qs=b[0].qs())

    gidney_anc = QuantumVariable(len(b) - 1, name="gidney_anc*", qs=b[0].qs())

    for i in range(len(b) - 1):
        if i != 0:
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])
        elif c_in is not None:
            cx(c_in, b[i])
            cx(c_in, a[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(c_in, gidney_anc[i])
        else:
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")

        if i != len(b) - 2:
            cx(gidney_anc[i], a[i + 1])
            cx(gidney_anc[i], b[i + 1])

    if ctrl is not None:
        mcx([ctrl, gidney_anc[-1]], b[-1])
        mcx([ctrl, a[-1]], gidney_control_anc[0], method="gidney")
        cx(gidney_control_anc[0], b[-1])
        mcx([ctrl, a[-1]], gidney_control_anc[0], method="gidney_inv")
    else:
        cx(gidney_anc[-1], b[-1])

    with invert():
        for i in range(len(b) - 1):
            if i != 0:
                if i != len(b) - 1:
                    cx(gidney_anc[i - 1], a[i])
                if ctrl is not None:
                    if i != len(b) - 1:
                        cx(gidney_anc[i - 1], b[i])
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney")
                    cx(gidney_control_anc[0], b[i])
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney_inv")
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])
            elif c_in is not None:
                cx(c_in, a[i])
                if ctrl is not None:
                    cx(c_in, b[i])
                    mcx([ctrl, c_in], gidney_control_anc[0], method="gidney")
                    cx(gidney_control_anc[0], b[i])
                    mcx([ctrl, c_in], gidney_control_anc[0], method="gidney_inv")
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(c_in, gidney_anc[i])
            else:
                if ctrl is not None:
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney")
                    cx(gidney_control_anc[0], b[i])
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney_inv")
                mcx([a[i], b[i]], gidney_anc[i], method="gidney")

    if ctrl is None:
        for i in range(len(a)):
            cx(a[i], b[i])
    else:
        gidney_control_anc.delete()

    gidney_anc.delete(verify=False)


# ============================================================================
# Jasp (tracing mode) semi-classical Gidney adder
# ============================================================================

@jit
def extract_boolean_digit(integer, digit):
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
    if isinstance(integer, BigInteger):
        return jnp.bool(integer.get_bit(digit))
    return jnp.bool((integer >> digit & 1))


@custom_control
def jasp_cq_gidney_adder(a, b, ctrl=None):
    if isinstance(b, list):
        n = len(b)
    else:
        n = b.size

    if isinstance(a, int):
        a = jnp.array(a, dtype=jnp.int64)

    with control(n > 1):
        i = 0
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")

        with control(extract_boolean_digit(a, i)):
            if ctrl is None:
                cx(b[i], gidney_anc[i])
            else:
                mcx([ctrl, b[i]], gidney_anc[i], method="gidney")

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
        with control(extract_boolean_digit(a, i)):
            if ctrl is not None:
                cx(ctrl, b[i])
            else:
                x(b[i])


# ============================================================================
# Jasp (tracing mode) quantum-quantum Gidney adder
# ============================================================================

@custom_control
def jasp_qq_gidney_adder(a, b, ctrl=None):
    if isinstance(b, list):
        n = min(len(a), len(b))
        perform_incrementation = n < len(b)
    else:
        n = jnp.min(jnp.array([a.size, b.size]))
        perform_incrementation = n < b.size

    if ctrl is not None:
        ctrl_anc = QuantumBool(name="gidney_anc_2*")

    with control(n > 1):
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")

        i = 0
        mcx([a[i], b[i]], gidney_anc[i], method="gidney")

        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], a[i])
            cx(gidney_anc[i - 1], b[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])

        with control(perform_incrementation):
            carry_out = QuantumBool()

            cx(gidney_anc[n - 2], a[n - 1])
            cx(gidney_anc[n - 2], b[n - 1])

            mcx([a[n - 1], b[n - 1]], carry_out[0], method="gidney")
            cx(gidney_anc[n - 2], carry_out[0])

            ctrl_list = [carry_out[0]]
            if ctrl is not None:
                ctrl_list.append(ctrl)

            with control(ctrl_list):
                jasp_cq_gidney_adder(1, b[n:])

            cx(gidney_anc[n - 2], carry_out[0])
            mcx([a[n - 1], b[n - 1]], carry_out[0], method="gidney_inv")
            carry_out.delete()

            cx(gidney_anc[n - 2], a[n - 1])
            if ctrl is not None:
                cx(gidney_anc[n - 2], b[n - 1])

        if ctrl is not None:
            mcx([ctrl, gidney_anc[n - 2]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b[n - 1])
            mcx([ctrl, gidney_anc[n - 2]], ctrl_anc[0], method="gidney_inv")

            mcx([ctrl, a[n - 1]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b[n - 1])
            mcx([ctrl, a[n - 1]], ctrl_anc[0], method="gidney_inv")
        else:
            with control(jnp.logical_not(perform_incrementation)):
                cx(gidney_anc[n - 2], b[n - 1])
            cx(a[n - 1], b[n - 1])

        for j in jrange(n - 2):
            i = n - j - 2
            cx(gidney_anc[i - 1], gidney_anc[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney_inv")

            if ctrl is not None:
                mcx([ctrl, a[i]], ctrl_anc[0], method="gidney")
                cx(ctrl_anc[0], b[i])
                mcx([ctrl, a[i]], ctrl_anc[0], method="gidney_inv")
                cx(gidney_anc[i - 1], a[i])
                cx(gidney_anc[i - 1], b[i])
            else:
                cx(gidney_anc[i - 1], a[i])
                cx(a[i], b[i])

        mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")
        gidney_anc.delete()

    with control((n == 1) & perform_incrementation):
        ctrl_list = [a[0], b[0]]
        if ctrl is not None:
            ctrl_list.append(ctrl)
        with control(ctrl_list):
            jasp_cq_gidney_adder(1, b[n:])

    if ctrl is not None:
        mcx([ctrl, a[0]], ctrl_anc[0], method="gidney")
        cx(ctrl_anc[0], b[0])
        mcx([ctrl, a[0]], ctrl_anc[0], method="gidney_inv")
    else:
        cx(a[0], b[0])

    if ctrl is not None:
        ctrl_anc.delete()


# ============================================================================
# Top-level dispatcher — the old gidney_adder
# ============================================================================

def gidney_adder_old(a, b, c_in=None, c_out=None):
    """
    In-place adder function based on https://arxiv.org/abs/1709.06648.
    Performs b += a.

    This is the OLD version that dispatches to separate qq/cq implementations.
    """
    if check_for_tracing_mode():
        if isinstance(a, (QuantumVariable, DynamicQubitArray)):
            return jasp_qq_gidney_adder(a, b)
        else:
            return jasp_cq_gidney_adder(a, b)

    if isinstance(a, (int, str, np.int_)):
        return custom_control(cq_gidney_adder)(a, b, c_in=c_in, c_out=c_out)
    else:
        return qq_gidney_adder(a, b, c_in=c_in, c_out=c_out)