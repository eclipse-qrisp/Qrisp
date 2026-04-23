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

    # when a is classical
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):

        # If `a` is a binary string (e.g. "011"), it is assumed to be in
        # little-endian order — meaning the first character represents the
        # least significant bit (LSB), i.e. the 2^0 place, the second
        # character represents 2^1, the third represents 2^2, and so on.
        # This matches how qubits are indexed: qubit 0 holds the smallest
        # power of two. Python's int(..., 2) expects big-endian order
        # (MSB first), so we reverse the string before parsing.
        # Example: "011" (little-endian) → reversed "110" → int("110", 2) = 6
        # An empty string is treated as 0.

        # a is used as a string input by the QCLA adder (For example, cq_sum_path.py)
        # using the Gidney adder. 
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        # modulo operation to ensure a fits within the number of bits in b
        # this is helpful given that the addition is performed in-place on b
        # so any bits of a beyond the length of b would not affect the result
        # and can be safely ignored
        if not check_for_tracing_mode() and isinstance(a, (int, np.integer)):
            a = int(a) % (1 << len(b))

        qs = None if check_for_tracing_mode() else b[0].qs()
        q_a = QuantumVariable(jlen(b), qs=qs)

        with conjugate(int_encoder)(q_a, a):
            gidney_adder(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return


    qs = None if check_for_tracing_mode() else b[0].qs()


    dim_a = jlen(a)
    dim_b = jlen(b)

    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs)

    a = a[:] + a_ext[:]
    b = b[:]


    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        one_anc = QuantumBool(name="gidney_cin_one*", qs=qs)
        x(one_anc[0])
        a = one_anc[:] + a
        b = c_in_qbs + b


    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        zero_anc = QuantumBool(name="gidney_cout_zero*", qs=qs)
        a = a + zero_anc[:]
        b = b + c_out_qbs

    n = jlen(a)


    with control(n > 1):
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*", qs=qs)

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

    a_ext.delete()