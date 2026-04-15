import jax.numpy as jnp
import numpy as np

from qrisp.core import QuantumVariable, x, cx, mcx
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control, conjugate
from qrisp.misc import int_encoder
from qrisp.jasp import jrange, jlen, DynamicQubitArray


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on https://arxiv.org/abs/1709.06648. Works in both standard
    Python and jasp tracing modes with a single code path.
    """

    # ------------------------------------------------------------------
    # Classical a: encode into a temp register and recurse into the
    # quantum-quantum path. Classify by negation so JAX tracer scalars
    # also take this path.
    # ------------------------------------------------------------------
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):

        q_a = QuantumVariable(jlen(b))
        with conjugate(int_encoder)(q_a, a):
            gidney_adder(q_a, b, c_in=c_in, c_out=c_out, ctrl=ctrl)
        q_a.delete()
        return

    # ------------------------------------------------------------------
    # Size normalization.
    # ------------------------------------------------------------------
    dim_a = jlen(a)
    dim_b = jlen(b)

    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*")

    a = a[:] + a_ext[:]
    b = b[:]

    # ------------------------------------------------------------------
    # c_in: prepend c_in as a new low bit of b with a fresh |1> ancilla
    # at the matching position of a.
    # ------------------------------------------------------------------
    one_anc = None
    if c_in is not None:
        if not isinstance(c_in, QuantumBool):
            raise TypeError("c_in must be a QuantumBool")
        one_anc = QuantumBool(name="gidney_cin_one*")
        x(one_anc[0])
        a = one_anc[:] + a
        b = c_in[:] + b

    # ------------------------------------------------------------------
    # c_out: append c_out to b with a fresh |0> ancilla on the a side.
    # ------------------------------------------------------------------
    zero_anc = None
    if c_out is not None:
        if not isinstance(c_out, QuantumBool):
            raise TypeError("c_out must be a QuantumBool")
        zero_anc = QuantumBool(name="gidney_cout_zero*")
        a = a + zero_anc[:]
        b = b + c_out[:]

    n = jlen(a)

    # ------------------------------------------------------------------
    # Main V-shaped circuit.
    # ------------------------------------------------------------------
    with control(n > 1):
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")

        # Initial Toffoli
        mcx([a[0], b[0]], gidney_anc[0], method="gidney")

        # Left part of the V
        for j in jrange(n - 2):
            i = j + 1
            cx(gidney_anc[i - 1], a[i])
            cx(gidney_anc[i - 1], b[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])

        # Tip of the V
        if ctrl is not None:
            mcx([ctrl, gidney_anc[n - 2]], b[n - 1])
            mcx([ctrl, a[n - 1]], b[n - 1])
        else:
            cx(gidney_anc[n - 2], b[n - 1])
            cx(a[n - 1], b[n - 1])

        # Right part of the V
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

        # Final gidney_anc[0] uncomputation
        mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")

        gidney_anc.delete()

    # ------------------------------------------------------------------
    # Top-right CX at position 0. Skipped when c_in is set: position 0
    # is then the c_in qubit, and a[0] is the |1> one_anc, so firing the
    # CX would flip c_in (a spurious side-effect). Bit-0 of the user's
    # result is correctly written by the i==1 iteration of the right V.
    # ------------------------------------------------------------------
    if c_in is None:
        if ctrl is not None:
            mcx([ctrl, a[0]], b[0])
        else:
            cx(a[0], b[0])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if one_anc is not None:
        x(one_anc[0])
        one_anc.delete()

    if zero_anc is not None:
        zero_anc.delete()

    a_ext.delete()


# FAILED core_tests/test_custom_control.py::test_custom_control - TypeError: '>' not supported between instances of 'str' and 'int'
# FAILED jax_tests/test_montgomery.py::test_montgomery_not_jasp_qq - Exception: Not enough qubits to encode integer 15
# FAILED jax_tests/test_montgomery.py::test_montgomery_find_order - Exception: Not enough qubits to encode integer 55
# FAILED primitives_tests/arithmetic_tests/test_gidney_adder.py::test_gidney_adder_string_input - TypeError: '>' not supported between instances of 'str' and 'int'
# FAILED primitives_tests/arithmetic_tests/test_gidney_adder.py::test_gidney_adder_empty_string_input - TypeError: '>' not supported between instances of 'str' and 'int'
# FAILED primitives_tests/arithmetic_tests/test_modular_arithmetic.py::test_modular_arithmetic - Exception: Not enough qubits to encode integer 40
# FAILED primitives_tests/arithmetic_tests/test_qcla.py::test_qq_qcla_adder - TypeError: c_out must be a QuantumBool
# FAILED primitives_tests/arithmetic_tests/test_qcla.py::test_cq_qcla_adder - TypeError: '>' not supported between instances of 'str' and 'int'
#FAILED primitives_tests/arithmetic_tests/test_quantum_arithmetic.py::test_quantum_arithmetic - Exception: Not enough qubits to encode integer 2
