import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx, merge
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.circuit import fast_append
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def extract_boolean_digit(integer, digit):
    """Return bit ``digit`` from an integer-like value as a JAX boolean.

    In tracing mode ``integer`` can be a symbolic big-integer representation, so this
    helper keeps the bit extraction logic in one place for both static and Jasp paths.
    """
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    # Use BigInteger method if available (e.g., in JAX tracing mode)
    if isinstance(integer, BigInteger):
        return jnp.bool_(integer.get_bit(digit))
    # Otherwise extract bit via bitwise shift and mask
    return jnp.bool_(integer >> digit & 1)


def execute_gidney_ladder(n, tracing, qs, init_step, forward_step, mid_step, backward_step, final_step):
    """Run the common Gidney carry ladder skeleton.

    The caller provides the step actions for each phase. This keeps the control flow
    (forward pass, midpoint, backward pass) identical across cq and qq variants while
    allowing mode-specific gate details.
    """
    # Use JAX-safe iteration for tracing mode, regular range otherwise
    range_fn = jrange if tracing else range

    # Carry ladder only runs when operand width > 1
    with control(n > 1):
        # Allocate n-1 ancilla qubits for the carry propagation chain
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*") if tracing \
            else QuantumVariable(n - 1, name="gidney_anc*", qs=qs)

        # Initialize first carry bit from input
        init_step(gidney_anc)

        # Forward pass: propagate carries from LSB to MSB
        for j in range_fn(n - 2):
            i = j + 1
            forward_step(gidney_anc, i)

        # Midpoint: finalize MSB and transition to backward pass
        mid_step(gidney_anc)

        # Backward pass: uncompute carries from MSB back to LSB
        for j in range_fn(n - 2):
            i = n - j - 2
            backward_step(gidney_anc, i)

        # Finalize first carry uncomputation
        final_step(gidney_anc)
        # Clean up ancilla register
        gidney_anc.delete(verify=False)


def apply_if_classical_bit_set(a_int, bit_index, tracing, op):
    """Execute ``op`` if classical bit ``bit_index`` is one.

    In tracing mode the condition must be expressed with a quantum control context;
    in static mode we can use a direct Python bit test.
    """
    if tracing:
        # In JAX tracing, wrap condition in quantum control for dynamic code paths
        with control(extract_boolean_digit(a_int, bit_index)):
            op()
    elif (int(a_int) >> bit_index) & 1:
        # In static mode, execute conditionally via Python if statement
        op()


def execute_shared_mid_step(gidney_anc, n, b_qbs, ctrl_qb=None, tracing=False, a_int=None, a_qbs=None):
    """Execute the midpoint of the ladder where the MSB sum bit is produced.

    ``a_qbs is None`` indicates semi-classical mode, otherwise quantum-quantum mode.
    """
    if a_qbs is None:
        cx(gidney_anc[n - 2], b_qbs[n - 1])
    else:
        if ctrl_qb is not None:
            mcx([ctrl_qb, gidney_anc[n - 2]], b_qbs[n - 1])
            mcx([ctrl_qb, a_qbs[n - 1]], b_qbs[n - 1])
        else:
            cx(gidney_anc[n - 2], b_qbs[n - 1])
            cx(a_qbs[n - 1], b_qbs[n - 1])


def execute_shared_init_step(gidney_anc, b_qbs, ctrl_qb=None, tracing=False, a_int=None, a_qbs=None):
    """Execute the first carry-generation step of the ladder."""
    if a_qbs is None:
        def first_forward_carry():
            if ctrl_qb is None:
                cx(b_qbs[0], gidney_anc[0])
            else:
                mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney")

        apply_if_classical_bit_set(a_int, 0, tracing, first_forward_carry)
    else:
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")


def execute_shared_forward_step(gidney_anc, i, b_qbs, ctrl_qb=None, tracing=False, a_int=None, a_qbs=None):
    """Execute one forward carry-propagation step at position ``i``."""
    cx(gidney_anc[i - 1], b_qbs[i])

    if a_qbs is None:
        def toggle_forward_anc():
            if ctrl_qb is None:
                x(gidney_anc[i - 1])
            else:
                cx(ctrl_qb, gidney_anc[i - 1])

        apply_if_classical_bit_set(a_int, i, tracing, toggle_forward_anc)
        mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")
        apply_if_classical_bit_set(a_int, i, tracing, toggle_forward_anc)
    else:
        cx(gidney_anc[i - 1], a_qbs[i])
        mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney")

    cx(gidney_anc[i - 1], gidney_anc[i])


def execute_shared_backward_step(gidney_anc, i, b_qbs, ctrl_qb=None, tracing=False, a_int=None, a_qbs=None):
    """Execute one backward uncomputation step at position ``i``."""
    cx(gidney_anc[i - 1], gidney_anc[i])

    if a_qbs is None:
        if ctrl_qb is not None:
            def toggle_backward_ctrl():
                cx(ctrl_qb, gidney_anc[i - 1])

            apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_ctrl)
            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
            apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_ctrl)
        else:
            def toggle_backward_anc():
                x(gidney_anc[i - 1])

            apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_anc)
            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
            apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_anc)
    else:
        mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney_inv")
        if ctrl_qb is not None:
            mcx([ctrl_qb, a_qbs[i]], b_qbs[i])
            cx(gidney_anc[i - 1], a_qbs[i])
            cx(gidney_anc[i - 1], b_qbs[i])
        else:
            cx(gidney_anc[i - 1], a_qbs[i])
            cx(a_qbs[i], b_qbs[i])


def execute_shared_final_step(gidney_anc, b_qbs, ctrl_qb=None, tracing=False, a_int=None, a_qbs=None):
    """Execute the final cleanup step for the first carry bit."""
    if a_qbs is None:
        def first_backward_carry():
            if ctrl_qb is None:
                cx(b_qbs[0], gidney_anc[0])
            else:
                mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney_inv")

        apply_if_classical_bit_set(a_int, 0, tracing, first_backward_carry)
    else:
        mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")


def semi_classical_gidney_core(a_int, b_qbs, c_in_qb=None, c_out_qb=None, ctrl_qb=None, tracing=False):
    """Semi-classical Gidney core for ``b += a_int``.

    This path never encodes ``a_int`` into a quantum register. Instead, each bit of
    ``a_int`` conditionally toggles the same ladder structure used by the qq variant.
    """
    # Prepare target register: use slicing in tracing mode (JAX-safe), list otherwise
    if tracing:
        # Dynamic QuantumVariables cannot be converted via list() during tracing.
        b_qbs = b_qbs[:]
    else:
        b_qbs = list(b_qbs)

    # Helper to conditionally apply operations based on classical bit values
    def apply_if_bit_set(bit_index, op):
        apply_if_classical_bit_set(a_int, bit_index, tracing, op)

    # Handle carry-in by logical prepending: prepend c_in qubit and shift a_int left with +1
    if c_in_qb is not None:
        if isinstance(c_in_qb, QuantumBool):
            c_in_qb = c_in_qb[0]
        # Carry-in is modeled as prepending a logical one-bit source.
        b_qbs = [c_in_qb] + b_qbs
        a_int = (a_int << 1) + 1

    # Handle carry-out by logical extension: append zero to both a and b for overflow capture
    if c_out_qb is not None:
        if isinstance(c_out_qb, QuantumBool):
            c_out_qb = c_out_qb[0]
        # Carry-out is modeled as extending the target register by one zero bit.
        b_qbs = b_qbs + [c_out_qb]

    # Determine working register width
    n = jlen(b_qbs) if tracing else len(b_qbs)

    # Optimization for single-qubit case (static mode only): apply XOR directly without ladder
    if (not tracing) and n == 1:
        def single_bit_update():
            if ctrl_qb is None:
                x(b_qbs[0])
            else:
                cx(ctrl_qb, b_qbs[0])

        apply_if_bit_set(0, single_bit_update)
        return

    # Define ladder phase callbacks that specialize the shared executors for semi-classical mode
    def init_step(gidney_anc):
        # Initialize carry chain with classical a_int bit 0
        execute_shared_init_step(gidney_anc, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_int=a_int)

    def forward_step(gidney_anc, i):
        # Propagate carry forward, conditioned on classical a_int bit i
        execute_shared_forward_step(gidney_anc, i, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_int=a_int)

    def mid_step(gidney_anc):
        # Produce MSB sum bit from carry chain
        execute_shared_mid_step(gidney_anc, n, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_int=a_int)

    def backward_step(gidney_anc, i):
        # Uncompute carry chain backward from position i
        execute_shared_backward_step(gidney_anc, i, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_int=a_int)

    def final_step(gidney_anc):
        # Finalize uncomputation of first carry bit
        execute_shared_final_step(gidney_anc, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_int=a_int)

    # Execute the Gidney carry-ladder with semi-classical phase specializations
    execute_gidney_ladder(
        n=n,
        tracing=tracing,
        qs=None if tracing else b_qbs[0].qs(),
        init_step=init_step,
        forward_step=forward_step,
        mid_step=mid_step,
        backward_step=backward_step,
        final_step=final_step,
    )

    # Post-ladder: apply source bits via XOR after carry chain has been uncomputed
    # This ensures all operations are Clifford + classically-controlled X gates (no ancilla garbage)
    range_fn = jrange if tracing else range
    for i in range_fn(n):
        # Skip carry-in qubit (already absorbed into ladder)
        if c_in_qb is not None and i == 0:
            continue

        def final_sum_update():
            if ctrl_qb is not None:
                cx(ctrl_qb, b_qbs[i])
            else:
                x(b_qbs[i])

        apply_if_bit_set(i, final_sum_update)


def quantum_gidney_core(a_qbs, b_qbs, ctrl_qb=None, tracing=False, qs=None):
    """Quantum-quantum Gidney core for ``b += a`` using the shared ladder phases."""
    # Determine width of quantum addition (must match b_qbs after alignment)
    n = jlen(a_qbs)

    # Define ladder phase callbacks that specialize the shared executors for quantum-quantum mode
    def init_step(gidney_anc):
        # Initialize carry chain using quantum a_qbs[0] and b_qbs[0]
        execute_shared_init_step(gidney_anc, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_qbs=a_qbs)

    def forward_step(gidney_anc, i):
        # Propagate carry forward using quantum bits a_qbs[i] and b_qbs[i]
        execute_shared_forward_step(gidney_anc, i, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_qbs=a_qbs)

    def mid_step(gidney_anc):
        # Produce MSB sum from carry chain and quantum MSBs
        execute_shared_mid_step(gidney_anc, n, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_qbs=a_qbs)

    def backward_step(gidney_anc, i):
        # Uncompute carries using quantum bits
        execute_shared_backward_step(gidney_anc, i, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_qbs=a_qbs)

    def final_step(gidney_anc):
        # Finalize uncomputation of first carry
        execute_shared_final_step(gidney_anc, b_qbs, ctrl_qb=ctrl_qb, tracing=tracing, a_qbs=a_qbs)

    # Execute the Gidney carry-ladder with quantum-quantum phase specializations
    execute_gidney_ladder(
        n=n,
        tracing=tracing,
        qs=qs,
        init_step=init_step,
        forward_step=forward_step,
        mid_step=mid_step,
        backward_step=backward_step,
        final_step=final_step,
    )


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.
    Based on https://arxiv.org/abs/1709.06648. Works in both standard
    Python and jasp tracing modes.
    """
    # BRANCH 1: Semi-classical path — a is classical (int/string/JAX scalar), b is quantum
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        # Convert string binary representation to integer
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        # Classify input: check if we're in JAX tracing mode and what type a is
        tracing_mode = check_for_tracing_mode()
        is_concrete_int = isinstance(a, (int, np.integer))
        is_scalar_like = getattr(a, "ndim", None) == 0
        is_biginteger_like = hasattr(a, "get_bit")

        # Detect semi-classical mode: concrete int, or in tracing mode with a JAX scalar/BigInteger
        # Treat scalar symbolic values (e.g. JAX tracers) and BigInteger-like values as
        # semi-classical sources. They provide bits, but are not sliceable qubit arrays.
        if is_concrete_int or (tracing_mode and (is_scalar_like or is_biginteger_like)):
            # Reduce a to the width of b to avoid unnecessary classical bits
            if is_concrete_int:
                a = int(a) % (1 << jlen(b))

            # JAX TRACING mode: execute semi-classical core with JAX integration
            if tracing_mode:
                if isinstance(a, int):
                    a = jnp.array(a, dtype=jnp.int64)
                semi_classical_gidney_core(a, b, c_in_qb=c_in, c_out_qb=c_out, ctrl_qb=ctrl, tracing=True)
                return

            # STATIC mode: merge QuantumSessions and execute with fast_append optimization
            qs_list = [b[0].qs()]
            for qb in [c_in, c_out, ctrl]:
                if qb is not None:
                    if isinstance(qb, QuantumBool):
                        qb = qb[0]
                    qs_list.append(qb.qs())
            merge(qs_list)

            # Execute semi-classical core in fast_append mode for efficiency
            with fast_append(3):
                semi_classical_gidney_core(a, b, c_in_qb=c_in, c_out_qb=c_out, ctrl_qb=ctrl, tracing=False)
            return

    # BRANCH 2: Quantum-quantum path — both a and b are quantum registers, align their widths
    # Determine quantum session (None in tracing mode)
    qs = None if check_for_tracing_mode() else b[0].qs()

    # Measure dimensions and truncate a to match b if needed
    dim_a = jlen(a)
    dim_b = jlen(b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    # If a is shorter than b, pad a with clean ancillas (zeros) to match width
    # If a is longer than b, a is already truncated above
    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs) if qs is not None \
        else QuantumVariable(extension_size, name="gidney_a_ext*")
    a = a[:] + a_ext[:]
    b = b[:]

    # Handle optional carry-in: prepend a |1> ancilla to both a and b logically
    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        # Create and initialize temporary |1> qubit for carry-in
        one_anc = QuantumBool(name="gidney_cin_one*", qs=qs) if qs is not None \
            else QuantumBool(name="gidney_cin_one*")
        x(one_anc[0])
        # Inject carry-in by prepending a temporary |1> source bit to a.
        a = one_anc[:] + a
        b = c_in_qbs + b

    # Handle optional carry-out: append a |0> ancilla to both a and b logically
    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        # Create temporary |0> qubit for carry-out capture
        zero_anc = QuantumBool(name="gidney_cout_zero*", qs=qs) if qs is not None \
            else QuantumBool(name="gidney_cout_zero*")
        # Capture carry-out by appending a temporary |0> source bit to a.
        a = a + zero_anc[:]
        b = b + c_out_qbs

    # Execute the quantum-quantum adder core
    quantum_gidney_core(a, b, ctrl_qb=ctrl, tracing=check_for_tracing_mode(), qs=qs)

    # Post-ladder: apply LSB of a (which was not processed by ladder when c_in is None)
    # This step is skipped when c_in is present (LSB absorbed into carry-in logic)
    if c_in is None:
        # Apply a[0] XOR onto b[0] after ladder uncomputation
        if ctrl is not None:
            mcx([ctrl, a[0]], b[0])
        else:
            cx(a[0], b[0])

    # Clean up temporary ancillas
    if one_anc is not None:
        # Reset and delete carry-in ancilla
        x(one_anc[0])
        one_anc.delete()

    if zero_anc is not None:
        # Delete carry-out ancilla (automatically in |0> state)
        zero_anc.delete()

    # Delete width-extension ancillas
    a_ext.delete(verify=False)
