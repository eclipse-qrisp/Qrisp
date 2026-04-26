
from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np
from qrisp.core import QuantumVariable, x, cx, mcx, merge
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.circuit import fast_append
from qrisp.jasp import jrange, jlen, DynamicQubitArray, check_for_tracing_mode


def _extract_boolean_digit(integer, digit):
    """Return bit ``digit`` from an integer-like value as a JAX boolean.

    In tracing mode ``integer`` can be a symbolic big-integer representation, so this
    helper keeps the bit extraction logic in one place for both static and Jasp paths.

    Parameters
    ----------
    integer : int, jnp.ndarray, or BigInteger
        The value from which a single bit is extracted.  Can be a concrete
        Python/NumPy int, a JAX scalar tracer, or a symbolic ``BigInteger``
        used during Jasp tracing.
    digit : int or jnp.ndarray
        Zero-based index of the bit to extract (0 = LSB).

    Returns
    -------
    jnp.bool_
        ``True`` when bit ``digit`` of ``integer`` is 1, ``False`` otherwise.
    """
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    # during JAX tracing, integers are represented as BigInteger objects that
    # store their bits symbolically – use the dedicated accessor in that case
    if isinstance(integer, BigInteger):
        return jnp.bool_(integer.get_bit(digit))
    # for concrete ints and JAX scalars, shift the target bit into the LSB
    # position and mask it, then cast to a JAX boolean so both paths return
    # the same dtype
    return jnp.bool_(integer >> digit & 1)


def _apply_gidney_adder_gates(n, tracing, qs, init_step, forward_step, mid_step, backward_step, final_step):
    """Run the common Gidney carry ladder skeleton.

    The caller provides the step actions for each phase.  This keeps the control
    flow (forward pass, midpoint, backward pass) identical across semi-classical
    and quantum-quantum variants while allowing mode-specific gate details.

    Parameters
    ----------
    n : int or jnp.ndarray
        Width (number of qubits) of the operands being added.
    tracing : bool
        ``True`` when running inside a JAX/Jasp tracing context, which
        requires ``jrange`` for loops and symbolic allocation.
    qs : QuantumSession or None
        The quantum session that owns the qubits.  ``None`` during tracing.
    init_step : callable(QuantumVariable)
        Callback that seeds the carry chain by computing the first carry
        from bits a[0] and b[0] into ``gidney_anc[0]``.
    forward_step : callable(QuantumVariable, int)
        Callback that propagates the carry forward at bit position *i*
        (called for *i* = 1 … n−2).
    mid_step : callable(QuantumVariable)
        Callback that finalises the MSB sum bit using the last carry.
    backward_step : callable(QuantumVariable, int)
        Callback that uncomputes the carry at bit position *i* during the
        backward pass (called for *i* = n−2 … 1).
    final_step : callable(QuantumVariable)
        Callback that erases the first carry and cleans up ``gidney_anc[0]``.
    """
    # the Gidney adder uses a "carry ladder" – a chain of ancilla qubits where
    # each ancilla stores the carry bit produced at that position.  the ladder
    # has three phases:
    #   forward  – compute carries from the least-significant towards the
    #              most-significant bit
    #   midpoint – finalise the MSB sum bit using the last carry
    #   backward – uncompute all intermediate carries so the ancillas return
    #              to |0> and can be released
    # jrange is used instead of range during JAX tracing so that the loop
    # bounds can be symbolic values
    range_fn = jrange if tracing else range

    # when the operand is only one bit wide there are no carries to propagate,
    # so the entire ladder is skipped – the control() environment guards this
    with control(n > 1):
        # allocate n-1 ancilla qubits, one per carry position between adjacent
        # bit pairs (bit i and bit i+1 share ancilla i)
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*") if tracing \
            else QuantumVariable(n - 1, name="gidney_anc*", qs=qs)

        # seed the carry chain: compute the carry out of the LSB pair (a[0], b[0])
        # into the first ancilla
        init_step(gidney_anc)

        # forward pass – for each subsequent bit position i (1 .. n-2), use the
        # incoming carry and the current a/b bits to produce the next carry
        for j in range_fn(n - 2):
            i = j + 1
            forward_step(gidney_anc, i)

        # midpoint – the last carry (ancilla[n-2]) together with the MSBs
        # determines the MSB of the sum; after this the backward pass begins
        mid_step(gidney_anc)

        # backward pass – walk back from the MSB to bit 1, reversing the
        # carry computation so each ancilla is restored to |0> while
        # simultaneously XOR-ing the sum bits into the b register
        for j in range_fn(n - 2):
            i = n - j - 2
            backward_step(gidney_anc, i)

        # clean up the very first carry ancilla (mirrors the init step)
        final_step(gidney_anc)
        # all ancillas are now back in |0> and can be safely deallocated
        gidney_anc.delete(verify=False)


def _apply_if_classical_bit_set(a_int, bit_index, tracing, op):
    """Execute ``op`` if classical bit ``bit_index`` of ``a_int`` is one.

    In tracing mode the condition must be expressed with a quantum control
    context; in static mode we can use a direct Python bit test.

    Parameters
    ----------
    a_int : int, jnp.ndarray, or BigInteger
        The classical integer whose bits are tested.
    bit_index : int or jnp.ndarray
        Zero-based index of the bit to test (0 = LSB).
    tracing : bool
        ``True`` when running inside a JAX/Jasp tracing context.
    op : callable
        Zero-argument callable that emits quantum gates.  It is invoked
        exactly once if the tested bit is 1, and not at all otherwise.
    """
    if tracing:
        # during JAX tracing the value of a_int is not yet known at Python
        # level, so the branch must be expressed as a quantum-controlled
        # environment that will be resolved when the traced program runs
        with control(_extract_boolean_digit(a_int, bit_index)):
            op()
    elif (int(a_int) >> bit_index) & 1:
        # at normal (non-tracing) time the integer is concrete, so a plain
        # Python if-check on the relevant bit suffices
        op()


def _gidney_adder_gate_order(phase, gidney_anc, b_qbs, ctrl_qb=None, tracing=False, a_int=None, a_qbs=None, i=None, n=None):
    """Execute one step of the Gidney carry ladder.

    Combines the init, forward, mid, backward, and final phases into a single
    entry point.  The ``phase`` string selects which operation to perform.

    Parameters
    ----------
    phase : str
        One of ``"init"``, ``"forward"``, ``"mid"``, ``"backward"``, ``"final"``.
    gidney_anc : QuantumVariable
        Ancilla register for the carry chain.
    b_qbs : list of Qubit
        Target qubit list.
    ctrl_qb : Qubit or None
        Optional external control qubit.
    tracing : bool
        ``True`` during JAX/Jasp tracing.
    a_int : int, jnp.ndarray, BigInteger, or None
        Classical addend (semi-classical mode).  Mutually exclusive with
        ``a_qbs``.
    a_qbs : list of Qubit or None
        Quantum addend qubit list (quantum-quantum mode).
    i : int or jnp.ndarray or None
        Bit position for ``"forward"`` and ``"backward"`` phases.
    n : int or jnp.ndarray or None
        Operand width, required for the ``"mid"`` phase.
    """

    # ── init: seed carry chain with carry_0 = a[0] AND b[0] ──────────────
    if phase == "init":
        if a_qbs is None:
            def first_forward_carry():
                if ctrl_qb is None:
                    cx(b_qbs[0], gidney_anc[0])
                else:
                    mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney")

            _apply_if_classical_bit_set(a_int, 0, tracing, first_forward_carry)
        else:
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney")

    # ── forward: propagate carry at position i into gidney_anc[i] ─────────
    elif phase == "forward":
        cx(gidney_anc[i - 1], b_qbs[i])

        if a_qbs is None:
            def toggle_forward_anc():
                if ctrl_qb is None:
                    x(gidney_anc[i - 1])
                else:
                    cx(ctrl_qb, gidney_anc[i - 1])

            _apply_if_classical_bit_set(a_int, i, tracing, toggle_forward_anc)
            mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney")
            _apply_if_classical_bit_set(a_int, i, tracing, toggle_forward_anc)
        else:
            cx(gidney_anc[i - 1], a_qbs[i])
            mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney")

        cx(gidney_anc[i - 1], gidney_anc[i])

    # ── mid: produce the MSB sum bit from the last carry ──────────────────
    elif phase == "mid":
        if a_qbs is None:
            cx(gidney_anc[n - 2], b_qbs[n - 1])
        else:
            if ctrl_qb is not None:
                mcx([ctrl_qb, gidney_anc[n - 2]], b_qbs[n - 1])
                mcx([ctrl_qb, a_qbs[n - 1]], b_qbs[n - 1])
            else:
                cx(gidney_anc[n - 2], b_qbs[n - 1])
                cx(a_qbs[n - 1], b_qbs[n - 1])

    # ── backward: uncompute carry at position i, produce sum bit ──────────
    elif phase == "backward":
        cx(gidney_anc[i - 1], gidney_anc[i])

        if a_qbs is None:
            if ctrl_qb is not None:
                def toggle_backward_ctrl():
                    cx(ctrl_qb, gidney_anc[i - 1])

                _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_ctrl)
                mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
                _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_ctrl)
            else:
                def toggle_backward_anc():
                    x(gidney_anc[i - 1])

                _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_anc)
                mcx([gidney_anc[i - 1], b_qbs[i]], gidney_anc[i], method="gidney_inv")
                _apply_if_classical_bit_set(a_int, i, tracing, toggle_backward_anc)
        else:
            mcx([a_qbs[i], b_qbs[i]], gidney_anc[i], method="gidney_inv")
            if ctrl_qb is not None:
                mcx([ctrl_qb, a_qbs[i]], b_qbs[i])
                cx(gidney_anc[i - 1], a_qbs[i])
                cx(gidney_anc[i - 1], b_qbs[i])
            else:
                cx(gidney_anc[i - 1], a_qbs[i])
                cx(a_qbs[i], b_qbs[i])

    # ── final: erase carry_0, mirror of init ──────────────────────────────
    elif phase == "final":
        if a_qbs is None:
            def first_backward_carry():
                if ctrl_qb is None:
                    cx(b_qbs[0], gidney_anc[0])
                else:
                    mcx([ctrl_qb, b_qbs[0]], gidney_anc[0], method="gidney_inv")

            _apply_if_classical_bit_set(a_int, 0, tracing, first_backward_carry)
        else:
            mcx([a_qbs[0], b_qbs[0]], gidney_anc[0], method="gidney_inv")


@custom_control
def gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):
    """
    In-place Gidney adder performing ``b += a``.

    Based on `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`_.  Works in
    both standard Python and Jasp tracing modes.  The function automatically
    selects between a semi-classical path (when ``a`` is a classical value) and
    a quantum-quantum path (when both ``a`` and ``b`` are quantum registers).

    Parameters
    ----------
    a : int, str, jnp.ndarray, BigInteger, QuantumVariable, DynamicQubitArray, or list
        The addend.  When ``a`` is a classical value (int, binary string,
        JAX scalar, or ``BigInteger``), the semi-classical path is taken
        and no quantum encoding of ``a`` is created.  When ``a`` is a
        quantum register or qubit list, the quantum-quantum path is used.
    b : QuantumVariable or DynamicQubitArray
        The target register that is updated in-place: ``b ← b + a``.
    c_in : QuantumBool, Qubit, or None
        Optional single-qubit carry-in.  When provided, the addition
        becomes ``b ← b + a + c_in``.
    c_out : QuantumBool, Qubit, or None
        Optional single-qubit carry-out.  When provided, this qubit
        receives the overflow carry of the addition.
    ctrl : Qubit or None
        Optional control qubit.  When provided, the entire addition is
        executed only if ``ctrl`` is in state |1⟩ (controlled addition).
    """
    # semi-classical path: a is a known classical value (int, string, JAX
    # scalar, or BigInteger) and b is a quantum register.  no quantum encoding
    # of a is needed – each bit of a is tested classically to decide which
    # gates to emit
    if not isinstance(a, (QuantumVariable, DynamicQubitArray, list)):
        # accept binary strings like "1101" (LSB-first) and convert to int
        if isinstance(a, str):
            a = int(a[::-1], 2) if a else 0

        # determine execution context: JAX tracing requires symbolic
        # operations; static mode can use concrete Python values directly
        tracing_mode = check_for_tracing_mode()
        is_concrete_int = isinstance(a, (int, np.integer))
        # a JAX tracer that is 0-dimensional (scalar) behaves like an int
        # but cannot be inspected at trace time
        is_scalar_like = getattr(a, "ndim", None) == 0
        # BigInteger objects expose individual bits via .get_bit()
        is_biginteger_like = hasattr(a, "get_bit")

        # any of the above qualifies as a semi-classical source – the value
        # provides classical bit information but is not a qubit array
        if is_concrete_int or (tracing_mode and (is_scalar_like or is_biginteger_like)):
            # mask a to the width of b so higher bits don't produce spurious
            # gates (addition is mod 2^n)
            if is_concrete_int:
                a = int(a) % (1 << jlen(b))

            # in JAX tracing mode, wrap the integer in a JAX array so
            # subsequent bit-shift operations stay within the JAX graph
            if tracing_mode:
                if isinstance(a, int):
                    a = jnp.array(a, dtype=jnp.int64)
            else:
                # in static mode every qubit involved must belong to the same
                # quantum session – merge them before emitting any gates
                qs_list = [b[0].qs()]
                for qb in [c_in, c_out, ctrl]:
                    if qb is not None:
                        if isinstance(qb, QuantumBool):
                            qb = qb[0]
                        qs_list.append(qb.qs())
                merge(qs_list)

            tracing = tracing_mode

            # obtain a flat qubit list from the quantum variable b;
            # slicing is JAX-safe, list() is not during tracing
            if tracing:
                b_qbs = b[:]
            else:
                b_qbs = list(b)

            # carry-in is modelled by prepending the carry-in qubit to b and
            # treating a as if it had an extra LSB of 1.  this way the carry
            # chain naturally absorbs the incoming carry without special logic
            if c_in is not None:
                c_in_qb = c_in[0] if isinstance(c_in, QuantumBool) else c_in
                b_qbs = [c_in_qb] + b_qbs
                # shift a left by one and set bit 0 to represent the carry-in
                a = (a << 1) + 1

            # carry-out is modelled by appending a fresh qubit to b; the
            # ladder will naturally propagate the overflow carry into it
            if c_out is not None:
                c_out_qb = c_out[0] if isinstance(c_out, QuantumBool) else c_out
                b_qbs = b_qbs + [c_out_qb]

            # n is the total width of the (possibly extended) target register
            n = jlen(b_qbs) if tracing else len(b_qbs)

            # when the target register is a single qubit the carry ladder
            # degenerates – just conditionally flip that qubit
            if (not tracing) and n == 1:
                def single_bit_update():
                    if ctrl is None:
                        x(b_qbs[0])
                    else:
                        cx(ctrl, b_qbs[0])

                _apply_if_classical_bit_set(a, 0, tracing, single_bit_update)
                return

            # wire the ladder callbacks to _gidney_adder_gate_order,
            # passing a as a classical integer (a_int) so each step can
            # test individual bits to decide which gates to emit
            def init_step(gidney_anc):
                _gidney_adder_gate_order("init", gidney_anc, b_qbs, ctrl_qb=ctrl, tracing=tracing, a_int=a)

            def forward_step(gidney_anc, i):
                _gidney_adder_gate_order("forward", gidney_anc, b_qbs, ctrl_qb=ctrl, tracing=tracing, a_int=a, i=i)

            def mid_step(gidney_anc):
                _gidney_adder_gate_order("mid", gidney_anc, b_qbs, ctrl_qb=ctrl, tracing=tracing, a_int=a, n=n)

            def backward_step(gidney_anc, i):
                _gidney_adder_gate_order("backward", gidney_anc, b_qbs, ctrl_qb=ctrl, tracing=tracing, a_int=a, i=i)

            def final_step(gidney_anc):
                _gidney_adder_gate_order("final", gidney_anc, b_qbs, ctrl_qb=ctrl, tracing=tracing, a_int=a)

            # fast_append batches gate emissions for better performance in
            # static mode; in tracing mode no batching is needed
            ctx = fast_append(3) if not tracing else nullcontext()
            with ctx:
                # run the three-phase carry ladder (forward, mid, backward)
                _apply_gidney_adder_gates(
                    n=n,
                    tracing=tracing,
                    qs=None if tracing else b_qbs[0].qs(),
                    init_step=init_step,
                    forward_step=forward_step,
                    mid_step=mid_step,
                    backward_step=backward_step,
                    final_step=final_step,
                )

                # after the carry chain has been fully uncomputed, each bit of
                # a still needs to be XOR-ed onto the corresponding b qubit to
                # complete the addition (the ladder only handled the carries)
                range_fn = jrange if tracing else range
                for i in range_fn(n):
                    # the carry-in qubit was prepended at position 0 and is
                    # already accounted for by the ladder – skip it
                    if c_in is not None and i == 0:
                        continue

                    def final_sum_update():
                        if ctrl is not None:
                            cx(ctrl, b_qbs[i])
                        else:
                            x(b_qbs[i])

                    # only emit the XOR gate when the corresponding
                    # classical bit of a is 1
                    _apply_if_classical_bit_set(a, i, tracing, final_sum_update)

            return

    # quantum-quantum path: both a and b are quantum registers.
    # the ladder requires both operands to have the same width, so we
    # truncate or pad a to match b
    tracing = check_for_tracing_mode()
    # in tracing mode there is no concrete QuantumSession; in static mode
    # all qubits share one session
    qs = None if tracing else b[0].qs()

    # if a is wider than b, extra high bits cannot affect the (mod 2^n)
    # result – drop them.  if a is narrower, pad with |0> ancillas below
    dim_a = jlen(a)
    dim_b = jlen(b)
    effective_size_a = jnp.minimum(dim_a, dim_b)
    a = a[:effective_size_a]

    # allocate zero-initialised ancillas to extend a to the same width as b
    extension_size = jnp.maximum(0, dim_b - dim_a)
    a_ext = QuantumVariable(extension_size, name="gidney_a_ext*", qs=qs) if qs is not None \
        else QuantumVariable(extension_size, name="gidney_a_ext*")
    # build flat qubit lists so indexing works uniformly for both paths
    a = a[:] + a_ext[:]
    b = b[:]

    # carry-in handling for the qq path: prepend a |1> ancilla to a and the
    # carry-in qubit to b.  the ladder then computes carry_0 = 1 AND c_in,
    # which equals c_in – exactly the desired incoming carry
    one_anc = None
    if c_in is not None:
        c_in_qbs = c_in[:] if isinstance(c_in, QuantumBool) else [c_in]
        one_anc = QuantumBool(name="gidney_cin_one*", qs=qs) if qs is not None \
            else QuantumBool(name="gidney_cin_one*")
        # flip the ancilla to |1> so the Toffoli in the init step fires when
        # c_in is |1>
        x(one_anc[0])
        a = one_anc[:] + a
        b = c_in_qbs + b

    # carry-out handling: append a |0> ancilla to a and the carry-out qubit
    # to b.  the MSB carry will propagate into c_out naturally
    zero_anc = None
    if c_out is not None:
        c_out_qbs = c_out[:] if isinstance(c_out, QuantumBool) else [c_out]
        zero_anc = QuantumBool(name="gidney_cout_zero*", qs=qs) if qs is not None \
            else QuantumBool(name="gidney_cout_zero*")
        a = a + zero_anc[:]
        b = b + c_out_qbs

    # n is the matched width after padding and carry extensions
    n = jlen(a)

    # wire the ladder callbacks to _gidney_adder_gate_order,
    # passing a as a quantum qubit list (a_qbs) so the steps use
    # quantum-quantum Toffolis instead of classically-controlled gates
    def init_step(gidney_anc):
        _gidney_adder_gate_order("init", gidney_anc, b, ctrl_qb=ctrl, tracing=tracing, a_qbs=a)

    def forward_step(gidney_anc, i):
        _gidney_adder_gate_order("forward", gidney_anc, b, ctrl_qb=ctrl, tracing=tracing, a_qbs=a, i=i)

    def mid_step(gidney_anc):
        _gidney_adder_gate_order("mid", gidney_anc, b, ctrl_qb=ctrl, tracing=tracing, a_qbs=a, n=n)

    def backward_step(gidney_anc, i):
        _gidney_adder_gate_order("backward", gidney_anc, b, ctrl_qb=ctrl, tracing=tracing, a_qbs=a, i=i)

    def final_step(gidney_anc):
        _gidney_adder_gate_order("final", gidney_anc, b, ctrl_qb=ctrl, tracing=tracing, a_qbs=a)

    # run the three-phase carry ladder for the quantum-quantum case
    _apply_gidney_adder_gates(
        n=n,
        tracing=tracing,
        qs=qs,
        init_step=init_step,
        forward_step=forward_step,
        mid_step=mid_step,
        backward_step=backward_step,
        final_step=final_step,
    )

    # the ladder handles carries and sum bits for positions 1..n-1, but the
    # LSB sum bit (b[0] ^= a[0]) is not produced by the ladder.  when carry-in
    # is present the LSB was the carry-in qubit and was already handled above
    if c_in is None:
        if ctrl is not None:
            # controlled XOR of a[0] onto b[0]
            mcx([ctrl, a[0]], b[0])
        else:
            cx(a[0], b[0])

    # clean up temporary ancillas that were created for carry-in / carry-out
    if one_anc is not None:
        # reset the |1> ancilla back to |0> before deletion
        x(one_anc[0])
        one_anc.delete()

    if zero_anc is not None:
        # the carry-out ancilla is already |0> after the ladder uncomputation
        zero_anc.delete()

    # release the zero-padding ancillas that extended a to match b's width
    a_ext.delete(verify=False)
