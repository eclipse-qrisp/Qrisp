# ********************************************************************************
# Copyright (c) 2026 the Qrisp authors
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# with the GNU Classpath Exception which is
# available at https://www.gnu.org/software/classpath/license.html.
#
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

import numpy as np

try:
    import cirq
except ImportError:
    cirq = None

from qrisp.circuit import ControlledOperation, Operation
from qrisp.circuit import standard_operations as ops
from qrisp.circuit.quantum_circuit import QuantumCircuit


def _transpile_to_known_gates(qrisp_circuit, gate_map):
    """Break down every Qrisp gate that Cirq doesn't know about.

    When you hand us a Qrisp circuit, it may well contain high-level gates
    (custom operations, xxyy, rxx, rzz, ...) that have no direct Cirq
    counterpart.  To deal with that, we keep asking Qrisp's own transpiler
    to decompose them into elementary instructions — the ones whose names
    show up in ``gate_map``.

    We have to do this iteratively, because a single transpilation pass can
    introduce *new* unknown gates (a composite gate may decompose into other
    composites).  So we loop until either every gate name is known or we
    stop making progress.

    How do we know we're stuck?  If the set of unknown names is identical
    before and after a pass, the remaining gates can't be broken down any
    further, and we give up with a ValueError.

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The circuit whose unknown gates we'll transpile.
    gate_map : dict
        Maps gate names to Cirq gates.  Anything not in here counts as
        "unknown" and gets transpiled.

    Returns
    -------
    QuantumCircuit
        A transpiled circuit in which every gate name is a key of
        ``gate_map``.

    Raises
    ------
    ValueError
        If some gate can't be decomposed down to a known name.

    """

    def _unknown_names(circ):
        return {instr.op.name for instr in circ.data if instr.op.name not in gate_map}

    while True:
        unknown = _unknown_names(qrisp_circuit)
        if not unknown:
            break

        def _transpile_predicate(op, _unknown=unknown):
            return op.name in _unknown

        try:
            transpiled = qrisp_circuit.transpile(transpile_predicate=_transpile_predicate)
        except Exception as exc:
            raise ValueError(
                f"Gates {unknown} could not be transpiled and are not supported by the Qrisp to Cirq converter."
            ) from exc

        new_unknown = _unknown_names(transpiled)
        if new_unknown == unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"The following gates could not be decomposed into elementary "
                f"instructions: {names}. Try transpiling the circuit with "
                f"Qrisp's transpile() method before calling to_cirq(), or "
                f"use only gates supported natively by the converter."
            )

        qrisp_circuit = transpiled

    return qrisp_circuit


def _ctrl_state_to_cirq_values(ctrl_state):
    """Turn a Qrisp control-state string into cirq control values.

    Qrisp writes control states as strings, e.g. ``"101"``, with one
    character per control qubit.  Cirq wants a list of integers in the
    same order.  This is just the little translation in between: ``"101"``
    becomes ``[1, 0, 1]``.

    convert_to_cirq uses this to pass control values to cirq's
    ``base.controlled()``.
    """
    return [int(c) for c in ctrl_state]


def _cirq_values_to_ctrl_str(control_values, obj_name):
    """Turn cirq control values back into a Qrisp control-state string.

    Cirq stores control values as a ProductOfSums object that iterates
    over one tuple per control qubit — so ``control_values=[1, 0, 1]``
    shows up as ``(1,), (0,), (1,)``.  We collapse that into Qrisp's
    compact ``"101"`` form.

    We only support single-valued controls, i.e. each control qubit must
    be active for exactly one basis state.  If you give us something fancier
    (a multi-valued control like ``(0, 1)``, meaning "active on both 0 and
    1") or a value outside ``{0, 1}``, we raise a ValueError.

    Parameters
    ----------
    control_values : cirq.ProductOfSums
        The control values from a Cirq ControlledGate or ControlledOperation.
    obj_name : str
        What we're converting, used in error messages so you can find the
        offending object.

    Returns
    -------
    str
        A Qrisp-style ``ctrl_state`` string.

    Raises
    ------
    ValueError
        If any control has multiple values or a value outside ``{0, 1}``.

    """
    result = []
    for v in control_values:
        if len(v) != 1:
            raise ValueError(f"Multi-valued control {v} in {obj_name} not supported.")
        val = v[0]
        if val not in (0, 1):
            raise ValueError(f"Unsupported control value {val} in {obj_name}.")
        result.append(str(val))
    return "".join(result)


def _unpack_cirq_control_layers(gate):
    """Peel the cirq.ControlledGate wrappers off a gate.

    Cirq lets you build controlled gates in a couple of ways, e.g.
    ``cirq.X.controlled(num_controls=2, control_values=[1, 0])``, which
    produces a ControlledGate sitting around the inner gate.  You can even
    nest them: ``cirq.X.controlled().controlled()`` gives you a stack of
    layers.

    Here we unwrap all of those layers, collecting:
      - ``(num_controls, ctrl_state)`` tuples, outermost layer first, and
      - the innermost, no-longer-controlled gate.

    Later, _apply_ctrl_layers puts those layers back onto the Qrisp
    operation.

    Parameters
    ----------
    gate : cirq.Gate
        A gate that may or may not be wrapped in ControlledGate layers.

    Returns
    -------
    tuple
        ``(inner_gate, list_of_layers)``, where each layer is
        ``(num_controls: int, ctrl_state: str)``.

    """
    ctrl_layers = []
    inner_gate = gate
    while isinstance(inner_gate, cirq.ControlledGate):
        ctrl_state = _cirq_values_to_ctrl_str(inner_gate.control_values, inner_gate)
        ctrl_layers.append((inner_gate.num_controls(), ctrl_state))
        inner_gate = inner_gate.sub_gate
    return inner_gate, ctrl_layers


def _apply_ctrl_layers(qrisp_op, ctrl_layers):
    """Put the control layers back onto a Qrisp operation, inside-out.

    _unpack_cirq_control_layers peels layers off from the outside in, so
    to rebuild the same gate in Qrisp we have to reverse the order:
    innermost control first, outermost last.

    Each layer becomes a ControlledOperation wrapping whatever we built so
    far.  For example, layers ``[(2, "10"), (1, "1")]`` produce::

        ControlledOperation(
            ControlledOperation(base, num_ctrl=2, ctrl_state="10"),
            num_ctrl=1, ctrl_state="1",
        )

    Parameters
    ----------
    qrisp_op : Operation
        The inner Qrisp operation we're wrapping.
    ctrl_layers : list[tuple[int, str]]
        The (num_controls, ctrl_state) layers, outermost first.

    Returns
    -------
    Operation
        The operation wrapped in its ControlledOperation layers.

    """
    for num_ctrl, ctrl_state in reversed(ctrl_layers):
        qrisp_op = ControlledOperation(
            base_operation=qrisp_op,
            num_ctrl_qubits=num_ctrl,
            ctrl_state=ctrl_state,
        )
    return qrisp_op


def convert_to_cirq(qrisp_circuit, cirq_qubits=None):
    """Convert a Qrisp QuantumCircuit into a Cirq Circuit.

    The conversion happens in three steps:

    1. **Transpile unknown gates.**  _transpile_to_known_gates recursively
       breaks down any gate whose name isn't in ``gate_map``.
    2. **Map the qubits.**  Each Qrisp qubit is paired one-to-one with a
       cirq qubit — either one you pass in, or a freshly created
       ``LineQubit``.
    3. **Convert instruction by instruction.**  For each Qrisp instruction:
       - global phase         -> ``cirq.GlobalPhaseGate``
       - gate with ``None``   -> decomposed through its ``.definition``
         in ``gate_map``          (recursively)
       - ControlledOperation  -> ``base.controlled()`` with the ctrl_state
       - standard gate        -> looked up in ``gate_map`` and parameterised

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The Qrisp circuit to convert.
    cirq_qubits : list[cirq.LineQubit], optional
        The Cirq qubits to map onto.  If omitted, we create fresh
        ``LineQubit`` instances automatically.

    Returns
    -------
    cirq.Circuit
        A cirq.Circuit equivalent to the input Qrisp circuit.

    Raises
    ------
    ImportError
        If Cirq isn't installed.
    ValueError
        If we meet a gate we can't handle.

    Notes
    -----
    We transpile all unknown gates up front, then check whether any *new*
    unknown gates appeared; we repeat until everything is known or
    transpilation stops making progress.

    One caveat about global phases: Qrisp's SXGate is defined as
    ``RX(pi/2)``, whereas cirq's ``X**0.5`` is ``exp(i*pi/4) * RX(pi/2)``.
    So converting ``sx``/``sx_dg`` introduces a global phase of
    ``exp(+-i*pi/4)`` relative to Qrisp's own unitary.  That's invisible
    in isolation, but it shows up in phase-sensitive comparisons (see also
    ``_build_cirq_gate``).

    """
    try:
        from cirq import Circuit, LineQubit  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("Cirq must be installed to be able to use the Qrisp to Cirq converter.") from exc

    from cirq import (  # noqa: PLC0415
        CNOT,
        CZ,
        SWAP,
        GlobalPhaseGate,
        H,
        I,
        M,
        R,
        S,
        T,
        X,
        XPowGate,
        Y,
        Z,
        ZPowGate,
        rx,
        ry,
        rz,
    )

    def _build_cirq_gate(name, params, cirq_gate, *, controlled_base=False):
        """Map a Qrisp gate name + params onto a concrete Cirq gate.

        Mostly this is a straight lookup, but a few names need special
        handling:
        - ``sx`` / ``sx_dg``  -> XPowGate with exponent 0.5 / -0.5
        - controlled ``cx`` / ``cz``  -> the *base* gate is X / Z (not
          CNOT / CZ), because cirq's ``.controlled()`` wraps a target gate.
        - ``p``  -> ZPowGate with exponent = params[0] / pi
        - parametrised gates (``rx``, ``ry``, ``rz``)  -> gate(*params)
        - gates without parameters  -> plain gate instance

        Global-phase caveat: Qrisp defines SXGate as ``RX(pi/2)``, while
        cirq's ``X**0.5`` carries an extra global phase ``exp(i*pi/4)``
        (cirq's convention is X^t = exp(i*pi*t/2) * RX(pi*t)).  Mapping
        ``sx`` to ``XPowGate(0.5)`` is therefore only correct up to a global
        phase — the same convention mismatch we handle on the way back in
        ``_conv_x`` / ``_needs_cirq_decomposition``.
        """
        known = {
            "sx": XPowGate(exponent=0.5),
            "sx_dg": XPowGate(exponent=-0.5),
        }
        if controlled_base:
            known["cx"] = X
            known["cz"] = Z
        if name in known:
            return known[name]
        if name == "p":
            return ZPowGate(exponent=params[0] / np.pi) if params else cirq_gate
        if params and name != "id":
            return cirq_gate(*params)
        return cirq_gate

    # gate_map: Qrisp gate name -> Cirq gate constructor / callable
    # A value of None means the gate has no direct Cirq equivalent and must
    # be decomposed via its .definition circuit instead.
    gate_map = {
        "cx": CNOT,
        "cy": Y,
        "cz": CZ,
        "swap": SWAP,
        "h": H,
        "x": X,
        "y": Y,
        "z": Z,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "s": S,
        "t": T,
        "s_dg": S**-1,
        "t_dg": T**-1,
        "measure": M,
        "reset": R,
        "id": I,
        "p": ZPowGate,
        "sx": XPowGate,
        "sx_dg": XPowGate,
        "gphase": None,
        "xxyy": None,
        "rxx": None,
        "rzz": None,
        # skip qubit allocation and deallocation ops in the converter
        "qb_alloc": None,
        "qb_dealloc": None,
    }

    # --- Stage 1: decompose unknown gates ---
    qrisp_circuit = _transpile_to_known_gates(qrisp_circuit, gate_map)

    # --- Stage 2: create Cirq circuit and qubit map ---
    cirq_circ = Circuit()

    if cirq_qubits is None:
        cirq_qubits = [LineQubit(i) for i in range(len(qrisp_circuit.qubits))]

    qubit_map = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = cirq_qubits[i]

    # --- Stage 3: convert each instruction ---
    for instr in qrisp_circuit.data:
        name = instr.op.name
        qubits = instr.qubits
        params = instr.op.params if hasattr(instr.op, "params") else []

        # Global phase: Qrisp applies it to a specific qubit; Cirq makes it
        # a zero-qubit gate. We drop the qubit argument.
        if name == "gphase":
            cirq_circ.append(GlobalPhaseGate(np.exp(1j * params[0]))())
            continue

        cirq_gate = gate_map[name]
        cirq_op_qubits = [qubit_map[q] for q in qubits]

        # Gate with no direct Cirq equivalent (None in gate_map):
        # decompose via its .definition circuit.
        if cirq_gate is None:
            if instr.op.definition:
                cirq_circ.append(convert_to_cirq(instr.op.definition, cirq_op_qubits))
                continue
            if name not in ("qb_alloc", "qb_dealloc"):
                raise ValueError(f"{name} gate has no Cirq equivalent and no definition to decompose.")
            continue

        # ControlledOperation: use cirq's base.controlled() API.
        # The base gate is looked up via _build_cirq_gate with
        # controlled_base=True so that cx/cz map to X/Z instead of CNOT/CZ.
        if isinstance(instr.op, ControlledOperation):
            base = _build_cirq_gate(name, params, cirq_gate, controlled_base=True)
            control_values = _ctrl_state_to_cirq_values(instr.op.ctrl_state)
            controlled = base.controlled(num_controls=len(instr.op.ctrl_state), control_values=control_values)
            cirq_circ.append(controlled(*cirq_op_qubits))
            continue

        # Standard gate: build and append
        gate = _build_cirq_gate(name, params, cirq_gate)
        cirq_circ.append(gate(*cirq_op_qubits))

    return cirq_circ


def _fractional_power_op(unitary_matrix, exponent, name, num_qubits):
    """Build a Qrisp Operation for a gate raised to a fractional power.

    Some Cirq gates (HPowGate, CXPowGate, SwapPowGate, ...) allow
    fractional exponents.  When the exponent doesn't land on a standard
    Qrisp gate, we compute the matrix power ourselves:

    1. Diagonalize the unitary.
    2. Replace eigenvalues close to -1 with ``exp(i * pi * exponent)`` —
       that -1 eigenvalue is the "flip" part of the gate.
    3. Reassemble the matrix from the modified eigenvalues.
    4. Return a Qrisp Operation carrying that unitary.

    This is our general fallback; the common fractional powers are handled
    by dedicated converter functions.

    Parameters
    ----------
    unitary_matrix : np.ndarray
        The unitary matrix we're raising to a power.
    exponent : float
        The power to raise it to.
    name : str
        Name for the resulting Qrisp Operation.
    num_qubits : int
        How many qubits the operation acts on.

    Returns
    -------
    Operation
        A Qrisp Operation with the computed unitary.

    """
    vals, vecs = np.linalg.eigh(unitary_matrix)
    vals_t = np.where(np.isclose(vals, -1), np.exp(1j * np.pi * exponent), 1.0)
    pow_mat = (vecs * vals_t) @ vecs.conj().T
    op = Operation(name=name, num_qubits=num_qubits)
    op.unitary = pow_mat
    return op


def _fractional_h_gate(exp):
    """Give us H^t by diagonalising the Hadamard matrix."""
    h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    return _fractional_power_op(h_mat, exp, "h_pow", 1)


# -----------------------------------------------------------------------
# Gate converter helpers  (Cirq -> Qrisp direction)
#
# Each _conv_* function receives a Cirq gate instance (the innermost
# non-controlled gate, after all ControlledGate layers have been peeled
# away) and returns the equivalent Qrisp Operation.
#
# Convention: integer exponents (mod 2 == 1) map directly to discrete
# Qrisp gates.  Fractional exponents are decomposed either into
# parametrised Qrisp rotations or computed via matrix diagonalisation.
# -----------------------------------------------------------------------


def _conv_h(inner_gate):
    """Convert cirq.HPowGate into an HGate, or H^t via diagonalisation."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_h_gate(inner_gate.exponent)
    return ops.HGate()


def _constant_converter(op_factory):
    """Wrap an op factory so it fits the converter dispatch signature.

    Some cirq gates map straight onto a fixed Qrisp op regardless of their
    details — CXPowGate -> CXGate, IdentityGate -> IDGate, and so on.  This
    takes the op factory (e.g. ``ops.CXGate``) and wraps it into a converter
    that accepts the inner gate and simply ignores it, so we can drop it
    straight into the _GATE_CONVERTERS registry like any other converter.

    Note we call ``op_factory()`` afresh on every invocation rather than
    caching a single op instance, so each converted instruction gets its own
    operation.
    """

    def converter(inner_gate):
        return op_factory()

    return converter


def _fractional_cz_gate(exp):
    """Build CZ^t as a CP(t*pi) gate."""
    return ops.CPGate(exp * np.pi)


def _conv_cz(inner_gate):
    """Convert cirq.CZPowGate into a CZGate, or CZ^t into CP(t*pi)."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_cz_gate(inner_gate.exponent)
    return ops.CZGate()


def _fractional_swap_gate(exp):
    """Build SWAP^t by diagonalising the SWAP matrix.

    Note the op is deliberately *not* named "swap": that name would collide
    with Qrisp's special-case Fredkin synthesis for controlled SWAP gates
    (see PTControlledOperation).
    """
    swap_mat = ops.SwapGate().get_unitary()
    return _fractional_power_op(swap_mat, exp, "swap_pow", 2)


def _conv_swap(inner_gate):
    """Convert cirq.SwapPowGate into a SwapGate, or SWAP^t fractionally."""
    if not np.isclose(inner_gate.exponent % 2, 1.0):
        return _fractional_swap_gate(inner_gate.exponent)
    return ops.SwapGate()


def _convert_pauli_pow(inner_gate, rotation_cls, rotation_fn, specials, fallback_fn):
    """Shared dispatcher for turning XPowGate / YPowGate / ZPowGate into Qrisp.

    Pauli power gates (X^t, Y^t, Z^t) come in a few flavours:
      - explicit rotation subclasses (cirq.Rx, cirq.Ry, cirq.Rz),
      - standard powers with discrete Qrisp equivalents (X, SX, S, T, ...),
      - generic fractional powers that fall back to a parametrised rotation
        (RX, RY, RZ, P).

    _conv_x, _conv_y and _conv_z each just supply their lookup tables and
    let us do the rest.  The order of business:

    1. If the gate is a rotation subclass (e.g. cirq.Rx), build the matching
       Qrisp rotation gate.
    2. Check ``exponent % 4`` against the ``specials`` list for known
       discrete gates.
    3. Otherwise fall back to the generic parametrised gate.

    Global-phase caveat: cirq's convention is X^t = exp(i*pi*t/2) * RX(pi*t)
    (likewise for Y), while Qrisp's RX/RY carry no such phase.  So the
    fractional fallback (step 3) only matches cirq up to a global phase.
    For an uncontrolled gate that's fine — a global phase is unobservable —
    but once you control it, that phase turns into a *relative* phase
    between the control branches.  That's why controlled fractional powers
    get decomposed by cirq first (see _needs_cirq_decomposition).

    Parameters
    ----------
    inner_gate : cirq.XPowGate | cirq.YPowGate | cirq.ZPowGate
        The innermost (non-controlled) gate to convert.
    rotation_cls : type
        The rotation subclass to look for (e.g. cirq.Rx).
    rotation_fn : Callable[[float], Operation]
        Factory for the Qrisp rotation gate (e.g. ops.RXGate).
    specials : list[tuple[float, Callable[[], Operation]]]
        (exponent_mod_4, factory) pairs for the standard powers.
    fallback_fn : Callable[[float], Operation]
        Factory for generic fractional powers (e.g. ops.PGate).

    Returns
    -------
    Operation

    """
    exp = inner_gate.exponent
    if isinstance(inner_gate, rotation_cls):
        return rotation_fn(exp * np.pi)
    for modulus, factory in specials:
        if np.isclose(exp % 4, modulus):
            return factory()
    return fallback_fn(exp * np.pi)


def _conv_x(inner_gate):
    """Convert cirq.XPowGate into X / SX / SX^dag / RX.

    We dispatch on ``exponent % 4``:

        exponent % 4 == 1.0  ->  XGate
        exponent % 4 == 3.0  ->  XGate   (X^3 = X^dagger = X)
        exponent % 4 == 2.0  ->  IDGate  (X^2 = I)
        exponent % 4 == 0.0  ->  IDGate
        exponent % 4 == 0.5  ->  SXGate
        exponent % 4 == 3.5  ->  SXDGGate
        otherwise            ->  RXGate(exp * pi)

    Anything that doesn't match a special becomes an RX rotation.
    """
    from cirq import Rx  # noqa: PLC0415

    return _convert_pauli_pow(
        inner_gate,
        Rx,
        ops.RXGate,
        [
            (1.0, ops.XGate),
            (3.0, ops.XGate),
            (2.0, ops.IDGate),
            (0.0, ops.IDGate),
            (0.5, ops.SXGate),
            (3.5, ops.SXDGGate),
        ],
        ops.RXGate,
    )


def _conv_y(inner_gate):
    """Convert cirq.YPowGate into Y / RY.

    We dispatch on ``exponent % 4``:

        exponent % 4 == 1.0  ->  YGate
        exponent % 4 == 3.0  ->  YGate   (Y^3 = Y^dagger = Y)
        exponent % 4 == 2.0  ->  IDGate  (Y^2 = I)
        exponent % 4 == 0.0  ->  IDGate
        otherwise            ->  RYGate(exp * pi)

    Anything that doesn't match a special becomes an RY rotation.
    """
    from cirq import Ry  # noqa: PLC0415

    return _convert_pauli_pow(
        inner_gate,
        Ry,
        ops.RYGate,
        [
            (1.0, ops.YGate),
            (3.0, ops.YGate),
            (2.0, ops.IDGate),
            (0.0, ops.IDGate),
        ],
        ops.RYGate,
    )


def _conv_z(inner_gate):
    """Convert cirq.ZPowGate into Z / S / S^dag / T / T^dag / P.

    We dispatch on ``exponent % 4``:

        exponent % 4 == 1.0   ->  ZGate
        exponent % 4 == 0.5   ->  SGate
        exponent % 4 == 3.5   ->  SGate.inverse()
        exponent % 4 == 0.25  ->  TGate
        exponent % 4 == 3.75  ->  TGate.inverse()
        otherwise             ->  PGate(exp * pi)

    Anything that doesn't match a special becomes a P phase gate.
    """
    from cirq import Rz  # noqa: PLC0415

    return _convert_pauli_pow(
        inner_gate,
        Rz,
        ops.RZGate,
        [
            (1.0, ops.ZGate),
            (0.5, ops.SGate),
            (3.5, lambda: ops.SGate().inverse()),
            (0.25, ops.TGate),
            (3.75, lambda: ops.TGate().inverse()),
        ],
        ops.PGate,
    )


def _conv_iswap(inner_gate):
    """Convert cirq.ISwapPowGate into an iSWAP (or iSWAP^dag).

    Only odd-integer exponents (mod 2 == 1) ever reach us here:
        exponent % 4 == 1.0  ->  iSWAP
        exponent % 4 == 3.0  ->  iSWAP^dag

    Fractional and even exponents get decomposed by cirq first (see
    _needs_cirq_decomposition).
    """
    op = ops.ISwapGate()
    if np.isclose(inner_gate.exponent % 4, 3.0):
        op = op.inverse()
    return op


_GATE_CONVERTERS = None


def _get_gate_converters():
    """Get (and lazily build) the cirq-type -> Qrisp converter registry.

    This is a list of ``(cirq gate type, converter function)`` pairs that
    the main loop of convert_from_cirq walks through (in order) to find a
    converter for the innermost gate, once all the ControlledGate layers
    have been unwrapped.

    The same list is reused by _decompose_unknown_ops (via _is_known) to
    decide whether an operation can be converted straight away or should be
    decomposed first.

    One detail worth knowing: the converters for the controlled variants
    (CXPowGate, CZPowGate, CCXPowGate, CCZPowGate) only ever see the *inner*
    gate, after control layers have been peeled off.  The control
    information is re-applied separately via _apply_ctrl_layers.

    We build this lazily to avoid circular imports and to make sure cirq is
    actually available before we touch the list.
    """
    global _GATE_CONVERTERS  # noqa: PLW0603
    if _GATE_CONVERTERS is None:
        _GATE_CONVERTERS = [
            (cirq.HPowGate, _conv_h),
            (cirq.CXPowGate, _constant_converter(ops.CXGate)),
            (cirq.CZPowGate, _conv_cz),
            (cirq.SwapPowGate, _conv_swap),
            (cirq.IdentityGate, _constant_converter(ops.IDGate)),
            (cirq.ResetChannel, _constant_converter(ops.Reset)),
            (cirq.XPowGate, _conv_x),
            (cirq.YPowGate, _conv_y),
            (cirq.ZPowGate, _conv_z),
            (cirq.ISwapPowGate, _conv_iswap),
            (cirq.CCXPowGate, _constant_converter(lambda: ops.MCXGate(control_amount=2))),
            (cirq.CCZPowGate, _constant_converter(lambda: ops.ZGate().control(2))),
        ]
    return _GATE_CONVERTERS


def _needs_cirq_decomposition(gate, controlled):
    """Should this power gate be decomposed by cirq before conversion?

    Cirq's convention is X^t = exp(i*pi*t/2) * RX(pi*t) (and the same for
    Y).  Qrisp's RX/RY rotations carry no such phase, so converting a
    fractional Pauli power directly is only phase-exact when the gate is
    *uncontrolled* — then the missing factor is just a global phase.  Once
    you control such a gate, that phase turns into a *relative* phase
    between the control branches and the conversion goes wrong.

    The inherently controlled power gates (CXPowGate, ISwapPowGate,
    CCXPowGate, CCZPowGate) are only exactly representable for odd-integer
    exponents; every other exponent (fractional or even) is delegated to
    ``cirq.decompose``, which reproduces cirq's phase convention exactly.

    The Z-based powers (ZPowGate, CZPowGate) are always phase-exact under
    this convention — the exp(i*pi*t/2) factor cancels against the rotation
    phase — so they're always converted directly.

    And the rotation subclasses (cirq.Rx, cirq.Ry, cirq.Rz) are exact by
    construction, so they must never be decomposed.

    Parameters
    ----------
    gate : cirq.Gate
        The innermost (non-controlled) gate to inspect.
    controlled : bool
        Whether the gate sits in a controlled context — either wrapped in a
        ControlledGate layer or the sub-operation of a ControlledOperation.

    Returns
    -------
    bool
        True if we should let cirq decompose it, False to convert directly.

    """
    if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
        return False

    exponent = getattr(gate, "exponent", None)
    if exponent is None:
        return False

    # Z-based powers are phase-exact and can always be converted directly.
    if isinstance(gate, (cirq.ZPowGate, cirq.CZPowGate)):
        return False

    odd_integer = np.isclose(exponent % 2, 1.0)

    if isinstance(
        gate,
        (cirq.CXPowGate, cirq.ISwapPowGate, cirq.CCXPowGate, cirq.CCZPowGate),
    ):
        return not odd_integer

    return controlled and not odd_integer


def _decompose_unknown_ops(cirq_circuit):
    """Keep decomposing unsupported Cirq ops until everything is convertible.

    This is the mirror image of _transpile_to_known_gates, for the reverse
    direction.  Plenty of Cirq gates (FSimGate, CSwapGate, fractional
    powers, ...) don't have a direct Qrisp equivalent but do implement
    Cirq's ``_decompose_`` protocol, so we use ``cirq.decompose`` to break
    them into simpler primitives.

    We loop because decomposition can go several levels deep: a gate may
    decompose into gates that are themselves unknown, which get decomposed
    on the next pass.

    A few edge cases worth knowing about:
      - ControlledOperation with forged/invalid control values: Cirq can't
        decompose these, so they stay unknown and we raise.
      - Custom gates without ``_decompose_``: same outcome.
      - Multi-valued controls (0 OR 1): Cirq *can* decompose these, so they
        now succeed where they previously raised.

    How do we know we're stuck?  If a full pass leaves the set of unknown
    gate type names unchanged, the leftovers can't be decomposed any
    further and we raise ValueError.

    """

    def _gate_of(op):
        """Safely pull the gate out of an operation.

        cirq.ControlledOperation doesn't expose its sub-operation's gate
        directly via ``op.gate`` — accessing it builds a ControlledGate
        that validates control values and can blow up on forged ones.  So
        we go through ``op.sub_operation.gate`` instead, just like the main
        conversion loop does.
        """
        if isinstance(op, cirq.ControlledOperation):
            return getattr(op.sub_operation, "gate", None)
        return getattr(op, "gate", None)

    def _outer_controls_known(op):
        """Check whether a ControlledOperation's outer controls are usable.

        The main converter runs _cirq_values_to_ctrl_str over the outer
        control values separately.  If that would raise, we can't convert
        the operation directly, so we treat it as unknown.
        """
        if isinstance(op, cirq.ControlledOperation):
            try:
                _cirq_values_to_ctrl_str(op.control_values, op)
            except ValueError:
                return False
        return True

    def _gate_name(op):
        """Give us a readable type name for an operation's gate."""
        gate = _gate_of(op)
        if gate is not None:
            return type(gate).__name__
        return type(op).__name__

    def _is_known(op):
        """Decide whether we can convert an operation directly.

        An operation counts as "known" when:
          1. its outer control values (if ControlledOperation) are valid,
          2. it has a ``.gate`` attribute that isn't None,
          3. the gate is a GlobalPhaseGate or MeasurementGate, or
          4. after unwrapping all ControlledGate layers, the innermost gate
             type matches an entry in _GATE_CONVERTERS *and* isn't a power
             gate that needs to be decomposed first (see
             _needs_cirq_decomposition).
        """
        if not _outer_controls_known(op):
            return False
        gate = _gate_of(op)
        if gate is None:
            return False
        if isinstance(gate, (cirq.GlobalPhaseGate, cirq.MeasurementGate)):
            return True
        try:
            inner_gate, ctrl_layers = _unpack_cirq_control_layers(gate)
        except ValueError:
            return False
        for gate_type, _ in _get_gate_converters():
            if isinstance(inner_gate, gate_type):
                controlled = isinstance(op, cirq.ControlledOperation) or len(ctrl_layers) > 0
                return not _needs_cirq_decomposition(inner_gate, controlled)
        return False

    while True:
        unknown_ops = [op for op in cirq_circuit.all_operations() if not _is_known(op)]
        if not unknown_ops:
            break

        unknown_identities = {_gate_name(op) for op in unknown_ops}

        all_ops = []
        for op in cirq_circuit.all_operations():
            if _is_known(op):
                all_ops.append(op)
            else:
                try:
                    all_ops.append(cirq.decompose(op))
                except (TypeError, ValueError):
                    all_ops.append(op)

        new_circ = cirq.Circuit(all_ops)

        new_unknown = [op for op in new_circ.all_operations() if not _is_known(op)]
        if not new_unknown:
            return new_circ

        new_unknown_identities = {_gate_name(op) for op in new_unknown}
        if new_unknown_identities == unknown_identities:
            names = ", ".join(sorted(unknown_identities))
            raise ValueError(
                f"The following Cirq gate(s) could not be decomposed "
                f"and are not supported by the Cirq to Qrisp converter: "
                f"{names}. Try decomposing them before calling from_cirq()."
            )

        cirq_circuit = new_circ

    return cirq_circuit


def convert_from_cirq(cirq_circuit):  # noqa: PLR0912, PLR0915
    """Convert a Cirq Circuit into a Qrisp QuantumCircuit.

    The conversion happens in three steps:

    1. **Decompose unknown ops.**  _decompose_unknown_ops leans on
       ``cirq.decompose`` to break unsupported gates into primitives.
    2. **Set up the qubits.**  We sort the Cirq qubits (they all have to be
       the same type, e.g. all LineQubit) and map them to fresh Qrisp
       qubits.
    3. **Convert operation by operation.**  Each Cirq operation becomes a
       Qrisp instruction:
       - cirq.ControlledOperation -> pull out the sub-op gate + outer controls
       - GlobalPhaseGate          -> GPhaseGate on the first qubit
       - MeasurementGate          -> qc.measure()
       - ControlledGate-wrapped op -> unwrap the layers, convert the inner
         gate, re-apply the layers via _apply_ctrl_layers
       - standard gate types      -> dispatch through _GATE_CONVERTERS

    **Control logic.**  We support both of Cirq's control styles:
      A) ``cirq.ControlledOperation(controls, sub_operation)`` — an outer
         wrapper whose controls come from ``op.controls``/``op.control_values``.
      B) ``cirq.GateOperation(ControlledGate(base, ...), qubits)`` — gate-level
         nesting that _unpack_cirq_control_layers unwraps for us.

    Both paths end up as the same Qrisp representation: nested
    ControlledOperation wrappers.

    Parameters
    ----------
    cirq_circuit : cirq.Circuit
        The Cirq Circuit to convert.

    Returns
    -------
    QuantumCircuit
        A Qrisp QuantumCircuit equivalent to the input Cirq circuit.

    Raises
    ------
    ImportError
        If Cirq isn't installed.
    ValueError
        If we meet a gate we can't handle.

    Notes
    -----
    One thing we don't preserve: measurement *key names*.  We let Cirq
    generate its own keys, so a round-trip (Qrisp -> Cirq -> Qrisp) loses
    the original classical-bit associations.

    """
    try:
        import cirq  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("Cirq must be installed to be able to use the Cirq to Qrisp converter.") from exc

    # --- Stage 1: decompose unsupported ops ---
    # This runs before qubit setup because decomposition may change the
    # set of qubits in use (e.g. multi-valued controls that Cirq drops).
    cirq_circuit = _decompose_unknown_ops(cirq_circuit)

    # --- Stage 2: qubit map ---
    # All Cirq qubits must be of the same type (e.g. all LineQubit) so
    # they can be sorted deterministically.  Mixed types raise TypeError,
    # which we catch and re-raise as ValueError with a clear message.
    all_qs = cirq_circuit.all_qubits()
    qc = QuantumCircuit(len(all_qs))

    try:
        cirq_qubits = sorted(all_qs)
    except TypeError as exc:
        types = {type(q).__name__ for q in cirq_circuit.all_qubits()}
        raise ValueError(
            f"Mixed qubit types {types} found in the circuit. The converter "
            f"requires all qubits to be of the same type (e.g. all LineQubit)."
        ) from exc
    qubit_map = {q: qc.qubits[i] for i, q in enumerate(cirq_qubits)}

    # --- Stage 3: convert each operation ---
    for op in cirq_circuit.all_operations():
        # ----------------------------------------------------------------
        # Path A: cirq.ControlledOperation (outer wrapper)
        #   Extract the sub-operation's gate directly (avoids the
        #   ControlledGate constructor which validates control values
        #   and can raise for forged/invalid values).
        #   Save orig_controls so the outer control layer is applied
        #   after the inner gate conversion.
        # ----------------------------------------------------------------
        if isinstance(op, cirq.ControlledOperation):
            sub_op = op.sub_operation
            gate = getattr(sub_op, "gate", None)
            if gate is None:
                raise ValueError(f"Controlled sub-operation {sub_op} is not supported by the Cirq to Qrisp converter.")
            orig_controls = (list(op.controls), list(sub_op.qubits), op.control_values)
        else:
            # Path B: standard GateOperation
            gate = getattr(op, "gate", None)
            if gate is None:
                raise ValueError(
                    f"Operation {op} without gate attribute is not supported by the Cirq to Qrisp converter."
                )
            orig_controls = None

        # ----------------------------------------------------------------
        # Special cases: GlobalPhaseGate and MeasurementGate
        # ----------------------------------------------------------------
        # Global phase: Cirq applies it to zero qubits; Qrisp requires
        # a qubit argument.  We attach it to the first available qubit.
        if isinstance(gate, cirq.GlobalPhaseGate):
            phi = np.angle(gate.coefficient)
            if cirq_qubits:
                qc.append(ops.GPhaseGate(phi), [qubit_map[cirq_qubits[0]]])
            continue

        # Measurement: use qc.measure() for automatic classical-bit
        # allocation.  Single-qubit and multi-qubit variants are handled.
        if isinstance(gate, cirq.MeasurementGate):
            qrisp_qubits = [qubit_map[q] for q in op.qubits]
            if len(qrisp_qubits) == 1:
                qc.measure(qrisp_qubits[0])
            else:
                qc.measure(qrisp_qubits)
            continue

        # ----------------------------------------------------------------
        # ControlledGate unwrapping (Path B inner nesting)
        #   Peel off all ControlledGate layers to get the innermost gate
        #   and a list of (num_controls, ctrl_state) tuples.
        # ----------------------------------------------------------------
        inner_gate, ctrl_layers = _unpack_cirq_control_layers(gate)

        # ----------------------------------------------------------------
        # Dispatch: find the matching converter and build the Qrisp op
        # ----------------------------------------------------------------
        qrisp_op = None
        for gate_type, converter in _get_gate_converters():
            if isinstance(inner_gate, gate_type):
                qrisp_op = converter(inner_gate)
                break
        if qrisp_op is None:
            raise ValueError(f"Gate {gate} is not supported by the Cirq to Qrisp converter.")

        # Re-apply ControlledGate layers inside-out
        qrisp_op = _apply_ctrl_layers(qrisp_op, ctrl_layers)

        # ----------------------------------------------------------------
        # Apply outer ControlledOperation wrapper (Path A)
        # ----------------------------------------------------------------
        if orig_controls is not None:
            controls, sub_qubits, cv = orig_controls
            ctrl_state = _cirq_values_to_ctrl_str(cv, op)
            qrisp_op = ControlledOperation(
                base_operation=qrisp_op,
                num_ctrl_qubits=len(controls),
                ctrl_state=ctrl_state,
            )
            qrisp_qubits = [qubit_map[q] for q in controls + sub_qubits]
        else:
            qrisp_qubits = [qubit_map[q] for q in op.qubits]

        qc.append(qrisp_op, qrisp_qubits)

    return qc
