# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""This module provides functionalities for converting between Qrisp and PyZX quantum circuits."""

from __future__ import annotations

import copy
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from qrisp import HGate, QuantumCircuit, RYGate, RZGate, SwapGate, SXGate, ZGate, u3Gate

if TYPE_CHECKING:
    from pyzx import Circuit
    from pyzx.circuit.gates import Gate

    from qrisp import Clbit, Operation, Qubit


def _phase(angle: float, max_denominator: int = 1024, tol: float = 1e-9) -> Fraction:
    """Convert an angle in radians to a PyZX phase in units of pi.

    PyZX simplification rules key on small rational phases such as ``1/4`` or
    ``1/2``. Since most angles are not exactly representable as binary floats,
    the result is snapped to a small denominator whenever the approximation is
    within ``tol``. Angles that are not close to such a rational are kept as
    their exact float value.

    Parameters
    ----------
    angle : float
        The angle in radians.
    max_denominator : int, optional
        The largest denominator allowed when snapping to a rational phase.
        The default is 1024.
    tol : float, optional
        The maximum absolute error (in units of pi) for accepting a snapped
        fraction. The default is 1e-9.

    Returns
    -------
    Fraction
        The angle expressed in units of pi.

    Examples
    --------
    >>> _phase(np.pi / 3)
    Fraction(1, 3)
    >>> _phase(np.pi / 4) == Fraction(1, 4)
    True

    """
    x = float(angle) / np.pi
    fraction = Fraction(x).limit_denominator(max_denominator)
    return fraction if abs(float(fraction) - x) < tol else Fraction(x)


def _ungadget(gate: Gate) -> Gate:
    """Return a copy of ``gate`` with the phase-gadget flag disabled.

    A copy is made for parity-phase gadgets so that converting a PyZX circuit
    does not mutate the caller's gate objects.

    Parameters
    ----------
    gate : pyzx.circuit.gates.Gate
        The gate to inspect.

    Returns
    -------
    pyzx.circuit.gates.Gate
        The original gate, or an un-gadgetted copy if it is a parity phase.

    """
    if gate.name != "ParityPhase":
        return gate
    result = copy.copy(gate)
    result.as_gadget = False
    return result


def _transpile(qrisp_circuit: QuantumCircuit, gate_map: dict[str, str | None]) -> QuantumCircuit:
    """Transpile a circuit until only operations known to ``gate_map`` remain.

    Unknown operations are repeatedly transpiled until no further progress can
    be made. The code was adapted from the Cirq converter.

    Parameters
    ----------
    qrisp_circuit : QuantumCircuit
        The circuit to transpile.
    gate_map : dict[str, str | None]
        Mapping from supported Qrisp operation names to their PyZX gate names.

    Returns
    -------
    QuantumCircuit
        A circuit containing only operations whose names occur in ``gate_map``.

    Raises
    ------
    ValueError
        If an unknown operation cannot be transpiled or if transpilation makes
        no further progress.

    """

    def _unknown_names(circuit: QuantumCircuit) -> set[str]:
        """Return the names of all operations that are not keys of ``gate_map``."""
        return {instr.op.name for instr in circuit.data if instr.op.name not in gate_map}

    while True:
        unknown = _unknown_names(qrisp_circuit)
        if not unknown:
            break

        def _transpile_predicate(op: Operation, _unknown: set[str] = unknown) -> bool:
            """Return whether ``op`` belongs to the set of unknown operations."""
            return op.name in _unknown

        try:
            transpiled = qrisp_circuit.transpile(transpile_predicate=_transpile_predicate)
        except Exception as exc:
            raise ValueError(
                f"Gates {unknown} could not be transpiled and are not supported by the Qrisp to PyZX converter."
            ) from exc

        new_unknown = _unknown_names(transpiled)
        if new_unknown == unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"The following gates could not be decomposed into elementary "
                f"instructions: {names}. Try transpiling the circuit with "
                f"Qrisp's transpile() method before calling to_pyzx(), or "
                f"use only gates supported natively by the converter."
            )

        qrisp_circuit = transpiled

    return qrisp_circuit


def convert_to_pyzx(qrisp_circuit: QuantumCircuit) -> Circuit:
    """Convert a Qrisp QuantumCircuit to a PyZX Circuit.

    Parameters
    ----------
    qrisp_circuit : qrisp.QuantumCircuit
        The Qrisp QuantumCircuit to convert.

    Returns
    -------
    pyzx.Circuit
        A pyzx.Circuit equivalent to the input Qrisp circuit.

    Raises
    ------
    ImportError
        If PyZX is not installed.
    ValueError
        If a gate is not supported by the converter.

    Notes
    -----
    Unknown gates are decomposed by Qrisp's transpiler before conversion.
    The converter transpiles all unknown gates together, then checks if any
    new unknown gates appeared.  This repeats until all gates are known
    or transpilation makes no progress.

    Examples
    --------
    Convert a Qrisp Bell-state circuit to PyZX and back:

    >>> from qrisp import QuantumCircuit, Qubit
    >>> from qrisp.interface import convert_from_pyzx, convert_to_pyzx
    >>> qc = QuantumCircuit()
    >>> _ = qc.add_qubit(Qubit("q0"))
    >>> _ = qc.add_qubit(Qubit("q1"))
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> print(qc)  # doctest: +NORMALIZE_WHITESPACE
        ┌───┐
    q0: ┤ H ├──■──
        └───┘┌─┴─┐
    q1: ─────┤ X ├
             └───┘
    >>> pyzx_circuit = convert_to_pyzx(qc)
    >>> print(pyzx_circuit)
    Circuit(2 qubits, 0 bits, 2 gates)
    >>> [(gate.name, gate.target) for gate in pyzx_circuit.gates]
    [('HAD', 0), ('CNOT', 1)]
    >>> qc_roundtrip = convert_from_pyzx(pyzx_circuit)
    >>> [(ins.op.name, [qc_roundtrip.qubits.index(q) for q in ins.qubits]) for ins in qc_roundtrip.data]
    [('h', [0]), ('cx', [0, 1])]

    Measurement destinations are preserved as PyZX result bits:

    >>> qc = QuantumCircuit(1, 1)
    >>> qc.measure(0, 0)
    >>> convert_to_pyzx(qc).gates[0].result_bit
    0

    Gates without a native PyZX counterpart are transpiled automatically:

    >>> qc = QuantumCircuit(3)
    >>> qc.ccx(0, 1, 2)
    >>> len(convert_to_pyzx(qc).gates) > 0
    True

    """
    try:
        from pyzx import Circuit
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("PyZX must be installed to be able to use the Qrisp to PyZX converter.") from exc

    gate_map: dict[str, str | None] = {
        "cx": "CNOT",
        "cy": "CY",
        "cz": "CZ",
        "swap": "SWAP",
        "h": "HAD",
        "x": "NOT",
        "y": "Y",
        "z": "Z",
        "rx": "XPhase",
        "ry": "YPhase",
        "rz": "ZPhase",
        "u1": "ZPhase",
        "u3": "U3",
        "prx": "U3",
        "s": "S",
        "t": "T",
        "s_dg": None,
        "t_dg": None,
        "p": None,
        "sx": "SX",
        "sx_dg": None,
        "xxyy": None,
        "rxx": "RXX",
        "rzz": "RZZ",
        "measure": "Measurement",
        "reset": "Reset",
        # skip qubit allocation and deallocation ops in the converter, as well as identity and global phases
        "qb_alloc": None,
        "qb_dealloc": None,
        "gphase": None,
        "id": None,
        "barrier": None,
    }

    qrisp_circuit = _transpile(qrisp_circuit, gate_map)

    num_qubits = qrisp_circuit.num_qubits()
    num_clbits = len(qrisp_circuit.clbits)
    pyzx_circuit = Circuit(qubit_amount=num_qubits, bit_amount=num_clbits)

    qubit_map: dict[Qubit, int] = {}
    for i, q in enumerate(qrisp_circuit.qubits):
        qubit_map[q] = i
    clbit_map: dict[Clbit, int] = {}
    for i, c in enumerate(qrisp_circuit.clbits):
        clbit_map[c] = i

    for instr in qrisp_circuit.data:
        name = instr.op.name
        qubits = instr.qubits
        clbits = instr.clbits
        params = instr.op.params if hasattr(instr.op, "params") else []

        pyxz_gate = gate_map[name]
        pyxz_op_qubits = [qubit_map[q] for q in qubits]
        pyxz_op_clbits = [clbit_map[c] for c in clbits]

        special_gate_actions: dict[str, Callable[[], None]] = {
            **dict.fromkeys(["id", "barrier", "gphase", "qb_alloc", "qb_dealloc"], lambda: None),
            "s_dg": lambda: pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, _phase(-np.pi / 2)),
            "t_dg": lambda: pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, _phase(-np.pi / 4)),
            "p": lambda: pyzx_circuit.add_gate("U3", *pyxz_op_qubits, 0, 0, _phase(params[0])),
            "sx_dg": lambda: pyzx_circuit.add_gate("XPhase", *pyxz_op_qubits, _phase(-np.pi / 2)),
        }

        # Gate with no direct PyZX equivalent.
        if pyxz_gate is None:
            if name in special_gate_actions:
                special_gate_actions[name]()
            # decompose via its .definition circuit (e.g. xxyy)
            elif instr.op.definition:
                pyzx_circuit.add_circuit(convert_to_pyzx(instr.op.definition), mask=pyxz_op_qubits)
            else:
                raise ValueError(f"{name} gate has no PyZX equivalent and no definition to decompose.")
            continue

        if name == "measure" and clbits:
            pyzx_circuit.add_gate("Measurement", pyxz_op_qubits[0], pyxz_op_clbits[0])
        elif params:
            pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits, *[_phase(p) for p in params])
        else:
            pyzx_circuit.add_gate(pyxz_gate, *pyxz_op_qubits)

    return pyzx_circuit


def convert_from_pyzx(pyzx_circuit: Circuit) -> QuantumCircuit:
    """Convert a PyZX Circuit to a Qrisp QuantumCircuit.

    Parameters
    ----------
    pyzx_circuit : pyzx.Circuit
        The PyZX Circuit to convert.

    Returns
    -------
    qrisp.QuantumCircuit
        A qrisp.QuantumCircuit equivalent to the input PyZX circuit.

    Raises
    ------
    ValueError
        If a gate is not supported by the converter.

    Notes
    -----
    Gates that exist in PyZX but not in Qrisp are either substituted directly if they
    have a straightforward equivalent (which applies to U2 and several controlled
    gates), or PyZX's ``to_basic_gates()`` method is used to decompose them.

    Examples
    --------
    Convert a PyZX Bell-state circuit to Qrisp and back:

    >>> from pyzx import Circuit
    >>> from qrisp.interface import convert_from_pyzx, convert_to_pyzx
    >>> pyzx_circuit = Circuit(2)
    >>> pyzx_circuit.add_gate("HAD", 0)
    >>> pyzx_circuit.add_gate("CNOT", 0, 1)
    >>> print(pyzx_circuit)
    Circuit(2 qubits, 0 bits, 2 gates)
    >>> qc = convert_from_pyzx(pyzx_circuit)
    >>> for i, qubit in enumerate(qc.qubits):
    ...     qubit.identifier = f"q{i}"
    >>> print(qc)  # doctest: +NORMALIZE_WHITESPACE
        ┌───┐
    q0: ┤ H ├──■──
        └───┘┌─┴─┐
    q1: ─────┤ X ├
             └───┘
    >>> pyzx_roundtrip = convert_to_pyzx(qc)
    >>> print(pyzx_roundtrip)
    Circuit(2 qubits, 0 bits, 2 gates)
    >>> [(gate.name, gate.target) for gate in pyzx_roundtrip.gates]
    [('HAD', 0), ('CNOT', 1)]

    Measurement result bits are mapped to the corresponding classical bits:

    >>> pyzx_circuit = Circuit(1, bit_amount=1)
    >>> pyzx_circuit.add_gate("Measurement", 0, result_bit=0)
    >>> qc = convert_from_pyzx(pyzx_circuit)
    >>> [(ins.op.name, qc.clbits.index(ins.clbits[0])) for ins in qc.data]
    [('measure', 0)]

    Gates without a direct Qrisp equivalent are decomposed by PyZX first:

    >>> from fractions import Fraction
    >>> pyzx_circuit = Circuit(3)
    >>> pyzx_circuit.add_gate("ParityPhase", Fraction(1, 2), 0, 1, 2)
    >>> [ins.op.name for ins in convert_from_pyzx(pyzx_circuit).data]
    ['cx', 'cx', 'rz', 'cx', 'cx']

    """
    qc = QuantumCircuit(pyzx_circuit.qubits, pyzx_circuit.bits)

    def _CSX_gate(x: Qubit, y: Qubit) -> None:
        """Append a controlled SX gate acting on ``x`` (control) and ``y`` (target)."""
        qc.append(SXGate().control(), [x, y])

    gate_map: dict[str, Callable[..., None] | None] = {
        # single-qubit gates
        "NOT": qc.x,
        "Y": qc.y,
        "Z": qc.z,
        "HAD": qc.h,
        "XPhase": qc.rx,
        "YPhase": qc.ry,
        "ZPhase": qc.rz,
        "U2": partial(qc.u3, np.pi / 2),
        "U3": qc.u3,
        "SX": lambda adj, x: qc.sx_dg(x) if adj else qc.sx(x),
        "S": lambda adj, x: qc.s_dg(x) if adj else qc.s(x),
        "T": lambda adj, x: qc.t_dg(x) if adj else qc.t(x),
        # multi-qubits gates
        "CNOT": qc.cx,
        "CY": qc.cy,
        "CZ": qc.cz,
        "CRX": qc.crx,
        "CRY": lambda phase, x, y: qc.append(RYGate(phase).control(), [x, y]),
        "CRZ": lambda phase, x, y: qc.append(RZGate(phase).control(), [x, y]),
        "CSX": _CSX_gate,
        "CPhase": qc.cp,
        "ParityPhase": None,
        "XCX": None,
        "SWAP": qc.swap,
        "CSWAP": lambda x, y, z: qc.append(SwapGate().control(), [x, y, z]),
        "CHAD": lambda x, y: qc.append(HGate().control(), [x, y]),
        "Tof": qc.ccx,
        "CCZ": lambda x, y, z: qc.append(ZGate().control(2), [x, y, z]),
        "CU3": lambda theta, phi, lam, x, y: qc.append(u3Gate(theta, phi, lam).control(), [x, y]),
        "CU": None,
        "RZZ": qc.rzz,
        "RXX": qc.rxx,
        "FSim": None,
        # non-unitary operations
        "Measurement": qc.measure,
        "Reset": qc.reset,
        "InitAncilla": None,
        "PostSelect": None,
        "DiscardBit": None,
        "ConditionalGate": None,
    }

    def add_gate(gate: Gate) -> None:
        """Dispatch a single PyZX gate to its Qrisp counterpart."""

        def _measurement(gate: Gate) -> None:
            """Append a measurement that stores its outcome in ``gate.result_bit``."""
            gate_map[gate.name](gate.target, gate.result_bit)

        def _single_qubit_parameter_free(gate: Gate) -> None:
            """Append a parameter-free single-qubit gate or non-unitary operation."""
            gate_map[gate.name](gate.target)

        def _single_qubit_with_adjoint_version(gate: Gate) -> None:
            """Append a single-qubit gate, honoring its ``adjoint`` flag."""
            gate_map[gate.name](gate.adjoint, gate.target)

        def _single_qubit_one_parameter(gate: Gate) -> None:
            """Append a single-qubit gate with one phase parameter (in units of pi)."""
            gate_map[gate.name](float(gate.phase) * np.pi, gate.target)

        def _single_qubit_multi_parameter(gate: Gate) -> None:
            """Append a single-qubit gate with multiple phase parameters (in units of pi)."""
            gate_map[gate.name](*[float(p) * np.pi for p in gate.phases], gate.target)

        def _two_qubit_parameter_free(gate: Gate) -> None:
            """Append a parameter-free two-qubit gate."""
            gate_map[gate.name](gate.control, gate.target)

        def _two_qubit_one_parameter(gate: Gate) -> None:
            """Append a two-qubit gate with one phase parameter (in units of pi)."""
            gate_map[gate.name](float(gate.phase) * np.pi, gate.control, gate.target)

        def _two_qubit_multi_parameter(gate: Gate) -> None:
            """Append a two-qubit gate with multiple phase parameters (in units of pi)."""
            gate_map[gate.name](*[float(p) * np.pi for p in gate.phases], gate.control, gate.target)

        def _multi_qubit_parameter_free(gate: Gate) -> None:
            """Append a parameter-free three-qubit gate."""
            gate_map[gate.name](gate.ctrl1, gate.ctrl2, gate.target)

        function_map: dict[str, Callable[[Gate], None]] = {
            **dict.fromkeys(["Measurement"], _measurement),
            **dict.fromkeys(["NOT", "Y", "Z", "HAD", "Reset"], _single_qubit_parameter_free),
            **dict.fromkeys(["SX", "S", "T"], _single_qubit_with_adjoint_version),
            **dict.fromkeys(["XPhase", "YPhase", "ZPhase"], _single_qubit_one_parameter),
            **dict.fromkeys(["U2", "U3"], _single_qubit_multi_parameter),
            **dict.fromkeys(["CNOT", "CY", "CZ", "CSX", "SWAP", "CHAD"], _two_qubit_parameter_free),
            **dict.fromkeys(["CRX", "CRY", "CRZ", "CPhase", "RZZ", "RXX"], _two_qubit_one_parameter),
            **dict.fromkeys(["CU3"], _two_qubit_multi_parameter),
            **dict.fromkeys(["CSWAP", "Tof", "CCZ"], _multi_qubit_parameter_free),
        }

        function_map[gate.name](gate)

    for gate in pyzx_circuit.gates:
        if gate.name in gate_map:
            if gate_map[gate.name] is not None:
                add_gate(gate)
            else:
                # Try with PyZX's basic gate decomposition.
                for _gate in _ungadget(gate).to_basic_gates():
                    if _gate.name in gate_map and gate_map[_gate.name] is not None:
                        add_gate(_gate)
                    else:
                        raise ValueError(f"{_gate.name} gate has no Qrisp equivalent and cannot be decomposed either.")
        else:
            raise ValueError(f"{gate.name} of PyZX is unknown and thus cannot be converted to Qrisp.")

    return qc
