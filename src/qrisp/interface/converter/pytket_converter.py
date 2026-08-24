"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qrisp import ClControlledOperation, ControlledOperation

if TYPE_CHECKING:
    from pytket import Circuit, OpType
    from pytket.circuit import CircBox

    from qrisp.circuit import Operation, QuantumCircuit

# Maps a Qrisp operation name to the corresponding pytket ``OpType`` attribute
# name. Values are strings (resolved via ``getattr`` inside the converter) so this
# table can live at module level without importing pytket eagerly.
_GATE_OPTYPES = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "s_dg": "Sdg",
    "t": "T",
    "t_dg": "Tdg",
    "sx": "SX",
    "sx_dg": "SXdg",
    "id": "noop",
    "rx": "Rx",
    "ry": "Ry",
    "rz": "Rz",
    "p": "U1",
    "u1": "U1",
    "u3": "U3",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "cp": "CU1",
    "rxx": "XXPhase",
    "rzz": "ZZPhase",
    "ryy": "YYPhase",
    "swap": "SWAP",
    "measure": "Measure",
    "reset": "Reset",
}

# Gates whose Qrisp angle params are in radians; pytket expects half-turns (pi
# multiples), so their params are divided by pi before emission.
_PI_UNIT_GATES = frozenset({"cp", "p", "rx", "rz", "ry", "rxx", "rzz", "ryy", "u1", "u3"})


def create_tket_instruction(op: Operation) -> OpType | CircBox:
    """Map a single Qrisp operation to its pytket instruction.

    Parameters
    ----------
    op : Operation
        The Qrisp operation to convert.

    Returns
    -------
    pytket.OpType or pytket.circuit.CircBox
        An elementary ``OpType`` for a known gate, or a ``CircBox`` wrapping the
        operation's definition for a composite gate.

    Raises
    ------
    ValueError
        If the operation is neither a known gate nor decomposable via a
        ``definition`` (e.g. ``r``/RGate).

    """
    from pytket import OpType
    from pytket.circuit import CircBox

    if op.name in _GATE_OPTYPES:
        return getattr(OpType, _GATE_OPTYPES[op.name])

    if op.definition:
        # Composite gate (including multi-controlled ops): wrap its definition as
        # an abstract CircBox. This is the single place a definition becomes a box.
        tket_definition = pytket_converter(op.definition, boxFlag=True)
        if tket_definition.n_qubits != op.num_qubits:  # pragma: no cover
            raise ValueError("Converted definition of '" + str(op.name) + "' has a mismatched qubit count")

        if isinstance(op, ControlledOperation):
            # Label the box after the controlled gate's base operation for
            # readability (e.g. "X" for an mcx).
            base_name = op.base_operation.name
            tket_definition.name = base_name.upper() if len(base_name) == 1 else base_name

        return CircBox(tket_definition)

    raise ValueError("Could not convert operation " + str(op.name) + " to PyTket")


def _instruction_params(op: Operation) -> list:
    """Return the pytket params for an operation.

    pytket expects rotation angles in half-turns (pi multiples), so angles of the
    gates listed in ``_PI_UNIT_GATES`` are divided by pi. Some gates (``sx``,
    ``sx_dg``, ``id``) carry spurious Qrisp params that pytket rejects.
    """
    if op.name in ["sx", "sx_dg", "id"]:
        return []

    params = list(op.params)
    if op.name in _PI_UNIT_GATES:
        params = [angle / np.pi for angle in params]

    return params


def _control_state_flip_qubits(op: Operation, qubit_list: list) -> list:
    """Return qubits that need an X conjugation to realise ``op``'s control state.

    The gate lookup table only encodes |1..1>-controlled optypes. For an
    elementary controlled gate with |0> controls, the controlled qubits are
    conjugated with X so the intended control state is realised.
    """
    if op.name in _GATE_OPTYPES and isinstance(op, ControlledOperation) and "0" in op.ctrl_state:
        return [qubit_list[i] for i, bit in enumerate(op.ctrl_state) if bit == "0"]

    return []


def _add_tket_instruction(tket_qc: Circuit, op: Operation, qubit_list: list, clbit_list: list) -> None:
    """Add a single Qrisp operation to an existing pytket circuit."""
    from pytket import OpType
    from pytket.circuit import CircBox

    params = _instruction_params(op)

    if op.name in ["qb_alloc", "qb_dealloc"]:
        return

    if op.name == "gphase":
        # Global phase: pytket tracks it circuit-wide via add_phase (half-turns).
        tket_qc.add_phase(params[0] / np.pi)
        return

    if op.name == "barrier":
        tket_qc.add_barrier(qubit_list)
        return

    if isinstance(op, ClControlledOperation):
        # Classically-controlled gate: emit the base op as a pytket conditional
        # gated on the classical bits it reads.
        tket_qc.add_gate(
            create_tket_instruction(op.base_op),
            _instruction_params(op.base_op),
            qubit_list,
            condition_bits=clbit_list,
            condition_value=int(op.ctrl_state, 2),
        )
        return

    tket_ins = create_tket_instruction(op)
    ctrl_flip_qubits = _control_state_flip_qubits(op, qubit_list)

    for flip_qubit in ctrl_flip_qubits:
        tket_qc.add_gate(OpType.X, [], [flip_qubit])

    if isinstance(tket_ins, CircBox):
        tket_qc.add_circbox(tket_ins, qubit_list)
    elif clbit_list:
        tket_qc.add_gate(tket_ins, params, qubit_list + clbit_list)
    else:
        tket_qc.add_gate(tket_ins, params, qubit_list)

    for flip_qubit in ctrl_flip_qubits:
        tket_qc.add_gate(OpType.X, [], [flip_qubit])


def pytket_converter(qc: QuantumCircuit, boxFlag: bool = False) -> Circuit:
    """Convert a Qrisp QuantumCircuit to a pytket Circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The Qrisp circuit to convert.
    boxFlag : bool, optional
        If True, build the circuit for use as an abstract ``CircBox``: qubits are
        indexed positionally rather than by named identifier. Used internally when
        recursing into composite/controlled gate definitions. The default is False.

    Returns
    -------
    pytket.Circuit
        A pytket Circuit equivalent to the input Qrisp circuit.

    Raises
    ------
    ImportError
        If pytket is not installed.
    ValueError
        If the circuit contains an operation with no pytket equivalent and no
        decomposable ``definition`` (e.g. ``r``/RGate).

    """
    try:
        from pytket import Circuit, Qubit
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError("PyTket must be installed to be able to use the Qrisp to PyTket converter.") from exc

    # This dict gives the pytket qubits/clbits when presented with their identifier.
    qubit_dic = {}
    tket_qc = Circuit()
    tketQubits = []
    for i in range(len(qc.qubits)):
        # add a named qubit
        tketQubits.append(Qubit(name=str(qc.qubits[i].identifier), index=i))
        qubit_dic[qc.qubits[i].identifier] = tketQubits[-1]
        tket_qc.add_qubit(tketQubits[-1])

    # Flag for alternative qubit assignment if we try to create an abstract CircBox
    if boxFlag:
        tket_qc = Circuit(len(qc.qubits))
        qubit_dic = dict()
        for i in range(len(qc.qubits)):
            qubit_dic[qc.qubits[i].identifier] = i

    clbit_dic = {}
    # Add Clbits
    if len(qc.clbits):
        c_reg = tket_qc.add_c_register(name="creg_std", size=len(qc.clbits))
    for i in range(len(qc.clbits)):
        clbit_dic[qc.clbits[i].identifier] = c_reg[i]
        # NOTE: all clbits share a single register ("creg_std"). An older constraint
        # (Aer / QASM only accepting one classical register) no longer applies --
        # verified that multiple registers now run on AerBackend -- so this is just a
        # simplification. A per-bit alternative (named Bits) is kept for reference:
        # tketClbits.append(Bit(name=str(qc.clbits[i].identifier)))
        # clbit_dic[qc.clbits[i].identifier] = tketClbits[-1]
        # tket_qc.add_bit(tketClbits[-1])

    for i in range(len(qc.data)):
        op = qc.data[i].op
        qubit_list = [qubit_dic[qubit.identifier] for qubit in qc.data[i].qubits]
        clbit_list = [clbit_dic[clbit.identifier] for clbit in qc.data[i].clbits]
        _add_tket_instruction(tket_qc, op, qubit_list, clbit_list)

    return tket_qc
