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

from qrisp import ControlledOperation

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
    "p": "Rz",
    "u1": "Rz",
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
}


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
        ``definition``.

    """
    from pytket import OpType
    from pytket.circuit import CircBox

    if op.name in _GATE_OPTYPES:
        return getattr(OpType, _GATE_OPTYPES[op.name])

    if op.definition:
        # Composite gate: wrap its definition as an abstract CircBox.
        tket_definition = pytket_converter(op.definition, boxFlag=True)
        if tket_definition.n_qubits != op.num_qubits:  # pragma: no cover
            raise ValueError("Converted definition of '" + str(op.name) + "' has a mismatched qubit count")

        return CircBox(tket_definition)

    raise ValueError("Could not convert operation " + str(op.name) + " to PyTket")


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

    """
    try:
        from pytket import Circuit, Qubit
        from pytket.circuit import CircBox
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

        params = list(op.params)
        # Prepare qubits
        qubit_list = [qubit_dic[qubit.identifier] for qubit in qc.data[i].qubits]
        clbit_list = [clbit_dic[clbit.identifier] for clbit in qc.data[i].clbits]

        # pytket expects rotation angles in half-turns (pi multiples).
        if op.name in ["cp", "p", "rx", "rz", "ry", "rxx", "rzz", "ryy", "u1", "u3"]:
            params = [index / np.pi for index in params]

        if op.name in ["qb_alloc", "qb_dealloc"]:
            continue

        if op.name == "gphase":
            # Global phase: pytket tracks it circuit-wide via add_phase (half-turns).
            tket_qc.add_phase(params[0] / np.pi)
            continue

        if op.name in ["sx", "sx_dg", "id"]:
            # Qrisp attaches spurious params to these; pytket takes none.
            params = []

        if op.name not in _GATE_OPTYPES and issubclass(op.__class__, ControlledOperation):
            # A composite controlled operation (e.g. mcx) is emitted as an abstract
            # CircBox of its definition.
            base_name = op.base_operation.name
            if len(base_name) == 1:
                base_name = base_name.upper()

            tket_definition = pytket_converter(op.definition, boxFlag=True)
            tket_definition.name = base_name
            tket_ins = CircBox(tket_definition)
        else:
            tket_ins = create_tket_instruction(op)

        if isinstance(tket_ins, CircBox):
            tket_qc.add_circbox(tket_ins, qubit_list)
        elif clbit_list:
            tket_qc.add_gate(tket_ins, params, qubit_list + clbit_list)
        else:
            tket_qc.add_gate(tket_ins, params, qubit_list)

    return tket_qc
