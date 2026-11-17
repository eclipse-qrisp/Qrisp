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

"""Bidirectional converter between Qrisp and Qiskit quantum circuits."""

from functools import reduce

import numpy as np
import qiskit.circuit.library.standard_gates as qsk_gates
from qiskit import QiskitError
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit.circuit import (
    Barrier,
    ControlFlowOp,
    ControlledGate,
    IfElseOp,
    Measure,
    Parameter,
    ParameterExpression,
    Reset,
)
from qiskit.circuit import Clbit as QiskitClbit
from qiskit.circuit.classical import expr
from sympy import Expr, lambdify, sympify

from qrisp import Clbit, QuantumCircuit, Qubit
from qrisp.circuit import ClControlledOperation, ControlledOperation, Instruction, PRXGate, U3Gate
from qrisp.circuit.standard_operations import op_list
from qrisp.misc import bin_rep

GLOBAL_PHASE_TOLERANCE = 1e-10


def convert_to_qiskit(qc: QuantumCircuit, transpile=False):
    """Convert a Qrisp QuantumCircuit to a Qiskit QuantumCircuit.

    Each Qrisp qubit and clbit becomes its own single-bit Qiskit register named after
    its identifier. Symbolic parameters become Qiskit Parameters. With transpile=True,
    the circuit is transpiled in Qrisp first.
    """
    if transpile:
        from qrisp.circuit.transpiler import transpile

        qc = transpile(qc)

    return _convert_to_qiskit(qc, {})


def _convert_to_qiskit(qc: QuantumCircuit, symbol_param_dic):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qiskit_qc = QuantumCircuit()

    # This dic gives the qiskit qubits/clbits when presented with their identifier
    bit_dic = {}

    for qubit in qc.qubits:
        qiskit_qc.add_register(QuantumRegister(1, name=qubit.identifier))
        bit_dic[qubit.identifier] = qiskit_qc.qubits[-1]

    for clbit in qc.clbits:
        qiskit_qc.add_register(ClassicalRegister(1, name=clbit.identifier))
        bit_dic[clbit.identifier] = qiskit_qc.clbits[-1]

    for instruction in qc.data:
        if (qc_element := _setup_qc_element(instruction, symbol_param_dic, bit_dic)) is not None:
            qiskit_qc.append(*qc_element)

    return qiskit_qc


def _setup_qc_element(instruction: Instruction, symbol_param_dic, bit_dic):
    op = instruction.op
    qiskit_params = [_to_qiskit_param(param, op, symbol_param_dic) for param in op.params]

    qubit_list = [bit_dic[qubit.identifier] for qubit in instruction.qubits]
    clbit_list = [bit_dic[clbit.identifier] for clbit in instruction.clbits]

    if op.name in ["qb_alloc", "qb_dealloc"]:
        return None

    if isinstance(op, ClControlledOperation):
        qiskit_ins = _convert_cl_controlled_operation_to_qiskit(
            op, qiskit_params, qubit_list, clbit_list, symbol_param_dic
        )
        return qiskit_ins, qubit_list, clbit_list

    if issubclass(op.__class__, ControlledOperation):
        return _convert_controlled_operation_to_qiskit(op, qiskit_params, symbol_param_dic), qubit_list, clbit_list
    return _create_qiskit_instruction(op, qiskit_params, symbol_param_dic), qubit_list, clbit_list


def _convert_cl_controlled_operation_to_qiskit(
    op: ClControlledOperation, params, qubit_list, clbit_list, symbol_param_dic
):
    ctrl_clbits = clbit_list[: op.num_control]
    base_clbits = clbit_list[op.num_control :]

    body_qc = QiskitQuantumCircuit(qubit_list + clbit_list)
    if op.base_op.definition:
        body_qc.compose(
            _convert_to_qiskit(op.base_op.definition, symbol_param_dic),
            qubits=qubit_list,
            clbits=base_clbits,
            inplace=True,
        )
    else:
        base_qiskit_ins = _create_qiskit_instruction(op.base_op, params, symbol_param_dic)
        body_qc.append(base_qiskit_ins, qubit_list, base_clbits)

    condition = reduce(
        expr.logic_and,
        [expr.lift(b) if s == "1" else expr.logic_not(b) for b, s in zip(ctrl_clbits, op.ctrl_state)],
    )
    return IfElseOp(condition, true_body=body_qc)


def _convert_controlled_operation_to_qiskit(op: ControlledOperation, params, symbol_param_dic):
    if op.base_operation.definition:
        qiskit_definition = _convert_to_qiskit(op.base_operation.definition, symbol_param_dic)
        base_gate = qiskit_definition.to_gate()
        base_gate.name = op.base_operation.name
        return base_gate.control(len(op.controls), ctrl_state=op.ctrl_state[::-1])

    elif op.base_operation.name == "gphase":
        return _create_qiskit_instruction(op, params, symbol_param_dic)
    elif op.num_qubits == op.base_operation.num_qubits:
        return _create_qiskit_instruction(op.base_operation, params, symbol_param_dic)
    else:
        base_gate = _create_qiskit_instruction(op.base_operation, params, symbol_param_dic)
        return base_gate.control(len(op.controls), ctrl_state=op.ctrl_state[::-1])


def _to_qiskit_param(param, op, symbol_param_dic):
    return _extract_expression(param, op, symbol_param_dic) if isinstance(param, Expr) else param


def _extract_expression(param: Expr, op, symbol_param_dic):
    free_symbols = list(param.free_symbols)
    for symbol in free_symbols:
        if symbol not in symbol_param_dic:
            symbol_param_dic[symbol] = Parameter(str(symbol))

    lambd_expr = lambdify(free_symbols, param)

    try:
        return lambd_expr(*[symbol_param_dic[symbol] for symbol in free_symbols])
    except (TypeError, AttributeError) as e:
        raise NotImplementedError(
            f"Could not convert the parameter {param} of operation {op.name} to a Qiskit parameter expression"
        ) from e


def _create_qiskit_instruction(op, params, symbol_param_dic):  # noqa: PLR0911, PLR0912, PLR0915

    if isinstance(op, PRXGate):
        return qsk_gates.RGate(params[0], params[1] + np.pi / 2)

    match op.name:
        case "cx":
            return qsk_gates.XGate().control(ctrl_state=op.ctrl_state)
        case "cz":
            return qsk_gates.ZGate().control(ctrl_state=op.ctrl_state)
        case "cy":
            return qsk_gates.YGate().control(ctrl_state=op.ctrl_state)
        case "rxx":
            return qsk_gates.RXXGate(*params)
        case "rzz":
            return qsk_gates.RZZGate(*params)
        case "ryy":
            return qsk_gates.RYYGate(*params)
        case "swap":
            return qsk_gates.SwapGate()
        case "measure":
            return Measure()
        case "barrier":
            return Barrier(op.num_qubits)
        case "cp":
            return qsk_gates.CPhaseGate(params[0])
        case "x":
            return qsk_gates.XGate()
        case "y":
            return qsk_gates.YGate()
        case "z":
            return qsk_gates.ZGate()
        case "rx":
            return qsk_gates.RXGate(params[0])
        case "ry":
            return qsk_gates.RYGate(params[0])
        case "rz":
            return qsk_gates.RZGate(params[0])
        case "sx":
            return qsk_gates.SXGate()
        case "h":
            return qsk_gates.HGate()
        case "p":
            return qsk_gates.PhaseGate(params[0])
        case "u1":
            return qsk_gates.RZGate(params[0])
        case "s":
            return qsk_gates.SGate()
        case "s_dg":
            return qsk_gates.SGate().inverse()
        case "sx_dg":
            return qsk_gates.SXdgGate()
        case "xxyy":
            return qsk_gates.XXPlusYYGate(*params, label="(XX+YY)")
        case "t":
            return qsk_gates.TGate()
        case "t_dg":
            return qsk_gates.TGate().inverse()
        case "r":
            return qsk_gates.RGate(*params)
        case "id":
            return qsk_gates.IGate()
        case "reset":
            return Reset()
        case "u3":
            if op.global_phase == 0:
                return qsk_gates.U3Gate(*params)
            # Qiskit's U3Gate has no global phase, so it is wrapped in a gate that
            # carries it, which keeps the phase once the gate is controlled.
            temp_qc = QiskitQuantumCircuit(1, global_phase=_to_qiskit_param(op.global_phase, op, symbol_param_dic))
            temp_qc.append(qsk_gates.U3Gate(*params), [0])
            qiskit_ins = temp_qc.to_gate()
            qiskit_ins.name = "u3"
            return qiskit_ins
        case "gphase":
            temp_qc = QiskitQuantumCircuit(1)
            temp_qc.global_phase += params[0]
            qiskit_ins = temp_qc.to_gate()
            qiskit_ins.name = "gphase"
            return qiskit_ins
        case _:
            if issubclass(op.__class__, ControlledOperation) and "gphase" not in op.name:
                return _convert_controlled_operation_to_qiskit(op, params, symbol_param_dic)

            if not op.definition:
                raise Exception("Could not convert operation " + str(op.name) + " to Qiskit")

            qiskit_definition = _convert_to_qiskit(op.definition, symbol_param_dic)
            try:
                qiskit_ins = qiskit_definition.to_gate()
            except QiskitError:
                qiskit_ins = qiskit_definition.to_instruction()

            qiskit_ins.name = op.name
            return qiskit_ins


op_dict = {op().name: op for op in op_list}
op_dict["u"] = op_dict["u3"]
# Qiskit's u1 is the phase gate, while Qrisp's U1Gate is RZ
op_dict["u1"] = op_dict["p"]


def convert_from_qiskit(qiskit_qc: QiskitQuantumCircuit):
    """Convert a Qiskit QuantumCircuit to a Qrisp QuantumCircuit.

    Controlled gates become ControlledOperations, if/else blocks become classically
    controlled operations and the circuit's global phase is kept. Loops and switches
    raise NotImplementedError.
    """
    qc = QuantumCircuit()

    for qubit in qiskit_qc.qubits:
        _add_qubit_to_qc(qubit, qc)

    for clbit in qiskit_qc.clbits:
        _add_clbit_to_qc(clbit, qc)

    qb_dict = {qiskit_qc.qubits[i]: qc.qubits[i] for i in range(qiskit_qc.num_qubits)}
    cb_dict = {qiskit_qc.clbits[i]: qc.clbits[i] for i in range(qiskit_qc.num_clbits)}

    for instruction in qiskit_qc.data:
        qiskit_op = instruction.operation
        qubits = [qb_dict[qb] for qb in instruction.qubits]
        clbits = [cb_dict[cb] for cb in instruction.clbits]

        condition_bits, condition_value = _extract_condition(qiskit_op, cb_dict)
        if isinstance(qiskit_op, IfElseOp):
            # ``ClControlledOperation`` expects its control clbits before the
            # clbits of the operation it wraps.
            for qrisp_op, body_clbits in _convert_if_else(qiskit_op, clbits, condition_bits, condition_value):
                qc.append(qrisp_op, qubits, condition_bits + body_clbits)
            continue

        # Loops and switches cannot be expressed as a single conditioned
        # operation, and approximating them as one would silently change the
        # circuit's semantics.
        if isinstance(qiskit_op, ControlFlowOp):
            raise NotImplementedError(f"Converting Qiskit {qiskit_op.name} operations to Qrisp is not supported")

        qrisp_op = _convert_operation(qiskit_op)
        if condition_bits:
            clbits += condition_bits
            qrisp_op = qrisp_op.c_if(len(condition_bits), condition_value)

        qc.append(qrisp_op, qubits, clbits)

    # Propagate Qiskit's global phase (stored as a circuit attribute,
    # not an instruction).  Dropping it changes the unitary — critical
    # for controlled operations built from gate definitions with phase.
    if abs(getattr(qiskit_qc, "global_phase", 0)) > GLOBAL_PHASE_TOLERANCE:
        qc.gphase(qiskit_qc.global_phase, qc.qubits[0])

    return qc


def _add_qubit_to_qc(qubit, qc):

    q_reg = qubit._register

    if q_reg is not None:
        qb_list = q_reg._bits if hasattr(q_reg, "_bits") else list(q_reg)
        qubit_name = q_reg.name
        if q_reg.size > 1:
            qubit_name += "." + str(qb_list.index(qubit))

        qc.add_qubit(Qubit(qubit_name))
    else:
        qc.add_qubit()


def _add_clbit_to_qc(clbit, qc):
    cl_reg = clbit._register

    if cl_reg is not None:
        clbit_name = cl_reg.name
        if cl_reg.size > 1:
            # Case distinction for Qiskit version compatibility.
            # Qiskit 2.0 moved registers to Rust and dropped the private
            # `_bits` attribute:
            # https://github.com/Qiskit/qiskit/commit/2fe67f0589d839aaa91633d96343bae5213f21b4
            cb_list = cl_reg._bits if hasattr(cl_reg, "_bits") else list(cl_reg)
            clbit_name += "." + str(cb_list.index(clbit))

        qc.add_clbit(Clbit(clbit_name))
    else:
        qc.add_clbit()


def _qiskit_to_qrisp_param_converter(param):
    match param:
        case np.number():
            return param.item()
        case ParameterExpression():
            return sympify(ParameterExpression.sympify(param))
        case float() | int():
            return float(param)
        case complex():
            return param
        case _:
            print(param)
            raise Exception(f"Could not convert parameter type {type(param)}")


def _convert_params(params):
    def flatten_conditionally(param):
        return list(param.flatten()) if isinstance(param, np.ndarray) else param

    def concat_as_list(result, current):
        return result + current if type(current) is list else result + [current]

    params_flattened = reduce(concat_as_list, map(flatten_conditionally, params), [])
    return [_qiskit_to_qrisp_param_converter(param) for param in params_flattened]


def _extract_condition(qiskit_op, cb_dict):
    # ``condition`` is the common ground across Qiskit versions:
    # gates, and instructions carry it before 2.0, as well as in ``IfElseOp``
    # Qiskit >= 2.0 where ``c_if`` and ``condition_bits`` are removed.
    # For the change release notes, see:
    # https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.0.
    # For the condition prior to 2.0:
    # https://github.com/Qiskit/qiskit/blob/1.4.0/qiskit/circuit/instruction.py#L178-L186,
    # For the condition in ifelseop 2.0 onwards:
    # https://github.com/Qiskit/qiskit/blob/88df8ee7bcae4711580b55a4c1a69edc01d79caf/qiskit/circuit/controlflow/if_else.py#L96-L99
    condition = getattr(qiskit_op, "condition", None)
    if condition is None:
        return [], None

    if not isinstance(condition, tuple):
        raise NotImplementedError("Converting Qiskit conditions given as classical expressions is not supported")

    target, value = condition
    qiskit_bits = [target] if isinstance(target, QiskitClbit) else list(target)

    return [cb_dict[cb] for cb in qiskit_bits], int(value)


def _convert_operation(qiskit_op):
    """Convert a Qiskit operation that is not control flow to the equivalent Qrisp operation."""
    from qrisp import Barrier

    # Qrisp represents controls as a wrapper around the uncontrolled operation,
    # so the base gate is converted on its own and then wrapped.
    if isinstance(qiskit_op, ControlledGate):
        if qiskit_op.base_gate.name == "u":
            # Qiskit's CU gates take a fourth parameter gamma, the global phase of the
            # controlled U.
            # Note: unlike CUGate, the ccu, cccu, ... gates generated by UGate.control with
            # two or more controls have no gamma parameter.
            theta, phi, lam, *gamma = _convert_params(qiskit_op.params)
            base_operation = U3Gate(theta, phi, lam, global_phase=gamma[0] if gamma else 0)
        else:
            base_operation = _convert_operation(qiskit_op.base_gate)
        return ControlledOperation(
            base_operation=base_operation,
            num_ctrl_qubits=qiskit_op.num_ctrl_qubits,
            ctrl_state=bin_rep(qiskit_op.ctrl_state, qiskit_op.num_ctrl_qubits)[::-1],
        )

    # The barrier in ``op_dic`` is constructed with its default span of one
    # qubit, so the qubit count has to be passed explicitly.
    if qiskit_op.name == "barrier":
        return Barrier(qiskit_op.num_qubits)

    params = _convert_params(qiskit_op.params)

    if qiskit_op.name == "r":
        return PRXGate(params[0], params[1])

    for name in (qiskit_op.name, qiskit_op.name.lower()):
        if name in op_dict:
            return op_dict[name](*params)

    if qiskit_op.definition is not None:
        return convert_from_qiskit(qiskit_op.definition).to_gate(name=qiskit_op.name)

    raise Exception("Could not convert Qiskit operation " + str(qiskit_op.name) + " to Qrisp")


def _convert_if_else_body(body, clbits):
    """Convert an ``IfElseOp`` branch body to a Qrisp operation and the outer clbits it acts on."""
    qc = convert_from_qiskit(body)

    # Qiskit's ``if_test`` builder adds the condition clbits to every body as
    # wires, even if the body never touches them. Keeping those would make the
    # branch act on its own condition bits.
    used_clbits = {cb for instr in qc.data for cb in instr.clbits}
    outer_clbits = [clbits[i] for i, cb in enumerate(qc.clbits) if cb in used_clbits]
    qc.clbits = [cb for cb in qc.clbits if cb in used_clbits]
    return qc.to_op(name=body.name), outer_clbits


def _convert_if_else(qiskit_op, clbits, condition_bits, condition_value):
    """Convert a Qiskit ``IfElseOp`` to classically controlled Qrisp operations, one per branch.

    Returns a list of ``(operation, body_clbits)`` pairs. ``body_clbits`` are the outer
    clbits the branch acts on, excluding the condition clbits.
    """
    true_body, false_body = qiskit_op.params

    branches = [(true_body, condition_value)]

    if false_body is not None:
        # Qrisp's ``c_if`` can only express "clbits == value". For a single clbit the
        # else branch is simply "clbit == not value", but for several clbits
        # "!= value" would need one operation per remaining value.
        if len(condition_bits) != 1:
            raise NotImplementedError(
                "Converting Qiskit if/else operations conditioned on multiple clbits is not supported"
            )
        branches.append((false_body, 1 - condition_value))

    ops = []
    for body, value in branches:
        op, body_clbits = _convert_if_else_body(body, clbits)
        # The branch is flattened into individually conditioned instructions,
        # so a branch that writes a condition bit would change whether its
        # remaining instructions (and the else branch) run.
        if set(body_clbits) & set(condition_bits):
            raise NotImplementedError(
                "Converting Qiskit if/else branches that use their own condition clbits is not supported"
            )
        ops.append((op.c_if(len(condition_bits), value), body_clbits))

    return ops
