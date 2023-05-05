"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""

# -*- coding: utf-8 -*-

import numpy as np

from qrisp import (
    PGate,
    QuantumBool,
    QuantumCircuit,
    QuantumSession,
    QuantumVariable,
    convert_to_qb_list,
    RYGate,
    TruthTable,
    XGate,
    bin_rep,
    cx,
    h,
    invert,
    mcx,
    s,
    s_dg,
    x,
)
from qrisp.misc.GMS_tools import GXX_wrapper


# Interface function to quickly change between different implementations of
# multi controlled not gates
def multi_cx(n, method=None):
    # from qrisp.circuit import transpile

    if method == "gms":
        return gms_multi_cx(n)

    elif method == "gray_pt":
        return pt_multi_cx(n)

    elif method == "gray_pt_inv":
        return pt_multi_cx(n).inverse()

    elif method in ["gray", "auto", None]:
        return gray_multi_cx(n)

    else:
        raise Exception('method "' + method + '" not implemented')


# Function to synthesize a multi controlled CX gate from GMS gates
def gms_multi_cx(n):
    qc = QuantumCircuit(n + 1)

    if n == 2:
        for i in range(3):
            qc.ry(np.pi / 2, i)

        qc.rz(np.pi / 4, 2)

        qc.append(GXX_wrapper(3, 3 * [3 * [np.pi / 2]]), qc.qubits)

        for i in range(3):
            qc.rx(-np.pi / 2, i)

        qc.rz(-np.pi / 2, 2)

        for i in range(3):
            qc.rx(-np.pi / 4, i)

        qc.append(GXX_wrapper(3, 3 * [3 * [np.pi / 4]]), qc.qubits)

        qc.rz(np.pi / 2, 2)

        qc.append(GXX_wrapper(3, 3 * [3 * [np.pi / 2]]), qc.qubits)

        for i in range(3):
            qc.rx(np.pi / 2, i)
            qc.ry(-np.pi / 2, i)

        result = qc.to_gate()

        result.name = "GMS toffoli"

        return result
    else:
        raise Exception(str(n) + "-controlled x gate not implemented for gms method")


def gray_pt_mcx(n, ctrl_state):
    input_qv = QuantumVariable(n)
    output_qv = QuantumVariable(1, qs=input_qv.qs)

    tt_str = 2 ** n * ["0"]
    tt_str[int(ctrl_state[::-1], 2)] = "1"

    tt = TruthTable([tt_str])

    tt.q_synth(input_qv, output_qv, method="gray_pt")

    res = input_qv.qs.copy()
    return res.to_gate(f"pt{n}cx")


# Function to synthesize a phase tolerant multi controlled X gate
def pt_multi_cx(n, reduced=False):
    res = QuantumCircuit(n + 1)

    if n == 1:
        res.cx(0, 1)

    # The special cases n = 2 and n = 3 are handled as described in
    # https://arxiv.org/pdf/1508.03273.pdf
    elif n == 2:
        if reduced:
            res.append(reduced_margolus_qc.to_gate("reduced margolus"), res.qubits)
        else:
            res.append(margolus_qc.to_gate("margolus"), res.qubits)

    elif n == 3:
        if reduced:
            res.append(reduced_maslov_qc.to_gate("reduced maslov"), res.qubits)
        else:
            res.append(maslov_qc.to_gate("maslov"), res.qubits)
    else:
        input_qv = QuantumVariable(n)
        output_qv = QuantumVariable(1, qs=input_qv.qs)

        tt = TruthTable([(2 ** (n) - 1) * "0" + "1"])

        tt.q_synth(input_qv, output_qv, method="gray_pt")

        res = input_qv.qs.copy()

    return res.to_gate(f"pt{n}cx")


# Toffoli Implementation according to https://arxiv.org/pdf/1206.0758.pdf
toffoli_qc = QuantumCircuit(3)

toffoli_qc.h(2)
toffoli_qc.p(-np.pi / 4, [0, 1])
toffoli_qc.cx(2, 0)
toffoli_qc.cx(1, 2)
toffoli_qc.p(np.pi / 4, 0)
toffoli_qc.cx(1, 0)
toffoli_qc.p(np.pi / 4, 2)
toffoli_qc.cx(1, 2)
toffoli_qc.p(-np.pi / 4, 0)
toffoli_qc.cx(2, 0)
toffoli_qc.p(np.pi / 4, 0)
toffoli_qc.p(-np.pi / 4, 2)
toffoli_qc.cx(1, 0)
toffoli_qc.h(2)

margolus_qc = QuantumCircuit(3)
G = RYGate(np.pi / 4)

margolus_qc.append(G, 2)
margolus_qc.cx(1, 2)
margolus_qc.append(G, 2)
margolus_qc.cx(0, 2)

reduced_margolus_qc = margolus_qc.copy()

margolus_qc.append(G.inverse(), 2)
margolus_qc.cx(1, 2)
margolus_qc.append(G.inverse(), 2)

maslov_qc = QuantumCircuit(4)

maslov_qc.h(3)
maslov_qc.t(3)
maslov_qc.cx(2, 3)
maslov_qc.t_dg(3)
maslov_qc.h(3)
maslov_qc.cx(0, 3)
maslov_qc.t(3)
maslov_qc.cx(1, 3)
maslov_qc.t_dg(3)
maslov_qc.cx(0, 3)

reduced_maslov_qc = maslov_qc.copy()

maslov_qc.t(3)
maslov_qc.cx(1, 3)
maslov_qc.t_dg(3)
maslov_qc.h(3)
maslov_qc.t(3)
maslov_qc.cx(2, 3)
maslov_qc.t_dg(3)
maslov_qc.h(3)


# Ancilla supported multi controlled X gates from https://arxiv.org/pdf/1508.03273.pdf
def maslov_mcx(n, ctrl_state=-1):
    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2 ** n
        ctrl_state = bin_rep(ctrl_state, n)

    res = QuantumCircuit(n + 1)
    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            res.x(i)

    if n == 1:
        res.cx(0, 1)
    elif n == 2:
        res.append(toffoli_qc.to_gate("toffoli"), res.qubits)
    elif n == 3:
        res.add_qubit()
        res.append(margolus_qc.to_gate(), [0, 1, 3])
        res.append(toffoli_qc.to_gate(), [2, 3, 4])
        res.append(margolus_qc.inverse().to_gate(), [0, 1, 3])
    elif n == 4:
        res.add_qubit()
        res.append(maslov_qc.to_gate(), [0, 1, 2, 4])
        res.append(toffoli_qc.to_gate(), [3, 4, 5])
        res.append(maslov_qc.inverse().to_gate(), [0, 1, 2, 4])
    else:
        raise Exception('Multi CX for method "Maslov" only defined for n <= 4')

    for i in range(len(ctrl_state)):
        if ctrl_state[i] == "0":
            res.x(i)

    return res.to_gate(f"maslov {n}cx")


# Function to synthesize a multi controlled X gate using gray synthesis
def gray_multi_cx(n):
    qs = QuantumSession()
    input_qv = QuantumVariable(n, qs)
    output_qv = QuantumVariable(1, qs)

    if n == 1:
        qs.cx(0, 1)
    elif n == 2:
        qs = toffoli_qc.copy()

    else:
        tt_array = np.zeros((2 ** n, 1))
        tt_array[-1, 0] = 1

        tt = TruthTable(tt_array)

        tt.q_synth(input_qv, output_qv, method="gray")

    result = qs.to_gate()
    result.name = "gray multi cx"

    return result


# Ancilla supported multi controlled X with logarithmic depth based on
# https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf
def balauca_mcx(input_qubits, target, ctrl_state=None, phase=None):
    hybrid_mcx(
        input_qubits, target, ctrl_state=ctrl_state, phase=phase, num_ancilla=np.inf
    )


# Algorithm based on https://link.springer.com/article/10.1007/s10773-017-3389-4
def yong_mcx(input_qubits, target, ancilla=None, ctrl_state=None):
    if isinstance(target, list):
        if len(target) != 1:
            raise Exception("Tried to execute yong algorithm with multiple targets")
        target = target[0]

    if ctrl_state is not None:
        for i in range(len(input_qubits)):
            input_qubits[i].ctrl_state = ctrl_state[i]

    if len(input_qubits) < 3:
        ctrl_state = ""
        for i in range(len(input_qubits)):
            if hasattr(input_qubits[i], "ctrl_state"):
                ctrl_state += input_qubits[i].ctrl_state
            else:
                ctrl_state += "1"

    if len(input_qubits) == 2:
        if ancilla is None:
            mcx(input_qubits, target, method="gray", ctrl_state=ctrl_state)
        else:
            target.qs().append(
                gray_pt_mcx(2, ctrl_state=ctrl_state), input_qubits + [target]
            )
            # mcx(input_qubits, target, method = "gray_pt", ctrl_state = ctrl_state)
        return

    ancilla_allocated = False

    if ancilla is None:
        ancilla_allocated = True
        ancilla_bl = QuantumBool(name="yong_anc*")
        ancilla = ancilla_bl[0]

    n = len(input_qubits)

    partition_k_1 = input_qubits[n // 2:]
    partition_k_2 = input_qubits[: n // 2]

    h(target)

    yong_mcx(input_qubits=partition_k_1, target=ancilla, ancilla=partition_k_2[-1])

    s(ancilla)

    yong_mcx(
        input_qubits=partition_k_2 + [target], target=ancilla, ancilla=partition_k_1[-1]
    )

    s_dg(ancilla)

    with invert():
        s(ancilla)

        yong_mcx(
            input_qubits=partition_k_2 + [target],
            target=ancilla,
            ancilla=partition_k_1[0],
        )

        s_dg(ancilla)

        yong_mcx(input_qubits=partition_k_1, target=ancilla, ancilla=partition_k_2[0])

    h(target)

    if ancilla_allocated:
        ancilla_bl.delete()

    if ctrl_state is not None:
        for i in range(len(input_qubits)):
            del input_qubits[i].ctrl_state


# Hybrid algorithm of yong and balauca with customizable ancilla qubit count.
# Performs several balauca layers and cancels the recursion with yong.
def hybrid_mcx(
        input_qubits,
        target,
        ctrl_state=None,
        phase=None,
        num_ancilla=np.inf,
        num_dirty_ancilla=0,
):
    """
    Function to dynamically generate mcx gates for a given amount of ancilla qubits.

    Parameters
    ----------
    input_qubits : list[Qubit]
        The Qubits to control on.
    target : Qubit
        The Qubit to target.
    ctrl_state : str, optional
        The control state to activate the X Gate on. The default is "11...".
    phase : float or sympy.Symbol, optional
        If given, this function performs a mcp instead of a mcx. The default is None.
    num_ancilla : int, optional
        The amount of ancillae this function is allowed to use. The default is np.inf.
    count_yong : bool, optional
        If set to False, the ancilla that is used by the yong recursion termination is
        not counted (because it can be dirty). The default is True.

    Returns
    -------
    None.

    """

    input_qubits = list(input_qubits)
    for i in range(len(input_qubits)):
        if isinstance(input_qubits[i], QuantumBool):
            input_qubits[i] = input_qubits[i][0]

    if ctrl_state is not None:
        for i in range(len(input_qubits)):
            input_qubits[i].ctrl_state = ctrl_state[i]

    target = list(target)

    qs = target[0].qs()

    if len(input_qubits) <= 3 or num_ancilla == 0:
        ctrl_state = ""
        for qb in input_qubits:
            if hasattr(qb, "ctrl_state"):
                ctrl_state += qb.ctrl_state
            else:
                ctrl_state += "1"

        if len(input_qubits) == 2:
            if phase is None:
                qs.append(
                    XGate().control(2, ctrl_state=ctrl_state, method="gray"),
                    input_qubits + [target],
                )
            else:
                qs.append(
                    PGate(phase).control(2, ctrl_state=ctrl_state),
                    input_qubits + [target],
                )

        elif num_dirty_ancilla and phase is None:
            balauca_dirty(
                input_qubits, target, k=num_dirty_ancilla, ctrl_state=ctrl_state
            )

        else:
            if phase is None:
                qs.append(
                    XGate().control(
                        len(input_qubits), ctrl_state=ctrl_state, method="gray"
                    ),
                    input_qubits + [target],
                )
            else:
                qs.append(
                    PGate(phase).control(len(input_qubits), ctrl_state=ctrl_state),
                    input_qubits + [target],
                )
        return

    structure = structure_decider(len(input_qubits), num_ancilla)  # [::-1]
    layer_input = []
    layer_structure = []
    layer_output = []
    remainder = list(input_qubits)

    for i in range(len(structure)):
        if not structure[0] <= len(remainder):
            break

        layer_output.append(QuantumBool(name="balauca_anc*"))

        for j in range(structure[0]):
            layer_input.append(remainder.pop(0))

        layer_structure.append(structure.pop(0))

    balauca_layer(layer_input, layer_output, structure=layer_structure, invert=False)

    hybrid_mcx(
        layer_output + remainder,
        target,
        num_ancilla=num_ancilla - len(layer_output),
        num_dirty_ancilla=num_dirty_ancilla,
        phase=phase,
    )

    balauca_layer(layer_input, layer_output, structure=layer_structure, invert=True)

    [qbl.delete() for qbl in layer_output]

    if ctrl_state is not None:
        for i in range(len(input_qubits)):
            del input_qubits[i].ctrl_state

    return


def balauca_layer(input_qubits, output_qubits, structure, invert=False):
    if not output_qubits:
        return

    qs = output_qubits[0].qs()
    input_qubits = list(input_qubits)

    counter = 0
    for i in range(len(output_qubits)):
        if structure[i] == 3:
            ctrl_qubits = [
                input_qubits[counter],
                input_qubits[counter + 1],
                input_qubits[counter + 2],
            ]
            counter += 3

            ctrl_state = ""
            for qb in ctrl_qubits:
                if hasattr(qb, "ctrl_state"):
                    ctrl_state += qb.ctrl_state
                else:
                    ctrl_state += "1"

            gate = XGate().control(3, method="gray_pt", ctrl_state=ctrl_state)

            if invert:
                gate = gate.inverse()

            qs.append(gate, ctrl_qubits + [output_qubits[i]])
        else:
            ctrl_qubits = [input_qubits[counter], input_qubits[counter + 1]]
            counter += 2

            ctrl_state = ""
            for qb in ctrl_qubits:
                if hasattr(qb, "ctrl_state"):
                    ctrl_state += qb.ctrl_state
                else:
                    ctrl_state += "1"

            gate = XGate().control(2, method="gray_pt", ctrl_state=ctrl_state)

            if invert:
                gate = gate.inverse()

            qs.append(gate, ctrl_qubits + [output_qubits[i]])

    return


def structure_decider(n, k):
    # Each element of a Balauca layer reduces the amount of
    # Control qubits for the next layer either by 1 (margolous gate)
    # or by 2 (phase toleratn maslov gate). Ideally, we reduce the amount
    # of control qubits to 3, so we can cancel the recursion
    # and deploy a Toffoli.

    # If we have n controls, p = n - 2 is the amount of reductions,
    # that need to be performed.

    # If we have k ancillae at our disposal, we need to satisfy:

    # p = 2*triple_mcx + (k-triple_mcx_count) = k + triple_mcx_count

    # <=> triple_mcx_count = n - 2 - k

    triple_mcx_count = n - 2 - k

    if triple_mcx_count <= 0:
        if n % 2:
            return (n - 3) // 2 * [2] + [3]
        else:
            return n // 2 * [2]
    elif triple_mcx_count > k:
        return k * [3]
    else:
        return triple_mcx_count * [3] + (k - triple_mcx_count) * [2]


def vchain_2_dirty(control, target, dirty_ancillae=None):
    control = convert_to_qb_list(control)
    target = convert_to_qb_list(target)

    if len(control) == 1:
        cx(control, target)
        return
    elif len(control) == 2:
        mcx(control, target, method="gray")

    n = len(control)
    k = n - 2

    dirty_ancilla_qbls = []
    if dirty_ancillae is None:
        dirty_ancilla_qbls = [QuantumBool(name="vchain_2_dirty*") for i in range(k)]
        dirty_ancillae = [qbl[0] for qbl in dirty_ancilla_qbls]

    def reduced_margolus(control, target):
        qs = target.qs()

        control = list(control)
        qs.append(reduced_margolus_qc.to_gate("reduced_margolus"), control + [target])
        # qs.append(XGate().control(2), control + [target])

    def margolus(control, target):
        qs = target.qs()

        control = list(control)
        qs.append(margolus_qc.to_gate("margolus"), control + [target])
        # qs.append(XGate().control(2), control + [target])

    control_temp_list = list(control)
    dirty_ancillae_temp_list = list(dirty_ancillae)
    qubit_list = []

    qubit_list.append(control_temp_list.pop(0))
    qubit_list.append(control_temp_list.pop(0))

    while control_temp_list:
        qubit_list.append(dirty_ancillae_temp_list.pop(0))
        qubit_list.append(control_temp_list.pop(0))

    qubit_list.append(target[0])

    m = len(qubit_list)

    for i in range(k):
        if i == k - 1:
            margolus(
                [qubit_list[m - 5 - 2 * i], qubit_list[m - 4 - 2 * i]],
                qubit_list[m - 3 - 2 * i],
            )
        else:
            reduced_margolus(
                [qubit_list[m - 5 - 2 * i], qubit_list[m - 4 - 2 * i]],
                qubit_list[m - 3 - 2 * i],
            )

    for i in range(k):
        if i == k - 1:
            reduced_margolus(
                [qubit_list[2 * i + 2], qubit_list[2 * i + 3]], qubit_list[2 * i + 4]
            )
        else:
            with invert():
                reduced_margolus(
                    [qubit_list[2 * i + 2], qubit_list[2 * i + 3]],
                    qubit_list[2 * i + 4],
                )

    for i in range(k):
        if i == k - 1:
            with invert():
                margolus(
                    [qubit_list[m - 5 - 2 * i], qubit_list[m - 4 - 2 * i]],
                    qubit_list[m - 3 - 2 * i],
                )
        else:
            reduced_margolus(
                [qubit_list[m - 5 - 2 * i], qubit_list[m - 4 - 2 * i]],
                qubit_list[m - 3 - 2 * i],
            )

    for i in range(k):
        if i == k - 1:
            with invert():
                reduced_margolus(
                    [qubit_list[2 * i + 2], qubit_list[2 * i + 3]],
                    qubit_list[2 * i + 4],
                )
        else:
            with invert():
                reduced_margolus(
                    [qubit_list[2 * i + 2], qubit_list[2 * i + 3]],
                    qubit_list[2 * i + 4],
                )

    [qbl.delete() for qbl in dirty_ancilla_qbls]


def balauca_dirty(control, target, k, dirty_ancillae=None, ctrl_state=None):
    control = convert_to_qb_list(control)
    target = convert_to_qb_list(target)
    qs = target[0].qs()

    n = len(control)

    k = min((n - 2) // 2 + 1, k)

    if k == 0:
        qs.append(
            XGate().control(n, ctrl_state=ctrl_state, method="gray"),
            list(control) + [target],
        )
        return

    if ctrl_state is not None:
        for i in range(len(ctrl_state)):
            if ctrl_state[i] == "0":
                x(control[i])

    def reduced_maslov(control, target):
        qs = target.qs()

        qs.append(reduced_maslov_qc.to_gate("reduced_maslov"), control + [target])
        # qs.append(XGate().control(2), control + [target])

    m_1 = int(np.ceil((n - 2 * (k - 1)) / 2))
    m_2 = int(np.floor((n - 2 * (k - 1)) / 2))

    dirty_ancilla_qbls = []
    if dirty_ancillae is None:
        dirty_ancilla_qbls = [
            QuantumBool(name="balauca_dirty*", qs=qs) for i in range(k)
        ]
        dirty_ancillae = [qbl[0] for qbl in dirty_ancilla_qbls]

    upper_block_qubits = control[:m_1] + [dirty_ancillae[0]]
    lower_block_qubits = control[-m_2:] + [dirty_ancillae[-1]]

    def balauca_dirty_helper(control, target, ancillae):
        if len(ancillae) == 0:
            vchain_2_dirty(control, target, dirty_ancillae=lower_block_qubits)
            # mcx(control, target)
            return

        reduced_maslov([ancillae[-1]] + control[-2:], target)

        balauca_dirty_helper(control[:-2], ancillae[-1], ancillae[:-1])

        with invert():
            reduced_maslov([ancillae[-1]] + control[-2:], target)

    vchain_2_dirty(lower_block_qubits, target, dirty_ancillae=upper_block_qubits)

    balauca_dirty_helper(control[:-m_2], dirty_ancillae[-1], dirty_ancillae[:-1])

    vchain_2_dirty(lower_block_qubits, target, dirty_ancillae=upper_block_qubits)

    balauca_dirty_helper(control[:-m_2], dirty_ancillae[-1], dirty_ancillae[:-1])

    [qbl.delete() for qbl in dirty_ancilla_qbls]

    if ctrl_state is not None:
        for i in range(len(ctrl_state)):
            if ctrl_state[i] == "0":
                x(control[i])
