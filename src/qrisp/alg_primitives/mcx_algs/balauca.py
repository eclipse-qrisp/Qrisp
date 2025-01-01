"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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
********************************************************************************/
"""

from jax.core import Tracer
import numpy as np
from qrisp.circuit import XGate, PGate, convert_to_qb_list, Qubit
from qrisp.qtypes import QuantumBool, QuantumVariable
from qrisp.core.gate_application_functions import x, cx, mcx
from qrisp.alg_primitives.mcx_algs.circuit_library import reduced_maslov_qc, margolus_qc, reduced_margolus_qc
from qrisp.alg_primitives.mcx_algs.gidney import GidneyLogicalAND
from qrisp.alg_primitives.mcx_algs.jones import jones_toffoli
from qrisp.environments.quantum_inversion import invert
from qrisp.jasp import check_for_tracing_mode, AbstractQubit

# Ancilla supported multi controlled X with logarithmic depth based on
# https://www.iccs-meeting.org/archive/iccs2022/papers/133530169.pdf
def balauca_mcx(input_qubits, target, ctrl_state=None, phase=None):
    hybrid_mcx(
        input_qubits, target, ctrl_state=ctrl_state, phase=phase, num_ancilla=np.inf
    )


# Hybrid algorithm of yong and balauca with customizable ancilla qubit count.
# Performs several balauca layers and cancels the recursion with yong.
def hybrid_mcx(
        input_qubits,
        target,
        ctrl_state=None,
        phase=None,
        num_ancilla=np.inf,
        num_dirty_ancilla=0,
        use_mcm=False,
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

    if isinstance(target, Qubit):
        target = [target]
    elif isinstance(target, QuantumVariable):
        target = list(target)

    if ctrl_state is None:
        ctrl_state = len(input_qubits)*"1"
    qs = target[0].qs()
    
    if len(input_qubits) <= 2 + int(not use_mcm) or num_ancilla == 0:
        if len(input_qubits) == 2 + int(not use_mcm):
            
            if phase is None:
                qs.append(
                    XGate().control(len(input_qubits), ctrl_state=ctrl_state, method="gray"),
                    input_qubits + target,
                )
            else:
                
                if use_mcm:
                    gate = GidneyLogicalAND(ctrl_state=ctrl_state)
                else:
                    gate = XGate().control(len(input_qubits), method="gray_pt", ctrl_state=ctrl_state)
                
                qs.append(gate,
                          input_qubits + target,
                          )
                
                qs.append(
                    PGate(phase),
                    target,
                )
                
                qs.append(gate.inverse(),
                          input_qubits + target,
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
                    input_qubits + target,
                )
            else:
                qs.append(
                    PGate(phase).control(len(input_qubits), ctrl_state=ctrl_state),
                    input_qubits + target,
                )
        return

    structure = structure_decider(len(input_qubits), num_ancilla)  # [::-1]
    layer_input = []
    layer_structure = []
    layer_output = []
    remainder = list(input_qubits)
    ctrl_state = list(ctrl_state)
    sub_ctrl_list = []

    for i in range(len(structure)):
        if not structure[0] <= len(remainder):
            break

        layer_output.append(QuantumBool(name="balauca_anc*", qs = input_qubits[0].qs()))

        for j in range(structure[0]):
            layer_input.append(remainder.pop(0))
            sub_ctrl_list.append(ctrl_state.pop(0))

        layer_structure.append(structure.pop(0))

    balauca_layer(layer_input, layer_output, structure=layer_structure, invert=False, use_mcm=use_mcm, ctrl_list = sub_ctrl_list)

    hybrid_mcx(
        layer_output + remainder,
        target,
        num_ancilla=num_ancilla - len(layer_output),
        num_dirty_ancilla=num_dirty_ancilla,
        phase=phase,
        use_mcm = use_mcm,
        ctrl_state = "".join(["1"*len(layer_output)] + ctrl_state)
    )

    balauca_layer(layer_input, layer_output, structure=layer_structure, invert=True, use_mcm=use_mcm, ctrl_list = sub_ctrl_list)

    [qbl.delete() for qbl in layer_output]

    return


def balauca_layer(input_qubits, output_qubits, structure, invert=False, use_mcm = False, ctrl_list = None):
    if not output_qubits:
        return
    
    if check_for_tracing_mode():
        from qrisp.jasp import TracingQuantumSession
        qs = TracingQuantumSession.get_instance()
    else:
        qs = input_qubits[0].qs()
        
    input_qubits = list(input_qubits)

    counter = 0
    for i in range(len(output_qubits)):
        if structure[i] == 3:
            ctrl_qubits = [
                input_qubits[counter],
                input_qubits[counter + 1],
                input_qubits[counter + 2],
            ]

            ctrl_state = "".join([ctrl_list[counter],
                                  ctrl_list[counter+1],
                                  ctrl_list[counter+2]])
            
            counter += 3

            gate = XGate().control(3, method="gray_pt", ctrl_state=ctrl_state)

            if invert:
                gate = gate.inverse()

            if isinstance(output_qubits[i], Qubit):
                target = output_qubits[i]
            else:
                target = output_qubits[i][0]
                
            qs.append(gate, ctrl_qubits + [target])
            
        else:
            ctrl_qubits = [input_qubits[counter], input_qubits[counter + 1]]

            ctrl_state = "".join([ctrl_list[counter],
                                  ctrl_list[counter+1]])
            
            counter += 2

            if use_mcm:
                gate = GidneyLogicalAND(ctrl_state=ctrl_state)
            else:
                gate = XGate().control(2, method="gray_pt", ctrl_state=ctrl_state)

            if invert:
                gate = gate.inverse()
            
            if isinstance(output_qubits[i], Qubit):
                target = output_qubits[i]
            elif isinstance(output_qubits[i], Tracer) and isinstance(output_qubits[i].aval, AbstractQubit):
                target = output_qubits[i]
            else:
                target = output_qubits[i][0]
                
            qs.append(gate, ctrl_qubits + [target])

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
        return n // 2 * [2]
    elif triple_mcx_count > k:
        return k * [3]
    else:
        return triple_mcx_count * [3] + (k - triple_mcx_count) * [2]


def balauca_dirty(control, target, k, dirty_ancillae=None, ctrl_state=None):
    control = convert_to_qb_list(control)
    target = convert_to_qb_list(target)
    qs = target[0].qs()

    n = len(control)

    k = min((n - 2) // 2 + 1, k)

    if k <= 1:
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
            mcx(
                [qubit_list[2 * i + 2], qubit_list[2 * i + 3]], qubit_list[2 * i + 4], method = "gray"
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
                mcx(
                    [qubit_list[2 * i + 2], qubit_list[2 * i + 3]],
                    qubit_list[2 * i + 4], method = "gray"
                )
        else:
            with invert():
                reduced_margolus(
                    [qubit_list[2 * i + 2], qubit_list[2 * i + 3]],
                    qubit_list[2 * i + 4],
                )

    [qbl.delete() for qbl in dirty_ancilla_qbls]