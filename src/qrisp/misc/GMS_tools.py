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


import numpy as np
from qrisp.circuit import Operation


class GXX_wrapper(Operation):
    def __init__(self, n, chi):
        if n == 1:
            pass
            # raise Exception("Tried to create 1D GXX-gate")
        flattened_chi = sum(chi, [])

        super().__init__(name="GXX", num_qubits=n, params=flattened_chi)

        self.chi = chi
        self.n = n
        from qrisp.circuit import QuantumCircuit

        qc = QuantumCircuit(n)

        for i in range(len(chi)):
            for j in range(len(chi[i])):
                if chi[i][j] == 0 or i == j:
                    continue
                qc.rxx(chi[i][j] / 2, qc.qubits[i], qc.qubits[j])

        self.definition = qc

    def inverse(self):
        chi_inv = [[-x for x in y] for y in self.chi]
        return GXX_wrapper(self.n, chi_inv)


# Mimics a circuit which performs n cphase gates, where each cp interacts with the last
# qubit.
# example
# q2874_0: ─■─────────────────────────────────────────────────────────────
#           │
# q2874_1: ─┼────────■────────────────────────────────────────────────────
#           │        │
# q2874_2: ─┼────────┼────────■───────────────────────────────────────────
#           │        │        │
# q2874_3: ─┼────────┼────────┼────────■──────────────────────────────────
#           │        │        │        │
# q2874_4: ─┼────────┼────────┼────────┼────────■─────────────────────────
#           │        │        │        │        │
# q2874_5: ─┼────────┼────────┼────────┼────────┼────────■────────────────
#           │        │        │        │        │        │
# q2874_6: ─┼────────┼────────┼────────┼────────┼────────┼────────■───────
#           │P(π/2)  │P(π/2)  │P(π/2)  │P(π/2)  │P(π/2)  │P(π/2)  │P(π/2)
# q2874_7: ─■────────■────────■────────■────────■────────■────────■───────
# Using only one GMS gate or two uniform GMS gates
def gms_multi_cp_gate_mono_phase(
    n, theta, use_uniform=True, phase_tolerant=False, basis="GXX"
):
    from qrisp import QuantumSession, QuantumVariable, cp, h, p

    qs = QuantumSession()
    qv = QuantumVariable(n + 1, qs)
    if basis == "GXX":
        if use_uniform:
            gms_1 = GXX_wrapper((n + 1), (n + 1) * [(n + 1) * [-theta / 2]])
            gms_2 = GXX_wrapper(n, n * [n * [theta / 2]])
        else:
            # gms = GXX_wrapper(n, (n+1)*[n*[0] + [-theta/2]])
            for i in range(n):
                cp(theta, qv[i], qv[n])
                # qs.cp(theta, i, n)
            return GXX_converter(qs).to_gate()

        if not phase_tolerant:
            for i in range(n):
                p(theta / 2, qv[i])
                # qs.p(theta/2, qv.reg[i])

        p(theta / 2 * n, qv[-1])
        # qs.p(theta/2*n, qv.reg[-1])

        h(qv)

        qs.append(gms_1, qv.reg)
        if not phase_tolerant:
            qs.append(gms_2, qv.reg[:-1])

        h(qv)

    elif basis == "GZZ":
        if use_uniform:
            gms_1 = GZZ_wrapper((n + 1), (n + 1) * [(n + 1) * [-theta / 2]])
            gms_2 = GZZ_wrapper(n, n * [n * [theta / 2]])
        else:
            # gms = GXX_wrapper(n, (n+1)*[n*[0] + [-theta/2]])
            for i in range(n):
                cp(theta, qv[i], qv[n])
                # qs.cp(theta, i, n)
            return GZZ_converter(qs).to_gate()

        if not phase_tolerant:
            for i in range(n):
                p(theta / 2, qv[i])
                # qs.p(theta/2, qv.reg[i])

        p(theta / 2 * n, qv[-1])
        # qs.p(theta/2*n, qv.reg[-1])

        qs.append(gms_1, qv.reg)
        if not phase_tolerant:
            qs.append(gms_2, qv.reg[:-1])

    else:
        raise "Basis choice must be GXX or GZZ"

    return qs.to_gate(name="gms_mono_phase_cp")


# qb_0: ──■─────────────────
#         │
# qb_1: ──┼────■────────────
#         │    │
# qb_2: ──┼────┼────■───────
#         │    │    │
# qb_3: ──┼────┼────┼────■──
#       ┌─┴─┐┌─┴─┐┌─┴─┐┌─┴─┐
# qb_4: ┤ X ├┤ X ├┤ X ├┤ X ├
#       └───┘└───┘└───┘└───┘
def gms_multi_cx_fan_in(n, use_uniform=True, phase_tolerant=False, basis="GXX"):
    from qrisp import QuantumSession, QuantumVariable, h

    qs = QuantumSession()
    qv = QuantumVariable(n + 1, qs)

    h(qv[-1])

    qs.append(
        gms_multi_cp_gate_mono_phase(
            n,
            np.pi,
            use_uniform=use_uniform,
            phase_tolerant=phase_tolerant,
            basis=basis,
        ),
        qv.reg,
    )

    h(qv[-1])

    # result = qs.to_gate(name = "gms_multi_cx_fan_in")
    # result.name = "gms_multi_cx_fan_in"

    return qs.to_gate(name="gms_multi_cx_fan_in")


# qubit_0: ──■────■────■────■──
#          ┌─┴─┐  │    │    │
# qubit_1: ┤ x ├──┼────┼────┼──
#          └───┘┌─┴─┐  │    │
# qubit_2: ─────┤ x ├──┼────┼──
#               └───┘┌─┴─┐  │
# qubit_3: ──────────┤ x ├──┼──
#                    └───┘┌─┴─┐
# qubit_4: ───────────────┤ x ├
#                         └───┘
def gms_multi_cx_fan_out(n, use_uniform=True, phase_tolerant=False, basis="GXX"):
    from qrisp import QuantumSession, QuantumVariable, h

    qs = QuantumSession()
    qv = QuantumVariable(n + 1, qs)

    for i in range(n):
        h(qv[i])

    qs.append(
        gms_multi_cp_gate_mono_phase(
            n,
            np.pi,
            use_uniform=use_uniform,
            phase_tolerant=phase_tolerant,
            basis=basis,
        ),
        qv.reg,
    )

    for i in range(n):
        h(qv[i])

    result = qs.to_gate()
    result.name = "gms_multi_cx_fan_out"

    return result


# This functions takes a circuit which only consists of phase and cphase gates
# and turns it into a GXX gate + some one-qubit gates
# The basic idea is that a cphase gate acts as
# CP |ab> = exp(i theta/2 (1 - (-1)^(ab))) |ab>
# We rewrite (-1)^(ab) = -1/2*((-1)^(a+b) - ((-1)^a + (-1)^b) - 1)
# The term (-1)^(a+b) can now be executed by a Moelmer-Soerensen:
# exp(i theta/2 Z_a (x) Z_b)
# The terms (-1)^a and (-1)^b are executed by single qubit phase gates
# The remaining -1 is an irrelevant global phase


def GXX_converter(qs):
    from qrisp import QuantumSession, QuantumVariable

    # Check if the given Circuit is valid
    global_phase = np.array([0.0])
    for i in range(len(qs.data)):
        if not qs.data[i].op.name in [
            "p",
            "cp",
            "id",
            "cz",
            "rz",
            "qb_alloc",
            "qb_dealloc",
        ]:
            raise Exception(qs.data[i].op.name + " is neither Phase nor CPhase gate")

        if not qs.data[i].op.name in ["cp", "cz", "qb_alloc", "qb_dealloc"]:
            global_phase += qs.data[i].op.global_phase

        # Subtract a 1/4 of the phase in case we are dealing with a cp gate (for more
        # elaboration, check eq. 99 of https://arxiv.org/abs/2112.10537)
        if qs.data[i].op.name == "cp":
            global_phase -= qs.data[i].op.params[0] / 4

        elif qs.data[i].op.name == "cz":
            global_phase -= np.pi / 4

    # Create qubit index dictionary
    qubit_index_dic = {qs.qubits[i]: i for i in range(len(qs.qubits))}

    # Create phase matrix
    # The (i,j) entry of this matrix contains the phase that is applied onto the
    # qubit pair (i,j), the indices with i == j represent single qubit rz gates
    n = len(qs.qubits)
    phase_matrix = np.zeros((n, n))

    # Go through all data entries of the circuit
    for i in range(len(qs.data)):
        # Set instruction alias
        ins = qs.data[i]

        if ins.op.name in ["id", "qb_alloc", "qb_dealloc"]:
            continue

        # If sinlge qubit gate => Collect phase
        if len(ins.qubits) == 1:
            index = qubit_index_dic[ins.qubits[0]]

            phase_matrix[index, index] += ins.op.params[0]
            continue

        # Get indices
        index_0 = qubit_index_dic[ins.qubits[0]]
        index_1 = qubit_index_dic[ins.qubits[1]]

        # Swap indices if neccessary such that the phase matrix is an upper triangle
        # matrix
        if index_0 >= index_1:
            temp = int(index_0)
            index_0 = index_1
            index_1 = temp

        # Collect phase
        if ins.op.name == "cz":
            phase_matrix[index_0, index_1] += np.pi
        else:
            phase_matrix[index_0, index_1] += ins.op.params[0]

    # Create Quantum Session
    qc_res = qs.clearcopy()

    if global_phase[0] != 0:
        qc_res.gphase(global_phase[0], qc_res.qubits[0])

    # Calculate the row-sum and the column sum of the phase matrix
    # to determine which single qubit phase has to be applied
    # This is because every entry with the same row / column represents a phase gate
    # where the qubit in question participated
    for i in range(n):
        qc_res.p((sum(phase_matrix[i, :]) + sum(phase_matrix[:, i])) / 2, qc_res.qubits[i])
        phase_matrix[i, i] = 0

    # Prepary Chi list for GXX gate
    chi_list = [[-phase_matrix[j, i] for i in range(n)] for j in range(n)]

    # Apply GXX Gate
    for qb in qc_res.qubits:
        qc_res.h(qb)
    qc_res.append(GXX_wrapper(n, chi_list), qc_res.qubits[:n])
    for qb in qc_res.qubits:
        qc_res.h(qb)

    return qc_res


# Similar to gms_multi_cp_gate_mono_phase but also allows more than one phase
# ie. converts circuits of the type
# q54264_0: ─■─────────────────────────────────────────
#            │
# q54264_1: ─┼─────────■───────────────────────────────
#            │         │
# q54264_2: ─┼─────────┼──────────■────────────────────
#            │         │          │
# q54264_3: ─┼─────────┼──────────┼──────────■─────────
#            │P(-π/5)  │P(-2π/5)  │P(-3π/5)  │P(-4π/5)
# q54264_4: ─■─────────■──────────■──────────■─────────
# The advantage over the GXX_converter is the use_uniform mode, which makes
# sure only uniform GMS gates are used
def gms_multi_cp_gate(n, phases, use_uniform=True, basis="GXX"):
    from qrisp import QuantumSession, QuantumVariable, p

    qs = QuantumSession()

    qv = QuantumVariable(n + 1, qs)

    for i in range(n):
        p(phases[i] / 2, qv[i])
        # qs.p(phases[i]/2, i)

    p(sum(phases) / 2, qv[n])
    # qs.p(sum(phases)/2, n)

    qs.append(
        gms_multi_cx_fan_out(
            n, use_uniform=use_uniform, phase_tolerant=False, basis=basis
        ),
        qv.reg,
    )

    for i in range(n):
        qs.p(-phases[i] / 2, i)

    qs.append(
        gms_multi_cx_fan_out(
            n, use_uniform=use_uniform, phase_tolerant=False, basis=basis
        ).inverse(),
        qv.reg,
    )

    result = qs.to_gate()

    result.name = "gms_multi_cp_gate"

    return result


class GZZ_wrapper(Operation):
    def __init__(self, n, chi):
        if n == 1:
            pass
            # raise Exception("Tried to create 1D GZZ-gate")
        flattened_chi = sum(chi, [])

        super().__init__(name="GZZ", num_qubits=n, params=flattened_chi)

        self.chi = chi
        self.n = n
        from qrisp.circuit import QuantumCircuit

        qc = QuantumCircuit(n)

        for i in range(len(chi)):
            for j in range(len(chi[i])):
                if chi[i][j] == 0 or i == j:
                    continue
                qc.rzz(chi[i][j] / 2, i, j)

        self.definition = qc


# This functions takes a circuit which only consists of phase and cphase gates
# and turns it into a GZZ gate + some one-qubit gates
# The basic idea is that a cphase gate acts as
# CP |ab> = exp(i theta/2 (1 - (-1)^(ab))) |ab>
# We rewrite (-1)^(ab) = -1/2*((-1)^(a+b) - ((-1)^a + (-1)^b) - 1)
# The term (-1)^(a+b) can now be executed by a Moelmer-Soerensen:
# exp(i theta/2 Z_a (x) Z_b)
# The terms (-1)^a and (-1)^b are executed by single qubit phase gates
# The remaining -1 is an irrelevant global phase
def GZZ_converter(qs):
    from qrisp import QuantumSession, QuantumVariable

    # Check if the given Circuit is valid
    global_phase = np.array([0.0])
    for i in range(len(qs.data)):
        if not qs.data[i].op.name in [
            "p",
            "cp",
            "id",
            "cz",
            "rz",
            "qb_alloc",
            "qb_dealloc",
        ]:
            raise Exception(qs.data[i].op.name + " is neither Phase nor CPhase gate")

        if not qs.data[i].op.name in ["cp", "cz", "qb_alloc", "qb_dealloc"]:
            global_phase += qs.data[i].op.global_phase

        # Subtract a 1/4 of the phase in case we are dealing with a cp gate (for more
        # elaboration, check eq. 99 of https://arxiv.org/abs/2112.10537)
        if qs.data[i].op.name == "cp":
            global_phase -= qs.data[i].op.params[0] / 4

        elif qs.data[i].op.name == "cz":
            global_phase -= np.pi / 4

    # Create qubit index dictionary
    qubit_index_dic = {qs.qubits[i]: i for i in range(len(qs.qubits))}

    # Create phase matrix
    # The (i,j) entry of this matrix contains the phase that is applied onto the
    # qubit pair (i,j), the indices with i == j represent single qubit rz gates
    n = len(qs.qubits)
    phase_matrix = np.zeros((n, n))

    # Go through all data entries of the circuit
    for i in range(len(qs.data)):
        # Set instruction alias
        ins = qs.data[i]

        if ins.op.name in ["id", "qb_alloc", "qb_dealloc"]:
            continue

        # If sinlge qubit gate => Collect phase
        if len(ins.qubits) == 1:
            index = qubit_index_dic[ins.qubits[0]]

            phase_matrix[index, index] += ins.op.params[0]
            continue

        # Get indices
        index_0 = qubit_index_dic[ins.qubits[0]]
        index_1 = qubit_index_dic[ins.qubits[1]]

        # Swap indices if neccessary such that the phase matrix is an upper triangle
        # matrix
        if index_0 >= index_1:
            temp = int(index_0)
            index_0 = index_1
            index_1 = temp

        # Collect phase
        if ins.op.name == "cz":
            phase_matrix[index_0, index_1] += np.pi
        else:
            phase_matrix[index_0, index_1] += ins.op.params[0]

    # Create Quantum Session
    qc_res = qs.clearcopy()

    if global_phase[0] != 0:
        qc_res.gphase(global_phase[0], 0)

    # Calculate the row-sum and the column sum of the phase matrix
    # to determine which single qubit phase has to be applied
    # This is because every entry with the same row / column represents a phase gate
    # where the qubit in question participated
    for i in range(n):
        qc_res.p((sum(phase_matrix[i, :]) + sum(phase_matrix[:, i])) / 2, i)
        phase_matrix[i, i] = 0

    # Prepary Chi list for GZZ gate
    chi_list = [[-phase_matrix[j, i] for i in range(n)] for j in range(n)]

    # Apply GZZ Gate
    qc_res.append(GZZ_wrapper(n, chi_list), range(n))

    return qc_res
