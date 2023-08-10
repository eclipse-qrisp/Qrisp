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


from qrisp import Clbit, QuantumCircuit, Qubit
from qrisp.simulator import ImpureQuantumState, QuantumState, run, single_shot_sim


class QuantumNetwork:
    def __init__(self):
        self.qs = QuantumState(0)
        self.qubit_map = {}
        self.overall_circuit = QuantumCircuit()
        self.client_measurement_counter = {}
        self.inbox_dict = {}
        self.qubit_creation_counter = {}

    def register_client(self, name):
        if name in list(self.client_measurement_counter.keys()):
            raise Exception("Tried to register an existing client.")

        self.client_measurement_counter[name] = 0
        self.inbox_dict[name] = []
        self.qubit_creation_counter[name] = 0

    def send_qubits(self, sender, recipient, qubits, annotation):
        keys = list(self.qubit_map.keys())
        ids = keys

        qubits = convert_qubit_names(qubits, sender)

        for qb in qubits:
            if qb.identifier in ids:
                index = ids.index(qb.identifier)

                if sender != self.qubit_map[keys[index]]:
                    raise Exception("Tried to send qubit not belonging to sender.")
            else:
                self.add_qubit(qb, sender)

            self.qubit_map[qb.identifier] = recipient

        self.inbox_dict[recipient].append((qubits, annotation))

    def get_clear_qc(self, name):
        qc = QuantumCircuit()

        for key in self.qubit_map.keys():
            if self.qubit_map[key] == name:
                qc.add_qubit(Qubit(key))

        return qc

    def request_qubits(self, amount, owner):
        res_list = []
        for i in range(amount):
            new_qb = Qubit(
                "qb_" + str(self.qubit_creation_counter[owner]) + "@" + owner
            )
            self.add_qubit(new_qb, owner)
            res_list.append(new_qb)

        return res_list

    def add_qubit(self, qubit, owner):
        if owner not in self.client_measurement_counter.keys():
            raise Exception("Tried to request qubits for non registered client")

        qubit = convert_qubit_names([qubit], owner)[0]

        if qubit in self.qubit_map.keys():
            raise Exception("Tried to create already existing qubit")

        self.qubit_map[qubit.identifier] = owner
        self.qubit_creation_counter[owner] += 1
        self.overall_circuit.add_qubit(qubit)
        self.qs.add_qubit()

        return [qubit]

    def run(self, qc, name):
        keys = list(self.qubit_map.keys())
        ids = keys

        qc = convert_qc_qubit_names(qc, name)

        for qb in qc.qubits:
            if qb.identifier in ids:
                index = ids.index(qb.identifier)

                if self.qubit_map[keys[index]] != name:
                    raise Exception("Tried to operate one foreign qubits")

            else:
                self.add_qubit(qb, name)

        temp_qc = self.overall_circuit.clearcopy()
        temp_qc.clbits = []

        translation_dic = {}

        for clbit in qc.clbits:
            overall_qc_clbit = Clbit(
                "cb" + "_" + name + "_" + str(self.client_measurement_counter[name])
            )
            self.client_measurement_counter[name] += 1

            self.overall_circuit.add_clbit(overall_qc_clbit)
            temp_qc.add_clbit(clbit)

            translation_dic[clbit] = overall_qc_clbit

        translation_dic.update({qb: qb for qb in qc.qubits})

        temp_qc.extend(qc)

        res, self.qs = single_shot_sim(temp_qc, self.qs)

        self.overall_circuit.barrier()
        self.overall_circuit.extend(qc, translation_dic)

        return {res: 1}

    def get_overall_qc(self):
        return self.overall_circuit.copy()

    def inbox(self, name):
        return self.inbox_dict[name]

    def reset_network_state(self):
        self.qs = QuantumState(0)
        self.qubit_map = {}
        self.overall_circuit = QuantumCircuit()
        self.client_measurement_counter = {}
        self.inbox = {}


def convert_qc_qubit_names(qc, name):
    res_qc = qc.clearcopy()

    res_qc.qubits = convert_qubit_names(qc.qubits, name)

    transl_dic = {
        qc.qubits[i].identifier: res_qc.qubits[i] for i in range(len(qc.qubits))
    }

    for instr in qc.data:
        res_qc.append(
            instr.op, [transl_dic[qb.identifier] for qb in instr.qubits], instr.clbits
        )

    return res_qc


def convert_qubit_names(qubits, name):
    res = []
    for i in range(len(qubits)):
        qb = qubits[i]
        if len(qb.identifier.split("@")) == 1:
            res.append(Qubit(qb.identifier + "@" + name))
        else:
            res.append(Qubit(qb.identifier))

    return res
