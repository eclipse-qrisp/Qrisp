"""
********************************************************************************
* Copyright (c) 2025 the Qrisp authors
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

import pennylane as qml 
from qrisp.interface import qml_converter

from qrisp.circuit.standard_operations import     XGate,YGate, ZGate,    CXGate,  CYGate,    CZGate,MCXGate,PGate,  CPGate,u3Gate,  HGate,RXGate,   RYGate,   RZGate,   MCRXGate,SGate , TGate, RXXGate,RZZGate,  SXGate,   SXDGGate,  Barrier, Measurement,  Reset,  QubitAlloc, QubitDealloc,GPhaseGate,  SwapGate,U1Gate,  IDGate
import numpy as np
from qrisp import QuantumVariable



# create randomized qrisp circuit, convert to pennylane, measure outcomes and compare if they are equivalent.

def randomized_ciruit_testing():

    ############ create the randomized circuit
    qvRand = QuantumVariable(10)
    qcRand = qvRand.qs

    single_gates = [    
        XGate(),
        YGate(),
        ZGate(),
        HGate(),SXGate(), SGate()]

    rot_gates = [
        RXGate(0.5),
        RYGate(0.5),
        RZGate(0.5),
        PGate(0.5)]

    c_gates = [
        CXGate(),
        CYGate(),
        CZGate()]

    mc_gates = [
            SwapGate(),     
            RXXGate(0.5),
            RZZGate(0.5)]
    
    special_gates = [
                    MCXGate(control_amount=3),
                    MCRXGate(0.5, control_amount=3)
                    ]

    op_list = [*mc_gates, *c_gates, *single_gates, *rot_gates, *special_gates]

    used_ops = []
    for index in range(30):
        used_ops.append(op_list[np.random.randint(0,len(op_list))])
    for op in used_ops:
        qubit_1 = qvRand[np.random.randint(7,9)]
        qubit_2 = qvRand[np.random.randint(3,6)]
        if op in single_gates or op in rot_gates:
            qcRand.append(op, qubit_1)
        # this is being called first due to mcx is subclass of cx reasons
        elif op in special_gates:
            qcRand.append(op, [qvRand[0],qvRand[2],qubit_2,qubit_1])
        elif op in c_gates or mc_gates:
            qcRand.append(op, [qubit_1,qubit_2])
    ######## done with the circuit


    # instanciate pennylane circuit
    qcRand_qubits = [qubit.identifier for qubit in qcRand.qubits]
    dev = qml.device('default.qubit', wires = qcRand_qubits, shots = 1000)
    pl_qcRand = qml_converter(qc=qcRand)
    circ = pl_qcRand
    @qml.qnode(dev)
    def circuit():
        circ()
        return qml.probs(wires=qcRand_qubits)
    
    #save result from probability measurement
    # --> look up   qml.probs()   documentation for further insights
    # --> the result is an ordered array containing measurement probabilities (ordered in terms of binary strings representing the measurement results)
    qml_res_w_zeros = circuit().tolist()
    qml_res = qml_res_w_zeros

    # get qrisp-qv measurement
    qrisp_res = qvRand.get_measurement()
    # sort the keys in binary order
    qrisp_keySort = sorted(qrisp_res, key=lambda x: int(x, 2))
    temp = True

    for index  in range(len(qrisp_keySort)):
        # get the index for qml_res array --> can trafo to int from binary 
        index2 = int(qrisp_keySort[index], 2)
        if qrisp_res[qrisp_keySort[index]] < 0.13:
            # small results are problematic... might have to adjust for accuracy
            continue
        # we allow for slight deviations in result probabilities

        if not qrisp_res[qrisp_keySort[index]]*0.8 <= qml_res[index2] <= qrisp_res[qrisp_keySort[index]]*1.2:

            temp = False
            break

    assert temp

def wrapper():
    for index in range(20):
        randomized_ciruit_testing()
