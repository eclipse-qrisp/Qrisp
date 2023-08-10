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

# Created by nik40643 at 23.06.2023

import time
import random
import numpy as np

from qrisp.core import QuantumSession, QuantumVariable
from qrisp import rz, rx, ry, p, cp
import qiskit.circuit.library.standard_gates as qsk_gates
from qiskit.circuit import Parameter, QuantumCircuit, ParameterExpression
#from qrisp.circuit.quantum_circuit import rxx, ryy, rzz
from sympy import Symbol, simplify, symbols
from qrisp.interface import circuit_converter, convert_from_qiskit

def test_abstract_params_printing():
    qiskit_qc = QuantumCircuit(5)
    
    params = symbols("c:5")
    #gates_list = [qsk_gates.RXGate(),qsk_gates.RYGate(),qsk_gates.RZGate(),qsk_gates.PhaseGate(),qsk_gates.CPhaseGate()]
    qiskit_ins_list = []
    theta = Parameter(str(params[0]))
    qiskit_qc.append(qsk_gates.RXGate(theta ), [0],[])
    qiskit_qc.append(qsk_gates.RYGate(Parameter(str(params[1]))), [1],[])
    qiskit_qc.append(qsk_gates.RZGate(Parameter(str(params[2]))), [2],[])
    qiskit_qc.append(qsk_gates.PhaseGate(Parameter(str(params[3]))), [3],[])
    #qiskit_qc.append(qsk_gates.CPhaseGate(Parameter(str(params[4]))), [4],[])
    #print(str(qiskit_qc.qasm()))
    str(qiskit_qc)
    #print(qiskit_qc)
    
    
    qv = QuantumVariable(5)
    rx(params[0], qv[0])
    ry(params[1], qv[1])
    rz(params[2], qv[2])
    p(params[3], qv[3])
    #print(str(qv.qs))
    #print(qv.qs)
    qisk = circuit_converter.convert_to_qiskit(qv.qs)
    #print(qisk)
    
    #assert str(qisk) == str(qiskit_qc)
    qisk_list = []
    for gate in qisk.data:
        qisk_list.append((gate[0].name,gate[0].params))
    
    qiskit_list = []
    for gate in qiskit_qc.data:
        qiskit_list.append((gate[0].name,gate[0].params))
    
    assert len(qisk_list) == len(qisk_list)
    for index in range(len(qisk_list)):
        assert str(qiskit_list[index]) == str(qisk_list[index])
    
    
    ### After conversion from Qiskit
    
    theto = Symbol("theta")
    gummo = Symbol("gamma")
    #theto = Parameter("theta")
    #gummo = Parameter("gamma")
    #delto = theto +gummo
    delto = 2*Parameter("theta") 
    
    n = 5
    
    qc = QuantumCircuit(5, 1)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    qc.rz(delto, range(5))
    #qc.barrier()
    for i in reversed(range(n-1)):
        qc.cx(i, i+1)
    qc.h(0)
    qc.measure(0, 0)
    
    qrisp_qc = convert_from_qiskit(qc)
    print(qrisp_qc)
    
    
    
    assert len(list((qrisp_qc.abstract_params))) == len(list(qc.parameters ))   
    
    delto = 2*Parameter("theta") + Parameter("gamma")
    n = 5
    qc = QuantumCircuit(5, 1)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    qc.rx(delto, range(5))
    #qc.barrier()
    for i in reversed(range(n-1)):
        qc.cx(i, i+1)
    qc.h(0)
    qc.measure(0, 0)
    
    qrisp_qc = convert_from_qiskit(qc)
    
    
    assert len(list((qrisp_qc.abstract_params))) == len(list(qc.parameters ))   
    
    
    # write some test to assert lambd_expr = ParameterExpression.sympify(p) 