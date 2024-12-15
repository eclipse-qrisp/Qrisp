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

# This file implements a performance benchmark of simulating molecules using PySCF data
# We compare the performance of the algorithm implemented in OpenFermion
# (described here: https://quantumai.google/openfermion/tutorials/intro_workshop_exercises#hamiltonian_simulation_with_trotter_formulas)
# vs. the Qrisp implementation https://www.qrisp.eu/general/tutorial/H2.html

# The amount of RZ gates is proportional to the required computation effort 
# because Hamiltonian Simulation is an algorithm, which requires fault tolerant devices.
# In this device model, arbitrary angle gates (like RZ, P or RX) need to be 
# synthesized by a sequence of Clifford+T gates. More details here: https://arxiv.org/abs/1403.2975

import numpy as np
import openfermion as of
import openfermionpyscf as ofpyscf
import cirq
from qrisp import QuantumCircuit, QuantumVariable
from qrisp.operators import FermionicOperator
from scipy.sparse import linalg
from numpy.linalg import norm

# Function to indicate whether an Operation object contributes
# to the RZ-depth
# For more details check "Gate-speed aware compilation" here:
# https://www.qrisp.eu/reference/Core/generated/qrisp.QuantumSession.compile.html
def rz_depth_indicator(op):
    if op.name in ["rz", "p", "u1", "rx", "ry"]:
        return 1
    # Some u3 operations in the OpenFermion circuit have u3 gates where the parameters 
    # are still  Clifford. We filter out the ones where all three are Clifford 
    # using some function that is defined below. For the other ones, we assume 
    # that they can be implemented with two arbitrary angle rotations 
    # (which is very conservative - normally you need three!)
    if op.name == "u3":
        return 2
    else:
        return 0

# Function to benchmark simulation of molecules
def benchmark_hamiltonian_simulation(molecule_data):
    geometry, basis, multiplicity, charge = molecule_data
    
    # Generate Hamiltonian
    hamiltonian = ofpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)
    n_qubits = of.count_qubits(hamiltonian)
    
    # OpenFermion implementation taken from here:
    # https://quantumai.google/openfermion/tutorials/intro_workshop_exercises#hamiltonian_simulation_with_trotter_formulas
    def openfermion_circuit():
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit(
            of.simulate_trotter(qubits, hamiltonian, time=1.0, n_steps=1, order=0, algorithm=of.LOW_RANK)
        )
        # Convert to Qrisp circuit via OpenQASM
        return QuantumCircuit.from_qasm_str(circuit.to_qasm())
    
    # Qrisp implementation taken from here:
    # https://www.qrisp.eu/general/tutorial/H2.html
    def qrisp_circuit():
        H = FermionicOperator.from_openfermion(hamiltonian_ferm_op)
        qv = QuantumVariable(n_qubits)
        U = H.trotterization()
        U(qv, steps=1, t=1)
        qc = qv.qs.compile(workspace = len(qv)//4, gate_speed = rz_depth_indicator)
        # Move the workspace qubits up to facilitate computation of the unitary        
        i = qc.num_qubits() - 1
        while i >= len(qv):
            qc.qubits.insert(0, qc.qubits.pop(-1))
            i -= 1
        return qc.transpile()

    # Function to determine the RZ count
    def count_rz(qc):
        ops_count_dic = qc.transpile().count_ops()
        return ops_count_dic.get("rz", 0) + ops_count_dic.get("u1", 0) + 2*ops_count_dic.get("u3", 0) + ops_count_dic.get("p", 0)
    
    # Function to count the CNOT gates
    # We don't count the SWAP gates within the OpenFermion circuit since these SWAP 
    # gates ensure linear connectivity.
    # The Qrisp algorithm can also be run on linear connectivity with a minimal number of
    # SWAPS, which is however not yet implemented.
    def count_cnot(qc):
        ops_count_dic = qc.count_ops()
        return ops_count_dic.get("cx", 0) + ops_count_dic.get("cz", 0) + 3*ops_count_dic.get("compiled_gidney_mcx", 0) + 0.5*ops_count_dic.get("compiled_gidney_mcx_inv", 0)
    
    # Function to compute the RZ depth
    def rz_depth(qc):
        return qc.depth(depth_indicator = rz_depth_indicator)
    
    # Function to compute the precision of the quantum circuit to the exact unitary
    def unitary_distance(qc, exact_unitary):
        qc_unitary = qc.get_unitary()
        qc_unitary = qc_unitary[:exact_unitary.shape[0],:exact_unitary.shape[0]]
        
        # Since OpenFermion doesn't properly implement the global phase 
        # (this is a bug!) we manually correct.
        argmax = np.argmax(np.abs(qc_unitary).ravel())
        qc_phase = np.angle(qc_unitary.ravel()[argmax])
        exact_phase = np.angle(exact_unitary.ravel()[0, argmax])
        
        qc_unitary = np.exp(1j*(exact_phase-qc_phase))*qc_unitary
        
        # Compute the distance
        return norm(qc_unitary - exact_unitary)


    # Compute circuits
    qc_of = openfermion_circuit()
    qc_qrisp = qrisp_circuit()
    
    # Compute metrics
    
    # Some gates in the OpenFermion implementation are arbitrary angle but have a
    # parameter value, which indicates that they are clifford (for instance (RZ(pi) = Z)). 
    # We filter out these gates since we only want to count arbitrary angle 
    # gates that need to be synthesized since these are the costly resource.
    clifford_filtered_qc = filter_clifford(qc_of)
    
    of_rz_count = count_rz(clifford_filtered_qc)
    of_cx_count = count_cnot(clifford_filtered_qc)
    of_rz_depth = rz_depth(clifford_filtered_qc)
    
    qrisp_rz_count = count_rz(qc_qrisp)
    qrisp_cx_count = count_cnot(qc_qrisp)
    qrisp_rz_depth = rz_depth(qc_qrisp)
    
    # If the circuit has more than 10 qubits, computing the unitary is not
    # feasible.
    # For the cases below 10 qubits, both Qrisp and OpenFermion implement almost the exact same
    # precision.
    
    if qc_qrisp.num_qubits() < 10:
        # Compute exact unitary
        hamiltonian_jw_sparse = of.get_sparse_operator(of.jordan_wigner(hamiltonian_ferm_op))
        exact_unitary = linalg.expm(-1j * hamiltonian_jw_sparse).todense()
        of_precision = unitary_distance(qc_of, exact_unitary)
        qrisp_precision = unitary_distance(qc_qrisp, exact_unitary)
    else:
        of_precision = 0
        qrisp_precision = 0
    
    return {
        "OpenFermion": {
            "RZ count": of_rz_count,
            "CX count": of_cx_count,
            "RZ depth": of_rz_depth,
            "Precision": of_precision
        },
        "Qrisp": {
            "RZ count": qrisp_rz_count,
            "CX count": qrisp_cx_count,
            "RZ depth": qrisp_rz_depth,
            "Precision": qrisp_precision
        }
    }

# Some gates in the OpenFermion implementation are arbitrary angle but have a
# parameter value, which indicates that they are clifford. We filter out these
# gates since we only want to count arbitrary angle gates that actually need to
# be synthesized (Clifford gates don't).
def filter_clifford(qc):
    
    qc_new = qc.clearcopy()
    
    # Filter out non-clifford operations
    for instr in qc.data:
        
        if len(instr.op.params) > 0:
            par = instr.op.params[0]
        
        if instr.op.name == "ry":
            if abs(par-np.pi/2) < 1E-4:
                qc_new.sx_dg(instr.qubits)
                qc_new.s(instr.qubits)
                qc_new.sx(instr.qubits)
        elif instr.op.name == "rx":
            if abs(par-np.pi/2) < 1E-4:
                qc_new.h(instr.qubits)
                qc_new.s(instr.qubits)
                qc_new.h(instr.qubits)
        elif instr.op.name in ["rz", "u1"]:
            if abs(par) < 1E-4:
                continue
            elif abs(abs(par) - np.pi) < 1E-4:
                qc_new.z(instr.qubits)
            elif abs(abs(par) - np.pi/2) < 1E-4:
                if par < 0:
                    qc_new.s_dg(instr.qubits)
                else:
                    qc_new.s(instr.qubits)
            elif abs(abs(par) - np.pi/4) < 1E-4:
                if par < 0:
                    qc_new.t_dg(instr.qubits)
                else:
                    qc_new.t(instr.qubits)                
            else:
                qc_new.append(instr)
        elif instr.op.name == "u3":
            
            for par in instr.op.params:
                if par not in [i*np.pi/4 for i in range(8)]:
                    break
            else:
                # For the case that all parameters are clifford,
                # we don't append any gate, so it is not counted.
                continue
            
            qc_new.append(instr)        
                
        else:
            qc_new.append(instr)
            
    return qc_new

# Compute the benchmark results

molecule_list = [
    ([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.8))], "sto-3g", 1, 0),
    ([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))], "sto-3g", 1, 0),
    ([("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.3))], "sto-3g", 1, 1),
    ([("B", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.2))], "sto-3g", 1, 0),
    ([("O", (0.0, 0.0, 0.0)), ("H", (0.757, 0.586, 0.0)), ("H", (-0.757, 0.586, 0.0))], "sto-3g", 1, 0),
    ([("N", (0.0, 0.0, 0.0)),
      ("H", (0.0, -0.934, -0.365)),
      ("H", (0.809, 0.467, -0.365)),
      ("H", (-0.809, 0.467, -0.365))], "sto-3g", 1, 0),
]

# Lists to store results
molecules = []
of_rz_counts = []
of_rz_depths = []
of_precisions = []
qrisp_rz_counts = []
qrisp_rz_depths = []
qrisp_cx_count = []
of_cx_count = []
qrisp_precisions = []

# Run benchmarks for each molecule
for idx, molecule_data in enumerate(molecule_list):
    print(molecule_data)
    results = benchmark_hamiltonian_simulation(molecule_data)
    
    molecules.append(f"Molecule {idx+1}")
    of_rz_counts.append(results["OpenFermion"]["RZ count"])
    of_rz_depths.append(results["OpenFermion"]["RZ depth"])
    of_precisions.append(results["OpenFermion"]["Precision"])
    qrisp_rz_counts.append(results["Qrisp"]["RZ count"])
    qrisp_rz_depths.append(results["Qrisp"]["RZ depth"])
    qrisp_precisions.append(results["Qrisp"]["Precision"])
    qrisp_cx_count.append(results["Qrisp"]["CX count"])
    of_cx_count.append(results["OpenFermion"]["CX count"])


# Plot the results

import matplotlib.pyplot as plt

# Set up the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
x = np.arange(len(molecules))
width = 0.35
molecules = ["$H_2$", "LiH", "BeH", "BH", "$H_2O$", "$NH_3$"]
ax1.grid()
ax2.grid()

# Plot RZ counts
ax1.bar(x + width/2, qrisp_rz_counts, width, label="Qrisp 0.5", color='#20306f', alpha=1, zorder = 100)
ax1.bar(x - width/2, of_rz_counts, width, label="Google: OpenFermion", color='#FFBA00', alpha=1, zorder = 100)
ax1.set_ylabel('RZ Count')
ax1.set_title('RZ Count Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(molecules)
ax1.legend()


# Plot RZ depths
ax2.bar(x + width/2, qrisp_rz_depths, width, label="Qrisp 0.5", color='#20306f', alpha=1, zorder = 100)
ax2.bar(x - width/2, of_rz_depths, width, label="Google: OpenFermion", color='#FFBA00', alpha=1, zorder = 100)
ax2.set_ylabel('RZ Depth')
ax2.set_title('RZ Depth Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(molecules)
ax2.legend()


plt.tight_layout()
plt.show()
plt.savefig("performance_comparison.png", dpi = 300)
