import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile as qiskit_transpile
from qiskit.circuit.library import PhaseEstimation
from qiskit.quantum_info import Statevector
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit

# Requires Qiskit < 2.0 for Qiskit nature support
# --- Qiskit Imports ---
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper


# --- Qiskit Setup ---
def prepare_qiskit_hamiltonian():
    driver = PySCFDriver(
        atom="Be 0 0 0; H 0 0 1.3",
        basis="sto3g",
        charge=1,
        spin=0,
    )
    problem = driver.run()
    mapper = ParityMapper()
    hamiltonian = problem.hamiltonian.second_q_op()
    H = mapper.map(hamiltonian)
    return generate_time_evolution_circuit(H, time=1.6)


hamiltonian_simulation = prepare_qiskit_hamiltonian()


def count_qpe_ops_qiskit(n):
    qpe_qc = PhaseEstimation(n, hamiltonian_simulation)
    qc = qiskit_transpile(qpe_qc, basis_gates=["cx", "u3"], optimization_level=0)
    return qc.count_ops()


# %%

# --- Qrisp Imports ---
from pyscf import gto

from qrisp import *
from qrisp.operators import FermionicOperator

# --- Qrisp Setup ---
BeH_mol = gto.M(atom="Be 0 0 0; H 0 0 1.3", basis="sto-3g", spin=0, charge=1)
H_ferm = FermionicOperator.from_pyscf(BeH_mol)
U = qache(H_ferm.trotterization())


@count_ops(meas_behavior="1")
def count_ops_qrisp(n):
    qv = QuantumVariable(H_ferm.find_minimal_qubit_amount())
    qpe_res = QPE(qv, U, n)
    return measure(qpe_res)


import time

t0 = time.time()
count_ops_qrisp(2)
print("Just-in-time compilation time: ", t0 - time.time())
# %%

# --- Benchmarking ---
precisions = list(range(1, 20))
results = {
    "qiskit": {"times": [], "cx_counts": []},
    "qrisp": {"times": [], "cx_counts": []},
}

import timeit

for n in precisions:
    print("Iteration: ", n)
    # Qiskit
    if n < 8:
        t0 = time.time()
        ops_qiskit = count_qpe_ops_qiskit(n)
        t1 = time.time()
        print("Qiskit time: ", t1 - t0)
        results["qiskit"]["times"].append(t1 - t0)
        results["qiskit"]["cx_counts"].append(ops_qiskit.get("cx", 0))

    ops_qrisp = count_ops_qrisp(n)

    execution_time = (
        timeit.timeit(f"count_ops_qrisp({n})", globals=globals(), number=10) / 10
    )
    results["qrisp"]["times"].append(execution_time)
    print("Qrisp time: ", execution_time)
    results["qrisp"]["cx_counts"].append(
        ops_qrisp.get("cx", 0) + ops_qrisp.get("cz", 0)
    )

# %%
n_qiskit = len(results["qiskit"]["times"])
fit_res = np.polyfit(
    list(range(1, n_qiskit + 1)), np.log(results["qiskit"]["times"]), deg=1
)
fit_data = np.exp(fit_res[0] * np.arange(1, len(precisions) + 1) + fit_res[1])

# %%
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(6.5, 5))


# Plot 1: Compilation Time (left axis)
color2 = "#d62728"
ax1.set_xlabel("QPE Precision (bits)")
ax1.set_xticks(list(range(1, 21, 4)))
ax1.plot(
    precisions[:n_qiskit],
    results["qiskit"]["cx_counts"],
    marker="x",
    label="Qiskit #CX",
    color=color2,
    linestyle="dashed",
    linewidth=2,
)
ax1.set_ylabel("CX Gate Count (log scale)", color=color2)
ax1.plot(
    precisions[:n_qiskit],
    results["qrisp"]["cx_counts"][:n_qiskit],
    marker="x",
    label="Qrisp #CX",
    color="#ff7f0e",
    linestyle="dashed",
    linewidth=2,
)
ax1.set_yscale("log")
ax1.tick_params(axis="y", labelcolor=color2)

# Create second y-axis for CX Gate Count
ax2 = ax1.twinx()
color1 = "#6929C4"
ax2.plot(
    precisions,
    fit_data,
    marker=".",
    label="Extrapolation",
    fillstyle="none",
    color="#7d7d7d",
)
ax2.set_ylabel("Resource Computation Time (s, log scale)", color=color1)
ax2.plot(
    precisions[:n_qiskit],
    results["qiskit"]["times"],
    marker="o",
    label="Qiskit Time",
    color=color1,
    linestyle="solid",
    linewidth=3,
)
ax2.plot(
    precisions,
    results["qrisp"]["times"],
    marker="o",
    label="Qrisp Time",
    color="#20306f",
    linestyle="solid",
    linewidth=3,
)
ax2.set_yscale("log")
ax2.tick_params(axis="y", labelcolor=color1)
ax2.grid(True)


# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

plt.title(
    "Resource Efficiency and Computation Speed \n vs. Beryllium hydride QPE Precision"
)
plt.tight_layout()
# plt.savefig("resource_computation_comparison.png", dpi = 300)
plt.show()
