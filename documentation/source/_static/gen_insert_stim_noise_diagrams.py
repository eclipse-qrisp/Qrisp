"""Generate Stim SVG diagrams for insert_stim_noise documentation."""

from qrisp import QuantumCircuit, insert_stim_noise, layerize

# Two rounds of repetition-code syndrome extraction: data qubits 0 and 2,
# ancilla qubit 1, one detector per round, rounds separated by a barrier.
qc = QuantumCircuit(3)
for _ in range(2):
    qc.cx(0, 1)
    qc.cx(2, 1)
    clbit = qc.add_clbit()
    qc.measure(1, clbit)
    qc.parity([clbit])
    qc.reset(1)
    qc.barrier(qc.qubits)

# Before
svg_before = str(qc.to_stim().diagram(type="timeline-svg"))
with open("documentation/source/_static/insert_stim_noise_before.svg", "w") as f:
    f.write(svg_before)
print("Saved before diagram")

# After.  layerize puts every channel back into the time step it was inserted
# for and marks the boundaries, so the diagram shows one column and one TICK per
# time step instead of the lazily placed idle noise of the raw output.
noisy = insert_stim_noise(
    depolarize_1_strength=0.001,
    depolarize_2_strength=0.01,
    X_error_strength=0.005,
)(qc)
noisy = layerize(insert_barriers=True)(noisy)
svg_after = str(noisy.to_stim().diagram(type="timeline-svg"))
with open("documentation/source/_static/insert_stim_noise_after.svg", "w") as f:
    f.write(svg_after)
print("Saved after diagram")
