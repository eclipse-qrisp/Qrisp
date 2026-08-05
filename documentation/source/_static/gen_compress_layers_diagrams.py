"""Generate Stim SVG diagrams for compress_layers documentation."""
from qrisp import QuantumCircuit, compress_layers

qc = QuantumCircuit(6)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.h(5)
qc.cx(5, 4)
qc.cx(4, 3)

# Before
stim_before = qc.to_stim()
svg_before = str(stim_before.diagram(type="timeline-svg"))
with open("documentation/source/_static/compress_layers_before.svg", "w") as f:
    f.write(svg_before)
print("Saved before diagram")

# After
qc_after = compress_layers(qc)
stim_after = qc_after.to_stim()
svg_after = str(stim_after.diagram(type="timeline-svg"))
with open("documentation/source/_static/compress_layers_after.svg", "w") as f:
    f.write(svg_after)
print("Saved after diagram")
