
import numpy as np 
import scipy
import matplotlib.pyplot as plt 

#from  8 up until 26 nodes
node_count = list(range(8,28,2))

#qrisp04_values --> from galois_arithmetic branch
#first list is depth (named it gc=gate_count)
#second list is total time
#third list is compile_time
qri04_gc = [81, 93, 93, 111, 126, 105, 135, 162, 102, 111]
qri04_time = [6.175459146499634, 13.721542596817017, 26.102425813674927, 57.216010093688965, 106.95222163200378, 132.99402260780334, 112.52425336837769, 102.72538018226624, 244.73981881141663, 2450.7348425388336]
qri04_compiletime =[0.04234766960144043, 0.07568597793579102, 0.08389472961425781, 0.10111689567565918, 0.11180233955383301, 0.33979058265686035, 0.2864654064178467, 0.1528022289276123, 0.16391611099243164, 0.3011641502380371]

#qrisp_values - up until 20 nodes
qri_gc = [246, 306, 366, 426, 486, 546, 606]
qri_time = [3.0877020359039307, 5.265552520751953, 10.789706230163574, 21.89113187789917, 57.72945308685303, 85.45771765708923, 46.431854248046875]
qri_compiletime = [0.03137803077697754, 0.04635977745056152, 0.05003046989440918, 0.05824756622314453, 0.147796630859375, 0.14275574684143066, 0.16557812690734863]

#qiskit_values
qis_gc = [726, 1130, 1614, 2202, 2874, 3638]
qis_time = [134.733, 242.202, 4899.137, 2075.768, 5724.088, 19419.096]
qis_compiletime = [0.9314613342285156, 1.4867534637451172, 2.1324996948242188, 2.785449743270874, 4.961149215698242]


# First Plot (Quantum Circuit Depth)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12.5, 4))

ax1.set_xlabel("MaxCut Instance Node Count", fontsize=15, color="#444444", fontname="Segoe UI")
ax1.set_ylabel("Quantum Circuit Depth", fontsize=15, color="#444444", fontname="Segoe UI")
ax1.set_xticks(ticks = range(8, 27, 4))
ax1.plot(node_count[:len(qri04_gc)], qri04_gc, c='#20306f', marker="o", linestyle='solid', linewidth=3, label="Qrisp 0.4")
ax1.plot(node_count[:len(qri_gc)], qri_gc, c="#7d7d7d", marker="o", linestyle='solid', linewidth=3, label="Qrisp 0.3")
ax1.plot(node_count[:len(qis_gc)], qis_gc, c="#6929C4", marker="o", linestyle='solid', linewidth=3, label="Qiskit")

ax1.legend(fontsize=14, labelcolor='linecolor')
ax1.grid()

# Second Plot (Compiletime)
# fig, ax2 = plt.subplots(figsize=(12, 5))

ax2.set_xlabel("MaxCut Instance Node Count", fontsize=15, color="#444444", fontname="Segoe UI")
ax2.set_ylabel("Compile time in [s]", fontsize=15, color="#444444", fontname="Segoe UI")
ax2.set_xticks(ticks = range(8, 27, 4))
ax2.plot(node_count[:len(qri04_compiletime)], qri04_compiletime, c='#20306f', marker="o", linestyle='solid', linewidth=3, label="Qrisp 0.4", zorder=3)
ax2.plot(node_count[:len(qri_compiletime)], qri_compiletime, c="#7d7d7d", marker="o", linestyle='solid', linewidth=3, label="Qrisp 0.3")
ax2.plot(node_count[:len(qis_compiletime)], qis_compiletime, c="#6929C4", marker="o", linestyle='solid', linewidth=3, label="Qiskit")

ax2.legend(fontsize=14, labelcolor='linecolor')
ax2.grid()

# Show both plots side by side
plt.tight_layout()

# plt.savefig("compiler_plot.svg", format = "svg", dpi = 80, bbox_inches = "tight")

plt.show()
