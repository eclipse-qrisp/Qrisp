"""
\********************************************************************************
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
********************************************************************************/
"""


import matplotlib.pyplot as plt
import numpy as np

from qrisp import QuantumFloat, h, QFT
from qrisp.misc import as_hamiltonian, multi_measurement


# Hamiltonian function
# This function should take an element from the outcome labels of qf and turns it into a
# phase. In this case we are handling a QuantumFloat, so the input is a float.
# Another case could be where we are handling a QuantumChar
# In this case the function should process a char
def hamiltonian(x):
    return np.pi * np.sin(x**2 * np.pi * 2) * x
    # If we were processing a QuantumChar, the hamiltonian could look like this:
    # if x == "a":
    #     return 0.5*np.pi
    # elif x == "b":
    #     return 0.25*np.pi
    # else:
    #     return np.pi


# Create quantum float
qf = QuantumFloat(5, -5, signed=True)


# Bring qf in uniform superposition state in order to be able to see which phase has
# been applied to which state
h(qf)

# Apply hamiltonian
qf.app_phase_function(hamiltonian)


# Now, we visualize the result

# Simulate phases
sv_function = qf.qs.statevector("function")

# Prepare x array (for plotting)
x = np.array([qf.decoder(i) for i in range(2**qf.size)])

sv_phase_array = list(np.angle([sv_function({qf: i}) for i in x])%(2*np.pi))
temp = list(sv_phase_array)
sv_phase_array.sort(key = lambda i : x[temp.index(i)])
sv_phase_array = np.array(sv_phase_array)
x = list(x)
x.sort()
x = np.array(x)
# Convert radians to fractions of π
pi_fractions = sv_phase_array



# Set custom y-axis ticks and labels for fractions of π
pi_labels = ['-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π']





# Plot results
plt.plot(x, (hamiltonian(x) +np.pi) % (2 * np.pi)/(np.pi), "-", label="Input Hamiltonian", color = "#7d7d7d", linewidth = 3)
plt.plot(x, (sv_phase_array +np.pi)% (2 * np.pi)/(np.pi), "o", label="Statevector Simulation", color = "#20306f", markersize =8)

plt.yticks(np.arange(0, 2.25, 0.25)[::2], pi_labels[::2], color = "#444444", fontname = "Segoe UI")
plt.xticks(x[::12], 2*x[::12], color = "#444444", fontname = "Segoe UI")

plt.ylabel("Δφ", fontsize = 18, color = "#444444", fontname = "Segoe UI")
plt.xlabel("QuantumFloat value", fontsize = 18, color = "#444444", fontname = "Segoe UI")
plt.grid()
plt.legend(loc = "upper center", fontsize = 13, labelcolor = "#444444")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout() 
plt.show()
# plt.savefig("hamiltonian.svg", format = "svg", dpi = 80, bbox_inches = "tight")