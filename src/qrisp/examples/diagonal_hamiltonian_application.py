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

import matplotlib.pyplot as plt
import numpy as np

from qrisp import QFT, QuantumFloat, h
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
sv_phase_array = np.angle([sv_function({qf: i}) for i in x])

# Plot results
plt.plot(x, hamiltonian(x) % (2 * np.pi), "o", label="Hamiltonian")
plt.plot(x, sv_phase_array % (2 * np.pi), ".", label="Simulated phases")

plt.ylabel("Phase [radian]")
plt.xlabel("Quantum Float outcome labels")
plt.grid()
plt.legend()
plt.show()


# %%

# In this example we will demonstrate how a phase function with multiple arguments can
# be synthesized. For this we will create a phase function which encodes the fourier
# transform of different integers on the QuantumFloat x conditioned on the value of a
# QuantumChar c. We will then apply the inverse Fourier transform to x and measure the
# results.


x_size = 3


# Create Variables
x = QuantumFloat(x_size, 0, signed=False)

from qrisp import QuantumChar

c = QuantumChar()


# Bring x into uniform superposition so the phase function application yields a fourier
# transformed computation basis state
h(x)


# Bring c into partial superposition (here |a> + |b> + |c> + |d>)
h(c[0])
h(c[1])


# In order to define the hamiltonian, we can use regular Python syntax.
# The decorator "as_hamiltonian" turns it into a function that takes Quantum Variables
# as arguments. The decorator will add the keyword argument t to the function which
# mimics the t in exp(i*H*t).
@as_hamiltonian
def apply_multi_var_hamiltonian(c, x):
    if c == "a":
        k = 2
    elif c == "b":
        k = 2
    elif c == "c":
        k = 3
    else:
        k = 4

    # Return phase value
    # This is the phase distribution of the Fourier-transform
    # of the computational basis state |k>
    return k * x * 2 * np.pi / 2**x_size


# Apply Hamiltonian
apply_multi_var_hamiltonian(c, x, t=1)


# Apply inverse Fourier transform
QFT(x, inv=True)


# Acquire measurement results
for mes_res in multi_measurement([c, x]):
    print(mes_res)

print(x.qs)
