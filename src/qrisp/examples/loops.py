"""
/********************************************************************************
* Copyright (c) 2023 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2 
* or later with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0
********************************************************************************/
"""


from qrisp import QuantumFloat, h, qRange
import time

# Create some QuantumFloats
n = QuantumFloat(5, signed=False)
qf = QuantumFloat(4)
# Initialize the value 4
n[:] = 4

# Apply H-gate to 0-th qubit
h(n[0])
n_results = n.get_measurement()


# Perform successive addition of increasing numbers
for i in qRange(n):
    qf += i

start_time = time.time()
print(qf)  # Yields {10.0: 0.5, 15.0: 0.5}

print("Excpected outcomes:", [n * (n + 1) / 2 for n in n_results.keys()])
# Yields n*(n+1)/2 as expected
print(start_time - time.time())
