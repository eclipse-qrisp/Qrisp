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

# -*- coding: utf-8 -*-


import random

import numpy as np

from qrisp import QuantumVariable, cp, rz
from qrisp.misc.GMS_tools import GZZ_converter  # GXX_converter

n = 5

qv = QuantumVariable(n)

# Generate random phase-only circuit
for i in range(n):
    for j in range(n):
        if random.randint(0, 1):
            if i == j:
                rz(np.pi / 2 ** random.randint(0, 3), qv[i])
            else:
                cp(np.pi / 2 ** random.randint(0, 3), qv[i], qv[j])


print(qv.qs)
# Apply converter
qc = GZZ_converter(qv.qs)
print(qc)
print("Is equal:", qv.qs.compare_unitary(qc, precision=4))


unitary = qc.get_unitary(4)

unitary_2 = qv.qs.get_unitary(4)
# unitary_2 = unitary_2*np.sum(unitary)/np.sum(unitary_2)

print("Is diagonal:", np.linalg.norm(unitary - np.diag(np.diagonal(unitary))) < 1e-3)
# print(np.round(np.angle(np.diagonal(qv.qs.get_unitary()))%(2*np.pi)/(2*np.pi), 3))
