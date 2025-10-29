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

import numpy as np

from qrisp import QuantumVariable, cp, h, transpile, x
from qrisp.misc.GMS_tools import (
    gms_multi_cp_gate,
    gms_multi_cx_fan_out,
)

n = 5
ctrl = QuantumVariable(1)
target = QuantumVariable(n)


h(ctrl)
ctrl.qs.append(gms_multi_cx_fan_out(n, use_uniform=True), list(target) + [ctrl])

print(target.qs.statevector())

print(transpile(ctrl.qs, 2))


# %%

n = 3

theta = [np.pi / (i + 1) for i in range(n - 1)]

qv_0 = QuantumVariable(n)

qv_0.qs.append(gms_multi_cp_gate(n - 1, theta), qv_0.reg)


qv_1 = QuantumVariable(n)

for i in range(n - 1):
    cp(theta[i], qv_1[i], qv_1[n - 1])


unitary = qv_0.qs.get_unitary()
print(qv_0.qs.get_unitary(decimals=4))


print("Is equal:", qv_0.qs.compare_unitary(qv_1.qs))


print("Is diagonal:", np.linalg.norm(unitary - np.diag(np.diagonal(unitary))) < 1e-3)
print(np.round(np.angle(np.diagonal(unitary)) % (2 * np.pi) / (np.pi), 3))
