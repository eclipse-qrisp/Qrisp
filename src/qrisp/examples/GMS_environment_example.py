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

from qrisp import QuantumVariable, QFT, cp, h, p, x
from qrisp.environments import GMSEnvironment
from qrisp.alg_primitives.arithmetic import QuantumFloat

qv1 = QuantumVariable(1)
qv2 = QuantumVariable(1)


# merge([qv1, qv2])

qs = qv1.qs

test = GMSEnvironment()

h(qv1)
x(qv2)


with test:
    cp(np.pi / 2, qv1[0], qv2[0])
    p(np.pi / 2, qv1[0])
    p(np.pi / 2, qv2[0])


h(qv1)

print(qv1.get_measurement())
print(qs)


# %%

qf = QuantumFloat(3)

qf.encode(1)

QFT(qf, use_gms=True)

sv = qf.qs.statevector("array", plot=True)
