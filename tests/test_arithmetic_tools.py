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
from qrisp import *
from qrisp.jasp import *

up_bound = 4
for N in range(2, up_bound):
    for k in range(2**N):
        for j in range(2**N):
            
            a = QuantumFloat(N)
            b = QuantumFloat(N)
            a[:] = j
            b[:] = k
            res = qmax(a,b).get_measurement()
            assert max(j,k) == list(res.keys())[0]
