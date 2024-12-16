"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

import numpy as np

from qrisp.jasp import qache, jrange, DynamicQubitArray
from qrisp.core import swap, h, p, cp
from qrisp.qtypes import QuantumFloat
from qrisp.environments import control, conjugate

@qache
def qft(qv):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    n = qv.size
    
    for i in jrange(n):
        # pass
        h(qv[n - 1- i])
        for k in jrange(n - i-1):
            cp(2. * np.pi / pow(2., (k + 2)), qv[n - 1 - (k + i + 1)], qv[n - 1 -i])
    
    for i in jrange(n//2):
        swap(qv[i], qv[n-i-1])
            

@qache
def jasp_fourier_adder(a, b):

    with conjugate(qft)(b):
        if isinstance(a, (QuantumFloat, DynamicQubitArray)):
            for i in jrange(a.size):
                with control(a[i]):
                    for j in jrange(b.size):
                        p(np.pi*2.**(j-b.size+1+i), b[j])
    
                    
        else: 
            for i in jrange(b.size):
                p(a*np.pi*2.**(i-b.size+1), b[i])
