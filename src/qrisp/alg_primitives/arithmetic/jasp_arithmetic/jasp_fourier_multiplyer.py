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

from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_fourier_adder import jasp_fourier_adder
from qrisp.jasp import qache, jrange
from qrisp.qtypes import QuantumFloat
from qrisp.environments import control

@qache
def jasp_fourier_multiplyer(a, b):
    
    s = QuantumFloat(a.size + b.size)
    
    for i in jrange(b.size):
        with control(b[i]):
            jasp_fourier_adder(a, s[i:i+a.size+1])
    
    return s