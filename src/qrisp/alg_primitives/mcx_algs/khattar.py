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

from qrisp.core.gate_application_functions import x, cx, mcx
from qrisp.qtypes import QuantumFloat
from qrisp.environments import invert, control, conjugate
from qrisp.jasp import jlen, jrange


def cca_mcx(ctrls, target, anc):

    n = jlen(ctrls)

    # STEP 1
    # This operation is always executed regardless of the lenght of ctrls
    mcx([ctrls[0], ctrls[1]], anc[0])

    for i in jrange((n - 2) // 2):
        mcx([ctrls[2 * i + 2], ctrls[2 * i + 3]], ctrls[2 * i + 1])

    x(ctrls[:-2])

    # STEP 2
    a = n - 3 + 2 * (n & 1)
    b = n - 5 + (n & 1)
    c = n - 6 + (n & 1)

    with control(c > -1):
        mcx([ctrls[a], ctrls[b]], ctrls[c])

    with invert():
        for i in jrange((c + 1) // 2):
            mcx([ctrls[2 * i + 2], ctrls[2 * i + 1]], ctrls[2 * i])
    return ctrls, target, anc


def khattar_mcx(ctrls, target):
    N = jlen(ctrls) 

    # CASE DISTINCTION
    with control(N == 1):
        cx(ctrls[0], target[0])

    with control(N == 2):
        mcx([ctrls[0], ctrls[1]], target[0])

    with control(N == 3):
        mcx(ctrls,target[0], method='balauca') # CHANGE 
        
    with control(N == 4):
        mcx(ctrls,target[0], method='balauca') # CHANGE
        
    with control(N > 4):
        khattar_anc = QuantumFloat(1)
        with conjugate(cca_mcx)(ctrls, target, khattar_anc):
            # STEP 3
            mcx([khattar_anc[0], ctrls[0]], target[0])
        khattar_anc.delete()
