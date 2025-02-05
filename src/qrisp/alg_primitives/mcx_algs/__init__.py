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

from qrisp.alg_primitives.mcx_algs.multi_cx import *
from qrisp.alg_primitives.mcx_algs.amy import *
from qrisp.alg_primitives.mcx_algs.balauca import *
from qrisp.alg_primitives.mcx_algs.gidney import *
from qrisp.alg_primitives.mcx_algs.gms import *
from qrisp.alg_primitives.mcx_algs.gray import *
from qrisp.alg_primitives.mcx_algs.gray_pt import *
from qrisp.alg_primitives.mcx_algs.jones import *
from qrisp.alg_primitives.mcx_algs.maslov import *
from qrisp.alg_primitives.mcx_algs.yong import *

# Interface function to quickly change between different implementations of
# multi controlled not gates
def multi_cx(n, method=None):
    # from qrisp.circuit import transpile

    if method == "gms":
        return gms_multi_cx(n)

    elif method in ["gray_pt"]:
        return pt_multi_cx(n)

    elif method == "gray_pt_inv":
        return pt_multi_cx(n).inverse()

    elif method in ["gray", "auto", None]:
        return gray_multi_cx(n)
    
    elif method == "gidney":
        return GidneyLogicalAND()
    elif method == "gidney_inv":
        return GidneyLogicalAND(inv = True)
    else:
        raise Exception('method "' + method + '" not implemented')





