"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
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


from qrisp.alg_primitives.arithmetic.adders import gidney_adder
from qrisp.core import cx
from qrisp.environments import control, invert
from qrisp.qtypes import QuantumBool


# @qache(static_argnames = "inpl_adder")
def jasp_mod_adder(a, b, modulus, inpl_adder=gidney_adder, ctrl=None):

    reduction_not_necessary = QuantumBool()
    # sign = QuantumBool()
    sign = b[-1]

    if isinstance(a, int):
        a = a % modulus

    # b = list(b) + [sign[0]]

    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)

    with invert():
        inpl_adder(modulus, b)

    cx(sign, reduction_not_necessary[0])

    with control(reduction_not_necessary[0]):
        inpl_adder(modulus, b)

    with invert():
        if ctrl is None:
            inpl_adder(a, b)
        else:
            with control(ctrl):
                inpl_adder(a, b)

    cx(sign, reduction_not_necessary[0])
    reduction_not_necessary.flip()

    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)

    reduction_not_necessary.delete()
