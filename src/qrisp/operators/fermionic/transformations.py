"""
********************************************************************************
* Copyright (c) 2024 the Qrisp authors
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

from functools import cache

from qrisp.operators.qubit.qubit_operator import QubitOperator
from qrisp.operators.qubit.qubit_term import QubitTerm


# Jordan-Wigner annihilation operaror
# @cache
def a_jw(j):
    d1 = {i: "Z" for i in range(j)}
    d1[j] = "X"
    d2 = {i: "Z" for i in range(j)}
    d2[j] = "Y"
    return QubitOperator({QubitTerm(d1): 0.5, QubitTerm(d2): 0.5j})


# Jordan-Wigner creation operator
# @cache
def c_jw(j):
    d1 = {i: "Z" for i in range(j)}
    d1[j] = "X"
    d2 = {i: "Z" for i in range(j)}
    d2[j] = "Y"
    return QubitOperator({QubitTerm(d1): 0.5, QubitTerm(d2): -0.5j})


@cache
def jordan_wigner(ladder):
    if ladder[1]:
        return c_jw(ladder[0])
    else:
        return a_jw(ladder[0])
