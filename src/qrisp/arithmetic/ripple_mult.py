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


from qrisp import cx, x
from qrisp.arithmetic import create_output_qf, inpl_add


def ripple_mult(factor_1, factor_2):
    if factor_1.signed or factor_2.signed:
        raise Exception("Signed ripple multiplication currently not supported")

    s = create_output_qf([factor_1, factor_2], op="mul")

    n = factor_1.size - int(factor_1.signed)

    factor_2.exp_shift(n)

    s.init_from(factor_2)

    factor_2.exp_shift(-n)

    x(s)

    inpl_add(s, factor_2)

    for i in range(n):
        cx(factor_1[i], s)

        inpl_add(s, factor_2)

        cx(factor_1[i], s)
        factor_2.exp_shift(1)

    x(s)

    factor_2.exp_shift(-n)
    s.exp_shift(-1)

    return s
