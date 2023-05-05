"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
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
