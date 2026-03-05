"""
********************************************************************************
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

# This file implements the carry generation art of the algorithm presented here:
# https://arxiv.org/abs/2304.02921

# For this we implement the Brent-Kung-Tree in a recursive manner. See:
# https://web.stanford.edu/class/archive/ee/ee371/ee371.1066/lectures/lect_04.pdf Page 13

from qrisp.core.gate_application_functions import x, cx
from qrisp.misc.utility import check_if_fresh
from qrisp.alg_primitives.arithmetic.adders.gidney import gidney_adder, cq_gidney_adder
from qrisp.alg_primitives.arithmetic.adders.incrementation import lin_incr

# Both of these functions are the semi-classical versions of the ones found in
# qrisp.alg_primitives.arithmetic.adders.qcla.quantum_quantum.qq_sum_path


def cq_sum_path(a, b, c, R, ctrl=None):

    if len(a) < len(b):
        a = a + "0" * (len(b) - len(a))

    if R == 1:
        # If R = 1, we are in the case of Drapers QCLA:
        # https://arxiv.org/abs/quant-ph/0406142
        # We can use the formula
        # S = A (+) B (+) C
        cx(c[:-1], b[1:])

        for i in range(len(a)):
            if a[i] == "1":
                x(b)
    else:
        # Execute addition using the corresponding carry values
        for i in range(len(c) + 1):

            # Determine the radix qubits to perform the addition on
            a_block = a[R * i : R * (i + 1)]
            b_block = b[R * i : R * (i + 1)]

            if i == 0 and a_block != len(a_block) * "0":
                gidney_adder(a_block, b_block, ctrl=ctrl)
            elif a_block != len(a_block) * "0":
                gidney_adder(a_block, b_block, c_in=c[i - 1], ctrl=ctrl)
            elif not check_if_fresh([c[i - 1]], c[i - 1].qs()):
                lin_incr(b_block, c_in=c[i - 1])


def cq_sum_path_direct_uncomputation(a, b, c, R, ctrl=None):

    if len(a) < len(b):
        a = a + "0" * (len(b) - len(a))

    if R == 1 and False:
        # If R = 1, we are in the case of Drapers QCLA:
        # https://arxiv.org/abs/quant-ph/0406142
        # We can use the formula
        # S = A (+) B (+) C
        for i in range(len(c)):
            cx(c[i], b[i + 1])

        for i in range(len(a)):
            if a[i] == "1":
                if ctrl is None:
                    x(b[i])
                else:
                    cx(ctrl, b[i])
    else:
        # Execute addition using the corresponding carry values
        for i in range(len(c) + 1)[::-1]:

            # Determine the radix qubits to perform the addition on
            a_block = a[R * i : R * (i + 1)]
            b_block = b[R * i : R * (i + 1)]

            if check_if_fresh([c[i - 1]], c[i - 1].qs(), ignore_q_envs=False) or i == 0:
                c_in = None
                if a_block == "0" * len(a_block):
                    continue
            else:
                c_in = c[i - 1]

            if i == len(c):
                c_out = None
            else:
                c_out = c[i]

            if a_block != len(a_block) * "0":
                cq_gidney_adder(a_block, b_block, c_in=c_in, ctrl=ctrl, c_out=c_out)
            else:
                lin_incr(b_block, c_in=c_in, c_out=c_out)
