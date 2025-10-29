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

import numpy as np

from qrisp.alg_primitives.arithmetic.adders.gidney import cq_gidney_adder
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_carry_path import (
    cq_calc_carry,
)
from qrisp.alg_primitives.arithmetic.adders.qcla.classical_quantum.cq_sum_path import (
    cq_sum_path,
    cq_sum_path_direct_uncomputation,
)
from qrisp.circuit import fast_append
from qrisp.core.gate_application_functions import cx, x
from qrisp.core.session_merging_tools import merge
from qrisp.environments import QuantumEnvironment, custom_control, invert
from qrisp.misc.utility import bin_rep, redirect_qfunction

verify_manual_uncomputations = np.zeros(1)


# This function performs the in-place addition
# b += a
# based on the higher radix qcla
# The overall radix can be specified as an exponential of the form
# R = radix_base**radix_exponent
@custom_control
def cq_qcla(a, b, radix_base=2, radix_exponent=1, t_depth_reduction=True, ctrl=None):

    if isinstance(a, int):
        a = bin_rep(a % (2 ** len(b)), len(b))[::-1]

    if len(a) > len(b):
        raise Exception(
            "Tried to add QuantumFloat of higher precision onto QuantumFloat of lower precision"
        )

    R = radix_base**radix_exponent

    merge([a, b])
    with fast_append(3):
        # The case that a only has a single qubit is simple.
        if len(b) == 1:
            if a[0] == "1":
                if ctrl is None:
                    x(b[0])
                else:
                    cx(ctrl, b[0])
            return
        elif len(b) <= R:
            cq_gidney_adder(a, b, ctrl=ctrl)
            return

        # Calculate the carry
        # Executing within a QuantumEnvironemnt accelerates the uncomputation algorithm
        # because it doesn't have to consider the operations appended outside of this function
        with QuantumEnvironment():
            merge([b[0].qs()])
            c = cq_calc_carry(a, b, radix_base, radix_exponent, ctrl=ctrl)

            if t_depth_reduction:
                cq_sum_path_direct_uncomputation(a, b, c, R, ctrl=ctrl)
            else:
                cq_sum_path(a, b, c, R, ctrl=ctrl)

                # To uncompute the carry we use Drapers strategy
                # CARRY(A,B) = CARRY(A, NOT(A+B))
                # We therefore bitflip the sum
                for i in range(len(b)):
                    x(b[i])

                # Contrary to Draper's adder we don't need to uncompute every carry digit.
                # Because of the above equivalence, the carries agree on every digit, so especially
                # on the digits representing the output of the calc_carry function. We can therefore
                # uncompute using calc_carry (even with higher radix) by inverting calc_carry.

                with invert():
                    # We use the redirect_qfunction decorator to steer the function onto c
                    redirect_qfunction(cq_calc_carry)(
                        a, b, radix_base, radix_exponent, target=c, ctrl=ctrl
                    )

                # Flip the sum back
                for i in range(len(b)):
                    x(b[i])

        # Delete c
        c.delete(verify=bool(verify_manual_uncomputations[0]))
