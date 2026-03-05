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

import jax.numpy as jnp

from qrisp.jasp import qache, jrange, AbstractQubit, make_jaspr, Jaspr
from qrisp.core import x, h, cx, t, t_dg, s, measure, cz, mcx, QuantumVariable
from qrisp.qtypes import QuantumBool
from qrisp.environments import control, custom_control
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_cq_gidney_adder import (
    jasp_cq_gidney_adder,
)

# Addition implementation based on https://arxiv.org/pdf/1709.06648


@custom_control
def jasp_qq_gidney_adder(a, b, ctrl=None):

    if isinstance(b, list):
        n = min(len(a), len(b))
        perform_incrementation = n < len(b)
    else:
        n = jnp.min(jnp.array([a.size, b.size]))
        perform_incrementation = n < b.size

    if ctrl is not None:
        ctrl_anc = QuantumBool(name="gidney_anc_2*")

    # If the addition is only a single qubit, it can be done with a CX gate (below)
    with control(n > 1):

        # Allocate the ancillae
        gidney_anc = QuantumVariable(n - 1, name="gidney_anc*")

        i = 0
        # Perform the initial mcx
        mcx([a[i], b[i]], gidney_anc[i], method="gidney")

        # Perform the left part of the V-Shape
        for j in jrange(n - 2):
            i = j + 1

            cx(gidney_anc[i - 1], a[i])
            cx(gidney_anc[i - 1], b[i])

            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])

        # This part handles the case that the addition target has more qubit than
        # the control value.
        # We solve this issue by performing a 1-incrementation on the remainder
        # if the carry out value is True
        with control(perform_incrementation):

            # Compute the carry out similar to the loop above

            carry_out = QuantumBool()

            cx(gidney_anc[n - 2], a[n - 1])
            cx(gidney_anc[n - 2], b[n - 1])

            mcx([a[n - 1], b[n - 1]], carry_out[0], method="gidney")
            cx(gidney_anc[n - 2], carry_out[0])

            # Perform a controlled incrtementation
            ctrl_list = [carry_out[0]]
            if ctrl is not None:
                ctrl_list.append(ctrl)

            with control(ctrl_list):
                jasp_cq_gidney_adder(1, b[n:])

            # Uncompute the carry
            cx(gidney_anc[n - 2], carry_out[0])
            mcx([a[n - 1], b[n - 1]], carry_out[0], method="gidney_inv")
            carry_out.delete()

            cx(gidney_anc[n - 2], a[n - 1])
            if ctrl is not None:
                cx(gidney_anc[n - 2], b[n - 1])

        if ctrl is not None:
            # This is the CX at the "tip" of the V shape
            mcx([ctrl, gidney_anc[n - 2]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b[n - 1])
            mcx([ctrl, gidney_anc[n - 2]], ctrl_anc[0], method="gidney_inv")

            # This is the CX at the lower right of the circuit
            mcx([ctrl, a[n - 1]], ctrl_anc[0], method="gidney")
            cx(ctrl_anc[0], b[n - 1])
            mcx([ctrl, a[n - 1]], ctrl_anc[0], method="gidney_inv")
        else:
            with control(jnp.logical_not(perform_incrementation)):
                cx(gidney_anc[n - 2], b[n - 1])
            cx(a[n - 1], b[n - 1])

        # Perform the right part of the V shape
        for j in jrange(n - 2):
            i = n - j - 2

            cx(gidney_anc[i - 1], gidney_anc[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney_inv")

            if ctrl is not None:
                # This is the controlled version described on page 4
                mcx([ctrl, a[i]], ctrl_anc[0], method="gidney")
                cx(ctrl_anc[0], b[i])
                mcx([ctrl, a[i]], ctrl_anc[0], method="gidney_inv")
                cx(gidney_anc[i - 1], a[i])
                cx(gidney_anc[i - 1], b[i])
            else:
                cx(gidney_anc[i - 1], a[i])
                cx(a[i], b[i])

        # The final uncomputation
        mcx([a[0], b[0]], gidney_anc[0], method="gidney_inv")

        # Delete the ancilla
        gidney_anc.delete()

    with control((n == 1) & perform_incrementation):

        ctrl_list = [a[0], b[0]]

        if ctrl is not None:
            ctrl_list.append(ctrl)

        with control(ctrl_list):
            jasp_cq_gidney_adder(1, b[n:])

    # Perform the CX gate at the top right of the circuit
    if ctrl is not None:
        mcx([ctrl, a[0]], ctrl_anc[0], method="gidney")
        cx(ctrl_anc[0], b[0])
        mcx([ctrl, a[0]], ctrl_anc[0], method="gidney_inv")
    else:
        cx(a[0], b[0])

    if ctrl is not None:
        ctrl_anc.delete()
