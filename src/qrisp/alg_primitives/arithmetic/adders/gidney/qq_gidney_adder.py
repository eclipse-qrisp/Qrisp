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

from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import cx, mcx
from qrisp.environments import custom_control, invert
from qrisp.qtypes import QuantumBool


# This function performs the Gidney adder from https://arxiv.org/pdf/1709.06648.pdf
@custom_control
def qq_gidney_adder(a, b, c_in=None, c_out=None, ctrl=None):

    if len(a) != len(b):
        raise Exception("Tried to call Gidney adder with inputs of unequal length")

    if c_out is not None:
        # Convert to qubit if neccessary
        if isinstance(c_out, QuantumBool):
            c_out = c_out[0]

        b = list(b) + [c_out]

    if len(b) == 1:
        if ctrl is not None:
            mcx([ctrl, a[0]], b)
        else:
            cx(a[0], b[0])
        if c_in is not None:
            if isinstance(c_in, QuantumBool):
                c_in = c_in[0]
            if ctrl is not None:
                cx(c_in, b[0])
            else:
                mcx([c_in, b[0]], b)
        return

    if ctrl is not None:
        gidney_control_anc = QuantumBool(name="gidney_control_anc*", qs=b[0].qs())

    gidney_anc = QuantumVariable(len(b) - 1, name="gidney_anc*", qs=b[0].qs())

    for i in range(len(b) - 1):

        if i != 0:
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(gidney_anc[i - 1], gidney_anc[i])
        elif c_in is not None:
            cx(c_in, b[i])
            cx(c_in, a[i])
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")
            cx(c_in, gidney_anc[i])
        else:
            mcx([a[i], b[i]], gidney_anc[i], method="gidney")

        if i != len(b) - 2:
            cx(gidney_anc[i], a[i + 1])
            cx(gidney_anc[i], b[i + 1])

    if ctrl is not None:
        mcx([ctrl, gidney_anc[-1]], b[-1])

        mcx([ctrl, a[-1]], gidney_control_anc[0], method="gidney")
        cx(gidney_control_anc[0], b[-1])
        mcx([ctrl, a[-1]], gidney_control_anc[0], method="gidney_inv")

    else:
        cx(gidney_anc[-1], b[-1])

    with invert():
        for i in range(len(b) - 1):

            if i != 0:
                if i != len(b) - 1:
                    cx(gidney_anc[i - 1], a[i])

                if ctrl is not None:
                    if i != len(b) - 1:
                        cx(gidney_anc[i - 1], b[i])
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney")
                    cx(gidney_control_anc[0], b[i])
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney_inv")

                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(gidney_anc[i - 1], gidney_anc[i])
            elif c_in is not None:
                cx(c_in, a[i])

                if ctrl is not None:
                    cx(c_in, b[i])
                    mcx([ctrl, c_in], gidney_control_anc[0], method="gidney")
                    cx(gidney_control_anc[0], b[i])
                    mcx([ctrl, c_in], gidney_control_anc[0], method="gidney_inv")

                mcx([a[i], b[i]], gidney_anc[i], method="gidney")
                cx(c_in, gidney_anc[i])
            else:

                if ctrl is not None:
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney")
                    cx(gidney_control_anc[0], b[i])
                    mcx([ctrl, a[i]], gidney_control_anc[0], method="gidney_inv")

                mcx([a[i], b[i]], gidney_anc[i], method="gidney")

    if ctrl is None:
        for i in range(len(a)):
            cx(a[i], b[i])
    else:
        gidney_control_anc.delete()

    gidney_anc.delete(verify=False)
