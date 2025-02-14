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

from qrisp.core.gate_application_functions import (
    x,
    cx,
    mcx,
    h,
    t,
    t_dg,
    sx,
    cz,
    measure,
)
from qrisp.qtypes import QuantumFloat
from qrisp.environments import invert, control, conjugate
from qrisp.jasp import jlen, jrange, check_for_tracing_mode

# Move this one layer up
import jax.numpy as jnp
from jax.lax import cond
from jax import jit


# Move this one layer up
@jit
def extract_boolean_digit(integer, digit):
    return jnp.bool((integer >> digit & 1))


def ctrl_state_conjugator(ctrls, ctrl_state):

    if isinstance(ctrls, list):
        xrange = range
    else:
        xrange = jrange

    N = jlen(ctrls)

    for i in xrange(N):
        with control(~extract_boolean_digit(ctrl_state, i)):
            x(ctrls[i])


def gidney_CCCZ(ctrls, target):
    gidney_anc = QuantumFloat(1)

    h(gidney_anc[0])
    t(gidney_anc[0])
    cx(ctrls[1], gidney_anc[0])
    t_dg(gidney_anc[0])
    cx(ctrls[0], gidney_anc[0])
    t(gidney_anc[0])
    cx(ctrls[1], gidney_anc[0])
    cx(ctrls[2], gidney_anc[0])
    t_dg(gidney_anc[0])
    cx(target[0], gidney_anc[0])
    t(gidney_anc[0])
    cx(ctrls[2], gidney_anc[0])
    t_dg(gidney_anc[0])
    cx(target[0], gidney_anc[0])
    sx(gidney_anc[0])
    cl_res = measure(gidney_anc[0])

    with control(cl_res):
        cz(ctrls[2], target[0])
    cz(ctrls[2], target[0])
        
    with control(cl_res):
        cz(ctrls[0], target[0])
        x(gidney_anc[0])
    
    gidney_anc.delete()
def cca_4ctrls(ctrls, target):

    cca4_anc = QuantumFloat(1)
    
    mcx([ctrls[0], ctrls[1]], cca4_anc[0]) 
    x(ctrls[1])
    mcx([ctrls[2], ctrls[3]], ctrls[1])
    
    mcx([ctrls[1], cca4_anc[0]], target[0])

    mcx([ctrls[2], ctrls[3]], ctrls[1])
    x(ctrls[1])
    mcx([ctrls[0], ctrls[1]], cca4_anc[0]) 
    
    cca4_anc.delete()


def cca_mcx(ctrls, target, anc):

    n = jlen(ctrls)

    # STEP 1
    # This operation is always executed regardless of the lenght of ctrls
    mcx(
        [ctrls[0], ctrls[1]], anc[0]
    )  # DISCUSS, MAYBE THIS SHOULD CALL GIDNEY LOGICAL AND

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


# SHOULD USE @qache DECORATOR?
def khattar_mcx(ctrls, target, ctrl_state):
    N = jlen(ctrls)

    ctrl_state = jnp.int64(ctrl_state)
    ctrl_state = cond(ctrl_state == -1, lambda x: x + 2**N, lambda x: x, ctrl_state)

    with conjugate(ctrl_state_conjugator)(ctrls, ctrl_state):

        # CASE DISTINCTION
        with control(N == 1):
            cx(ctrls[0], target[0])

        with control(N == 2):
            mcx([ctrls[0], ctrls[1]], target[0])

        with control(N == 3):
            if check_for_tracing_mode():
                h(target[0])
                gidney_CCCZ(ctrls, target)
                h(target[0])
            else:
                mcx(ctrls, target[0], method="balauca")  # CHANGE

        with control(N == 4):
            cca_4ctrls(ctrls, target)

        with control(N > 4):
            khattar_anc = QuantumFloat(1)
            with conjugate(cca_mcx)(ctrls, target, khattar_anc):
                # STEP 3
                print(type(khattar_anc))
                mcx([khattar_anc[0], ctrls[0]], target[0])
            khattar_anc.delete()
