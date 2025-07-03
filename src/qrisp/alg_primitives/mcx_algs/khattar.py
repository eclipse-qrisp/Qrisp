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

from qrisp.core.gate_application_functions import (
    x,
    cx,
    mcx,
    h,
    t,
    t_dg,
    sx,
    cz,
    p,
    cp,
    measure,
)
from qrisp.qtypes import QuantumFloat
from qrisp.environments import invert, control, conjugate, custom_inversion
from qrisp.jasp import jlen, jrange, check_for_tracing_mode, qache

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

@custom_inversion
def gidney_CCCX(ctrls, target, inv=False):
    """
    Implements a CCCZ gate using the Gidney 
    method described in https://arxiv.org/abs/2106.11513 using only 6 T gates.
    Args:
        ctrls (list): A list of control qubits. It is expected to contain three qubits.
        target (list): A list containing the target qubit. It is expected to contain one qubit.
    """
    
    h(target[0])
    
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
    
    with invert():
        sx(gidney_anc[0])
    cl_res = measure(gidney_anc[0])

    with control(cl_res):
        cz(ctrls[2], target[0])
    cz(ctrls[2], target[0])

    with control(cl_res):
        cz(ctrls[0], ctrls[1])
        x(gidney_anc[0])
        
    h(target[0])

    gidney_anc.delete()


def cca_4ctrls(ctrls, target):
    #This function handles the case of 4 control qubits in the conditionally clean ancillae MCX
    cca4_anc = QuantumFloat(1)

    mcx([ctrls[1], ctrls[0]], cca4_anc[0], method="gray_pt")
    x(ctrls[1])
    mcx([ctrls[3], ctrls[2]], ctrls[1], method="gray_pt")

    mcx([ctrls[1], cca4_anc[0]], target[0])

    mcx([ctrls[2], ctrls[3]], ctrls[1], method="gray_pt")
    x(ctrls[1])
    mcx([ctrls[0], ctrls[1]], cca4_anc[0], method="gray_pt")

    cca4_anc.delete()


def cca_mcx(ctrls, target, anc):

    n = jlen(ctrls)

    if isinstance(ctrls, list):
        xrange = range
    else:
        xrange = jrange
        
    # STEP 1
    mcx([ctrls[0], ctrls[1]], anc[0], method="gidney")

    for i in xrange((n - 2) // 2):
        mcx([ctrls[2 * i + 3], ctrls[2 * i + 2]], ctrls[2 * i + 1], method="gray_pt")

    x(ctrls[:-2])

    # STEP 2
    a = n - 3 + 2 * (n & 1)
    b = n - 5 + (n & 1)
    c = n - 6 + (n & 1)

    with control(c > -1):
        mcx([ctrls[b], ctrls[a]], ctrls[c], method="gray_pt")

    with invert():
        for i in xrange((c + 1) // 2):
            mcx([ctrls[2 * i + 2], ctrls[2 * i + 1]], ctrls[2 * i], method="gray_pt")

    return ctrls, target, anc


@qache
def khattar_mcx(ctrls, target, ctrl_state):
    """
    Implements the Khattar multi-controlled X (MCX) gate methode using conditionally clean ancillae described in https://arxiv.org/abs/2407.17966.
    The behavior of the function varies depending on the number of 
    control qubits (N) and the control state.
    Args:
        ctrls (list): A list of control qubits.
        target (list): A list containing the target qubit(s).
        ctrl_state (int or str): The control state, which can be provided as an integer or a binary string.
            If provided as a string, it is reversed and converted to an integer.
    Behavior:
        - For N = 1: A single-controlled X gate is applied.
        - For N = 2: A two-controlled X gate is applied.
        - For N = 3: If in tracing mode, a decomposition using Hadamard gates and a CCCZ gate is applied.
          Otherwise, a specific MCX method ("balauca") is used.
        - For N = 4: A custom 4-controlled gate (`cca_4ctrls`) is applied.
        - For N > 4: An ancillary qubit is used to decompose the operation into smaller steps.
    """
    
    N = jlen(ctrls)

    if isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state[::-1], 2)

    ctrl_state = jnp.int64(ctrl_state)
    ctrl_state = cond(ctrl_state == -1, lambda x: x + 2**N, lambda x: x, ctrl_state)

    with conjugate(ctrl_state_conjugator)(ctrls, ctrl_state):

        if isinstance(ctrls, list):
            if N == 1:
                cx(ctrls[0], target[0])

            if N == 2:
                mcx([ctrls[0], ctrls[1]], target[0])

            if N == 3:
                if check_for_tracing_mode():
                    gidney_CCCX(ctrls, target)
                else:
                    mcx(ctrls, target[0], method="balauca")  # CHANGE

            if N == 4:
                cca_4ctrls(ctrls, target)

            if N > 4:
                khattar_anc = QuantumFloat(1)
                with conjugate(cca_mcx)(ctrls, target, khattar_anc):
                    # STEP 3
                    mcx([khattar_anc[0], ctrls[0]], target[0])
                khattar_anc.delete()

        else:
            with control(N == 1):
                cx(ctrls[0], target[0])

            with control(N == 2):
                mcx([ctrls[0], ctrls[1]], target[0])

            with control(N == 3):
                if check_for_tracing_mode():
                    gidney_CCCX(ctrls, target)
                else:
                    mcx(ctrls, target[0], method="balauca")  # CHANGE

            with control(N == 4):
                cca_4ctrls(ctrls, target)

            with control(N > 4):
                khattar_anc = QuantumFloat(1)
                with conjugate(cca_mcx)(ctrls, target, khattar_anc):
                    # STEP 3
                    mcx([khattar_anc[0], ctrls[0]], target[0])
                khattar_anc.delete()


@qache
def khattar_mcp(phi, ctrls, ctrl_state):
    """
    Implements the multi-controlled phase (MCP) gate based on the Khattar MCX implementation.
    
    """
    
    N = jlen(ctrls)

    if isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state[::-1], 2)

    ctrl_state = jnp.int64(ctrl_state)
    ctrl_state = cond(ctrl_state == -1, lambda x: x + 2**N, lambda x: x, ctrl_state)
    target = QuantumFloat(1)

    with conjugate(ctrl_state_conjugator)(ctrls, ctrl_state):

        with control(N == 1):
            cp(phi, ctrls[0], target[0])

        with control(N == 2):
            with conjugate(mcx)([ctrls[0], ctrls[1]], target[0], method="gray_pt"):
                p(phi, target[0])

        with control(N == 3):
            if check_for_tracing_mode():
                gidney_CCCX(ctrls, target)
                
                p(phi, target[0])

                gidney_CCCX(ctrls, target)
            else:
                with conjugate(mcx)(ctrls, target[0], method="balauca"):
                    p(phi, target[0])

        with control(N == 4):
            with conjugate(cca_4ctrls)(ctrls, target):
                p(phi, target[0])

        with control(N > 4):
            khattar_anc = QuantumFloat(1)
            with conjugate(cca_mcx)(ctrls, target, khattar_anc):
                with conjugate(mcx)([khattar_anc[0], ctrls[0]], target[0]):
                    p(phi, target[0])
            khattar_anc.delete()

    target.delete()
