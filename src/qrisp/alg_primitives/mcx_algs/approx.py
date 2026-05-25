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

import math
import random
from numbers import Integral

from qrisp.core import QuantumVariable
from qrisp.core.gate_application_functions import cx, mcx, x
from qrisp.jasp import check_for_tracing_mode
from qrisp.misc import bin_rep
from qrisp.qtypes import QuantumBool


def sample_approx_mcx_masks(control_amount, k, seed=None):
    """
    Sample the parity masks used by :func:`approx_mcx`.

    Parameters
    ----------
    control_amount : int
        Number of control qubits in the MCX gate.
    k : int
        Number of random parity masks to sample.
    seed : int or random.Random, optional
        Randomness seed or random number generator object.
    """

    control_amount = int(control_amount)
    k = int(k)

    if isinstance(seed, random.Random):
        rng = seed
    else:
        rng = random.Random(seed)
        
    return [rng.getrandbits(control_amount) for _ in range(k)]


def _resolve_sample_count(epsilon, k):
    """
    Resolve the number of random parity masks used by :func:`approx_mcx`.

    Parameters
    ----------
    epsilon : float, optional
        Target error bound for choosing the number of masks.
    k : int, optional
        Number of random parity masks to sample.
    """

    if epsilon is None:
        return k

    min_k = max(1, math.ceil(math.log2(1 / epsilon)))

    if k is None:
        return min_k

    if k < min_k:
        raise Exception(
            f"k={k} is too small to guarantee epsilon={epsilon}; "
            f"need at least {min_k} samples"
        )

    return k


def approx_mcx(
    controls,
    target,
    epsilon=None,
    k=None,
    ctrl_state=-1,
    seed=None,
    inner_method="auto",
):
    """
    Apply the mixed-unitary approximate multi-controlled X from arXiv:2510.07223.

    Each call samples a Clifford+T circuit by first computing a small number of random
    control parities and then applying an exact MCX on those parity ancillas with
    negative controls. The approximation has one-sided error: the target always flips
    on the requested control state and flips spuriously with probability at most
    ``2**(-k)`` on any other basis state.

    Parameters
    ----------
    controls : list[Qubit] or QuantumVariable
        The qubits to control on.
    target : Qubit or QuantumBool
        The target qubit.
    epsilon : float, optional
        Target error bound. If given, ``k`` defaults to ``ceil(log2(1 / epsilon))``.
    k : int, optional
        Number of sampled parity checks.
    ctrl_state : int or str, optional
        The control state to activate the X gate on. The default is all ones.
    seed : int, optional
        Seed for deterministic mask sampling.
    inner_method : str, optional
        Exact MCX method used for the small ``k``-control gate. Phase-tolerant methods
        such as ``gray_pt`` are not supported here.
    """

    if check_for_tracing_mode():
        raise Exception("approx_mcx is not supported in tracing mode")

    if inner_method in ["approx", "gray_pt", "gray_pt_inv"]:
        raise Exception(
            f'inner_method "{inner_method}" is not supported for approx_mcx'
        )

    # Input conversion
    controls = list(controls)
    for i in range(len(controls)):
        if isinstance(controls[i], QuantumBool):
            controls[i] = controls[i][0]
    # Target conversion
    if isinstance(target, QuantumBool):
        target = target[0]
    elif isinstance(target, (QuantumVariable, list)):
        if len(target) != 1:
            raise Exception("approx_mcx target is not of type Qubit or QuantumBool")
        target = target[0]

    control_amount = len(controls)
    # Control state conversion
    if not isinstance(ctrl_state, str):
        if ctrl_state == -1:
            ctrl_state += 2**control_amount
        ctrl_state = bin_rep(ctrl_state, control_amount)[::-1]

    if len(ctrl_state) != control_amount:
        raise Exception(
            f"Given control state {ctrl_state} does not match "
            f"control qubit amount {control_amount}"
        )

    k = _resolve_sample_count(epsilon, k)
    masks = sample_approx_mcx_masks(control_amount, k, seed=seed)

    parity_register = QuantumVariable(
        k, qs=controls[0].qs(), name="approx_mcx_parity*"
    )
    # create a mask integer with only control bits set
    ctrl_one_mask = 0
    for index, desired_bit in enumerate(ctrl_state):
        if desired_bit == "1":
            ctrl_one_mask |= 1 << index
    # compute parity ancillas
    for ancilla_index, mask in enumerate(masks):
        ancilla = parity_register[ancilla_index]

        for control_index, control in enumerate(controls):
            # is bit control_index of mask set
            if (mask >> control_index) & 1:
                cx(control, ancilla)

        # The paper approximates OR over mismatch bits y_i = x_i xor ctrl_state_i.
        # So compute XOR over the raw controls and then correct it by the
        # parity of the selected desired-one bits.
        # & 1 is a cool trick to check if odd 
        if (mask & ctrl_one_mask).bit_count() & 1:
            x(ancilla)
        # So here ancilla is xor over selected mismatch bits
    
    # Do inner mcx flipping target only if all k parity register entries are 0
    mcx(
        parity_register,
        target,
        method=inner_method,
        ctrl_state="0" * k,
    )

    # Uncomputation
    for ancilla_index in range(k - 1, -1, -1):
        ancilla = parity_register[ancilla_index]
        mask = masks[ancilla_index]

        if (mask & ctrl_one_mask).bit_count() & 1:
            x(ancilla)

        for control_index in range(control_amount - 1, -1, -1):
            if (mask >> control_index) & 1:
                cx(controls[control_index], ancilla)

    parity_register.delete()

    return controls, target
