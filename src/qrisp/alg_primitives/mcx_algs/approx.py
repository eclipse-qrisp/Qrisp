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


def _resolve_rng(seed):
    if isinstance(seed, random.Random):
        return seed

    return random.Random(seed)


def sample_approx_mcx_masks(control_amount, k, seed=None):
    """
    Sample the parity masks used by :func:`approx_mcx`.

    The sampling follows the construction in Gosset, Kothari, and Zhang,
    "Multi-qubit Toffoli with exponentially fewer T gates"
    (arXiv:2510.07223).
    """

    if not isinstance(control_amount, Integral) or int(control_amount) < 0:
        raise Exception("control_amount must be a non-negative integer")

    if not isinstance(k, Integral) or int(k) < 1:
        raise Exception("k must be a positive integer")

    control_amount = int(control_amount)
    k = int(k)

    rng = _resolve_rng(seed)
    return [rng.getrandbits(control_amount) for _ in range(k)]


def _normalize_ctrl_state(ctrl_state, control_amount):
    if isinstance(ctrl_state, str):
        normalized_ctrl_state = ctrl_state
    else:
        if ctrl_state == -1:
            ctrl_state += 2**control_amount
        normalized_ctrl_state = bin_rep(ctrl_state, control_amount)[::-1]

    if len(normalized_ctrl_state) != control_amount:
        raise Exception(
            f"Given control state {ctrl_state} does not match "
            f"control qubit amount {control_amount}"
        )

    return normalized_ctrl_state


def _resolve_sample_count(epsilon, k):
    if epsilon is None and k is None:
        raise Exception('approx_mcx requires either "epsilon" or "k"')

    if k is not None:
        if not isinstance(k, Integral) or int(k) < 1:
            raise Exception("k must be a positive integer")
        k = int(k)

    if epsilon is None:
        return k

    if not 0 < epsilon < 1:
        raise Exception("epsilon must satisfy 0 < epsilon < 1")

    min_k = max(1, math.ceil(math.log2(1 / epsilon)))

    if k is None:
        return min_k

    if k < min_k:
        raise Exception(
            f"k={k} is too small to guarantee epsilon={epsilon}; "
            f"need at least {min_k} samples"
        )

    return k


def _normalize_controls(controls):
    normalized_controls = []

    for control in list(controls):
        if isinstance(control, QuantumBool):
            normalized_controls.append(control[0])
        else:
            normalized_controls.append(control)

    return normalized_controls


def _normalize_target(target):
    if isinstance(target, QuantumBool):
        return target[0]

    if isinstance(target, QuantumVariable):
        if len(target) != 1:
            raise Exception("approx_mcx target is not of type Qubit or QuantumBool")
        return target[0]

    if isinstance(target, list):
        if len(target) != 1:
            raise Exception("approx_mcx target is not of type Qubit or QuantumBool")
        return target[0]

    return target


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
        raise Exception("approx_mcx is currently not supported in tracing mode")

    if inner_method in ["approx", "gray_pt", "gray_pt_inv"]:
        raise Exception(
            f'inner_method "{inner_method}" is not supported for approx_mcx'
        )

    controls = _normalize_controls(controls)
    target = _normalize_target(target)

    control_amount = len(controls)

    if control_amount == 0:
        return controls, target

    ctrl_state = _normalize_ctrl_state(ctrl_state, control_amount)
    k = _resolve_sample_count(epsilon, k)
    masks = sample_approx_mcx_masks(control_amount, k, seed=seed)

    parity_register = QuantumVariable(
        k, qs=controls[0].qs(), name="approx_mcx_parity*"
    )

    ctrl_one_mask = 0
    for index, desired_bit in enumerate(ctrl_state):
        if desired_bit == "1":
            ctrl_one_mask |= 1 << index

    for ancilla_index, mask in enumerate(masks):
        ancilla = parity_register[ancilla_index]

        for control_index, control in enumerate(controls):
            if (mask >> control_index) & 1:
                cx(control, ancilla)

        # The paper approximates OR over mismatch bits y_i = x_i xor ctrl_state_i.
        # We therefore compute XOR over the raw controls and then correct it by the
        # parity of the selected desired-one bits.
        if (mask & ctrl_one_mask).bit_count() & 1:
            x(ancilla)

    mcx(
        parity_register,
        target,
        method=inner_method,
        ctrl_state="0" * k,
    )

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
