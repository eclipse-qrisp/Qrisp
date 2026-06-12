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

from collections.abc import Sequence
from functools import partial
from typing import Any
import numpy as np
from qrisp.core import QuantumVariable, Qubit
from qrisp.core.gate_application_functions import cx, ry, p, h, rz
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_w_state
from qrisp.environments import control

_FOQCS_SPIN_GLASS_TOL = 1e-12

def foqcs_prep_heisenberg(
    prep_qv: QuantumVariable | Sequence[Qubit],
    L: int,
    g: dict,
    J: dict,
    conjugate: bool = False
) -> None:
    r"""
    FOQCS-LCU state preparation, based on the methodology established in https://arxiv.org/pdf/2507.20887, pages 8-9.
    Implements the FOQCS-LCU PREP oracle for the Heisenberg Hamiltonian by preparing selector amplitudes
    for local X/Y/Z fields and nearest-neighbor XX/YY/ZZ couplings, then mapping them into the
    two FOQCS activation registers using Dicke-state and CNOT-ladder structures.
    
    
    Parameters
    ----------
    prep_qv : QuantumVariable | Sequence[Qubit]
        Ancillae qubits register for PREP.

    L : int
        Number of system qubits in the Heisenberg chain.

    g : dict (length 3)
        Dictionary of preprocessed local field coefficients.
        {
            "X": sqrt(gx * L) / norm,
            "Y": sqrt(gy * L * -1j) / norm,
            "Z": sqrt(gz * L) / norm
        }

    J : dict (length 3)
        Dictionary of preprocessed coupling coefficients for the Heisenberg interaction.
        {
            "X": sqrt(Jx * (L - 1)) / norm,
            "Y": sqrt(Jy * -(L - 1)) / norm,
            "Z": sqrt(Jz * (L - 1)) / norm
        }

    conjugate : bool = False
        Indicates whether the prep is PREP_R or PREP_L.
        In case of PREP_L, the conjugates of g and J are used.
        The default is False, which indicates it is PREP_R.

    Raises
    ------
    ValueError
        If ``g`` or ``J`` has length not equal to 3.

    """

    # Ref: Heisenberg Hamiltonian with local fields and nearest-neighbor
    # XX/YY/ZZ couplings is Eq. (50).
    # Ref: Table II maps each Pauli term to FOQCS-LCU coefficient-state pairs.

    # Check that received g and J dictionaries exactly contain the expected entries.
    req_keys = {"X", "Y", "Z"}

    g_keys = set(g.keys())
    J_keys = set(J.keys())

    if g_keys != req_keys:
        missing = req_keys - g_keys
        extra = g_keys - req_keys
        raise ValueError(
            f"g must contain exactly keys {sorted(req_keys)}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    if J_keys != req_keys:
        missing = req_keys - J_keys
        extra = J_keys - req_keys
        raise ValueError(
            f"J must contain exactly keys {sorted(req_keys)}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    # Ref: Fig. 5 shows that the Heisenberg PREP oracle uses 2L + 6 ancillae;
    # the first 6 ancillae select gx, gy, gz, Jx, Jy, Jz.
    extra_anc = 6 # Depends on the method and can be potentially decreased

    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")

    _g[0] = g["X"]
    _g[1] = g["Y"]
    _g[2] = g["Z"]
    _J[0] = J["X"]
    _J[1] = J["Y"]
    _J[2] = J["Z"]

    if conjugate:
        _g = np.conj(_g)
        _J = np.conj(_J)

    # SUBPREP
    # Ref: Eq. (56) gives the 6-entry selector state alpha used in Fig. 6.
    unbalanced_w_state(prep_qv[:extra_anc], np.block([_g, _J]))
    
    # PREP
    # Ref: Fig. 5 and Fig. 6 split the FOQCS ancillae into two L-qubit
    # activation registers: one for X activation and one for Z activation.
    fh1 = extra_anc                # First qubit first half
    lh1 = extra_anc + L - 1        # Last qubit first half
    fh2 = extra_anc + L            # First qubit second half
    lh2 = extra_anc + (L * 2) - 1  # Last qubit second half

    # 3 specific CNOTs
    # Ref: Table II: gx maps to |2^l>|0>, gy maps to |2^l>|2^l>,
    # and gz maps to |0>|2^l>.
    cx(prep_qv[0], prep_qv[lh1]) # from gx
    cx(prep_qv[1], prep_qv[lh1]) # from gy
    cx(prep_qv[2], prep_qv[lh2]) # from gz
    # 2 parallel Gamma gates
    # Ref: Fig. 2(a) defines Gamma(theta); Fig. 6 uses Gamma(theta_{n-1})
    # as part of the compact Heisenberg PREP circuit.
    theta = 2 * np.acos(np.sqrt(1 / (L - 1 + 1)))
    _gamma_gate(prep_qv[lh1 - 1], prep_qv[lh1], theta)
    _gamma_gate(prep_qv[lh2 - 1], prep_qv[lh2], theta)
    # 3 specific CNOTs
    # Ref: Table II: Jx/Jy/Jz nearest-neighbor terms map to two-excitation
    # patterns |2^l + 2^{l+1}> in the relevant activation register(s).
    cx(prep_qv[3], prep_qv[lh1 - 1]) # from Jx
    cx(prep_qv[4], prep_qv[lh1 - 1]) # from Jy
    cx(prep_qv[5], prep_qv[lh2 - 1]) # from Jz
    # 2 parallel delta gates
    # Ref: Fig. 2(d) defines the recursive Delta_n subcircuit used to prepare
    # the Dicke-state structure appearing in Fig. 6.
    theta_coeffs = []
    for i in range(1, L - 1):
        theta_coeffs.append(2 * np.acos(np.sqrt(1 / (i + 1))))
    _delta_gate(prep_qv[fh1:lh1], theta_coeffs)
    _delta_gate(prep_qv[fh2:lh2], theta_coeffs)
    # One spec CNOT
    # Ref: Fig. 6 compresses the Jx/Jy branches before the controlled ladder.
    cx(prep_qv[3], prep_qv[4]) # from Jx to Jy
    # Controlled CNOT ladder
    # Ref: Fig. 2(b) defines CL_1; Eq. (35) uses Delta plus CL_1 to build
    # nearest-neighbor two-excitation Dicke states.
    with control(prep_qv[4]): # from Jy
        _cx_ladder(prep_qv[fh1:lh1 + 1], L)
    # One spec CNOT
    cx(prep_qv[3], prep_qv[4]) # from Jx to Jy
    # Controlled CNOT ladder
    # Ref: Same CL_1 construction, now for the second activation register.
    with control(prep_qv[5]): # from Jz
        _cx_ladder(prep_qv[fh2:], L)
    # One spec CNOT
    cx(prep_qv[1], prep_qv[4]) # from gy to Jy
    # Controlled Element-wise CNOT
    # Ref: Fig. 2(c) defines EC; Fig. 6 uses EC to copy the first-register
    # Dicke pattern into the second register for Y-type terms.
    with control(prep_qv[4]): # from Jy
        cx(prep_qv[fh1:lh1 + 1], prep_qv[fh2:])
    # One spec CNOT
    cx(prep_qv[1], prep_qv[4]) # from gy to Jy

def foqcs_prep_spin_glass(
    prep_qv: QuantumVariable | Sequence[Qubit],
    L: int,
    g: dict,
    J: dict,
    conjugate: bool = False
) -> None:
    r"""
    FOQCS-LCU state preparation, based on the methodology established in https://arxiv.org/pdf/2507.20887, pages 9-11.
    Implements the FOQCS-LCU PREP oracle for the spin-glass Hamiltonian by preparing weighted selector states
    for non-uniform local X/Y/Z fields and distance-dependent XX/YY/ZZ couplings,
    then encoding them into FOQCS activation registers via unbalanced W/Dicke states,
    CNOT ladders, and register-copying for Y terms.
    Specifically, this is based on the optimal implementation as provided by the authors of the paper here:
    https://github.com/QuantumComputingLab/foqcs-lcu/blob/main/src/foqcs_lcu/spin_glass_block_encoding.py#L254
    See Eq. (58) for more.

    Parameters
    ----------
    prep_qv : QuantumVariable | Sequence[Qubit]
        Ancillae qubits register for PREP.

    L : int
        Number of system qubits.

    g : dict (length 3)
        Dictionary of local field coefficients.
        {"X" : [gx_1, gx_2, ..., gx_k],
         "Y" : [gy_1, gy_2, ..., gy_k],
         "Z" : [gz_1, gz_2, ..., gz_k]}

    J : dict (length 3)
        Dictionary of coupling coefficients.
        {"X" : [[Jx_12, Jx_23, ...], [Jx_13, Jx_24, ...], ...],
         "Y" : [[Jy_12, Jy_23, ...], [Jy_13, Jy_24, ...], ...],
         "Z" : [[Jz_12, Jz_23, ...], [Jz_13, Jz_24, ...], ...]}

    conjugate : bool = False
        Indicates whether the prep is PREP_R or PREP_L.
        In case of PREP_L, the conjugates of g and J are used.
        The default is False, which indicates it is PREP_R.

    Raises
    ------
    ValueError
        If ``g`` or ``J`` has length not equal to 3.

    """
    components = ["X", "Y", "Z"]

    # Normalize input representation.
    g_arr = {
        dim: np.asarray(g[dim], dtype=complex)
        for dim in components
    }

    J_arr = {
        dim: [
            np.asarray(J[dim][k - 1], dtype=complex)
            for k in range(1, L)
        ]
        for dim in components
    }

    if conjugate:
        g_arr = {
            dim: np.conj(g_arr[dim])
            for dim in components
        }
        J_arr = {
            dim: [np.conj(diag) for diag in J_arr[dim]]
            for dim in components
        }

    # Selector layout:
    #   X block: controls 0, ..., L-1
    #   Y block: controls L, ..., 2L-1
    #   Z block: controls 2L, ..., 3L-1
    #
    # Within each block:
    #   offset 0 selects g
    #   offset k selects J at distance k, for k = 1, ..., L-1
    extra_anc = 3 * L

    fh1 = extra_anc
    lh1 = extra_anc + L - 1
    fh2 = extra_anc + L
    lh2 = extra_anc + 2 * L - 1

    # SUBPREP over the 3L selector qubits.
    subprep_coeffs = []

    for dim in components:
        subprep_coeffs.append(np.linalg.norm(g_arr[dim]))

        for k in range(1, L):
            subprep_coeffs.append(np.linalg.norm(J_arr[dim][k - 1]))

    subprep_coeffs = np.asarray(subprep_coeffs, dtype=complex)
    subprep_norm = np.linalg.norm(subprep_coeffs)

    if subprep_norm <= _FOQCS_SPIN_GLASS_TOL:
        raise ValueError("Cannot prepare spin-glass PREP state: all coefficients are zero.")

    unbalanced_w_state(prep_qv[:extra_anc], subprep_coeffs / subprep_norm)

    # Compute unbalanced Dicke/W angles using absolute values.
    theta, cutoff = _theta_cutoff_foqcs_spin_glass_optimal(
        L,
        g_arr,
        J_arr
    )

    # Main optimal unbalanced-Dicke construction.
    for i in range(L - 1):
        # Activating CNOTs.
        cx(prep_qv[i], prep_qv[extra_anc + L - 1 - i])              # X branch
        cx(prep_qv[2 * L + i], prep_qv[extra_anc + 2 * L - 1 - i])  # Z branch
        cx(prep_qv[L + i], prep_qv[extra_anc + L - 1 - i])          # Y branch

        # X and Y branches share the first half register.
        theta_list = []
        qubit_list = []

        for k in range(i + 1):
            for axis in range(2):  # X, Y
                if cutoff[axis][k] >= i - k:
                    theta_list.append(theta[axis][k][i - k])
                    qubit_list.append(prep_qv[axis * L + k])

        _cgamma_opt(
            qubit_list,
            prep_qv[extra_anc + L - 2 - i],
            prep_qv[extra_anc + L - 1 - i],
            theta_list,
        )

        # Z branches use the second half register.
        theta_list = []
        qubit_list = []

        for k in range(i + 1):
            if cutoff[2][k] >= i - k:
                theta_list.append(theta[2][k][i - k])
                qubit_list.append(prep_qv[2 * L + k])

        _cgamma_opt(
            qubit_list,
            prep_qv[extra_anc + 2 * L - 2 - i],
            prep_qv[extra_anc + 2 * L - 1 - i],
            theta_list,
        )

    # Final activation CNOTs.
    cx(prep_qv[L - 1], prep_qv[extra_anc])          # X g/J last selector
    cx(prep_qv[2 * L - 1], prep_qv[extra_anc])      # Y g/J last selector
    cx(prep_qv[3 * L - 1], prep_qv[extra_anc + L])  # Z g/J last selector

    # Phase fixes for g branches:
    # The unbalanced Dicke preparation only uses the magnitudes of the
    # coefficients to set the rotation angles. This prepares the correct
    # probability weights, but drops any complex phases. Reinsert those phases
    # on the selected branch so that the prepared LCU coefficient state contains
    # the original complex amplitudes, not just their absolute values.
    with control(prep_qv[0]):
        _phase_fix_dicke1_unbalanced(
            prep_qv[fh1:lh1 + 1],
            g_arr["X"],
            cutoff[0][0]
        )

    with control(prep_qv[L]):
        _phase_fix_dicke1_unbalanced(
            prep_qv[fh1:lh1 + 1],
            g_arr["Y"],
            cutoff[1][0]
        )

    with control(prep_qv[2 * L]):
        _phase_fix_dicke1_unbalanced(
            prep_qv[fh2:lh2 + 1],
            g_arr["Z"],
            cutoff[2][0]
        )

    # Phase fixes for J branches:
    # Same logic as the phase fixes for g branches.
    for k in range(1, L):
        with control(prep_qv[k]):
            _phase_fix_dicke1_unbalanced(
                prep_qv[fh1:fh1 + L - k],
                J_arr["X"][k - 1],
                cutoff[0][k]
            )

        with control(prep_qv[L + k]):
            _phase_fix_dicke1_unbalanced(
                prep_qv[fh1:fh1 + L - k],
                J_arr["Y"][k - 1],
                cutoff[1][k]
            )

        with control(prep_qv[2 * L + k]):
            _phase_fix_dicke1_unbalanced(
                prep_qv[fh2:fh2 + L - k],
                J_arr["Z"][k - 1],
                cutoff[2][k]
            )

    # CNOT ladders complete the two-body J terms.
    for k in range(1, L):
        # Activate X -> Y branch so the same controlled ladder handles X and Y.
        cx(prep_qv[k], prep_qv[L + k])

        with control(prep_qv[L + k]):
            _cx_ladder(prep_qv[fh1:lh1 + 1], L, k)

        with control(prep_qv[2 * L + k]):
            _cx_ladder(prep_qv[fh2:lh2 + 1], L, k)

        # Undo activation.
        cx(prep_qv[k], prep_qv[L + k])

    # Element-wise CNOT for all Y terms.
    for i in range(L, 2 * L - 1):
        cx(prep_qv[i], prep_qv[i + 1])

    with control(prep_qv[2 * L - 1]):
        cx(prep_qv[fh1:lh1 + 1], prep_qv[fh2:lh2 + 1])

    for i in reversed(range(L, 2 * L - 1)):
        cx(prep_qv[i], prep_qv[i + 1])

###################################
############# Helpers #############
###################################
    
def get_foqcs_lcu_prep_num_of_ancillae(prep: partial, num_q_ops: int = 1) -> int:
    r"""
        Gets a number of ancillae qubits for the FOQCS-LCU circuit that uses
        ``prep`` method to encode ``num_q_ops`` qubits.

        Parameters
        ----------
        prep : partial
            Partially initialised FOQCS-LCU PREP method.
        
        num_q_ops : int
            Number of operand qubits (L argument for FOQCS-LCU PREP routines).
            The default is 1.

        Returns
        -------
        int
            An integer with number of ancillae required by the received FOQCS-LCU PREP method
    """
    if prep.func == foqcs_prep_heisenberg:
        return num_q_ops * 2 + 6
    elif prep.func == foqcs_prep_spin_glass:
        return num_q_ops * 5
    else:
        raise ValueError(f"Received unknown FOQCS-LCU PREP routine: {prep}")

def _angles_dicke_unbalanced(coeff: Sequence[float]) -> tuple[list[float], int]:
    """
    Helper equivalent of angles_dicke1_unbalanced from foqcs-lcu
    https://github.com/QuantumComputingLab/foqcs-lcu

    coeff must be normalized, real, and non-negative.
    """

    coeff = np.asarray(coeff, dtype=float)
    L = len(coeff)

    theta = []
    sum_squared = 0.0
    cutoff = L - 1

    for i in reversed(range(L)):
        if sum_squared >= 1:
            cutoff = max(L - i - 2, 0)
            break

        denom = np.sqrt(max(1 - sum_squared, 0.0))

        if denom == 0:
            angle = 0.0
        else:
            angle = 2 * np.arccos(np.clip(coeff[i] / denom, -1, 1))

        theta.append(angle)
        sum_squared += coeff[i] ** 2

    return theta, cutoff

def _theta_cutoff_foqcs_spin_glass_optimal(
    L: int,
    g: dict,
    J: dict
) -> tuple[list[list[list[float]]], list[list[int]]]:
    """
    Computes the angle/cutoff data used by foqcs_prep_spin_glass_optimal.

    theta[axis][k] contains the unbalanced-Dicke angles for:
        k = 0: local field g[axis]
        k > 0: kth-nearest-neighbour coupling J[axis][k - 1]
    """

    components = ["X", "Y", "Z"]

    theta = []
    cutoff = []

    for dim in components:
        theta_k = []
        cutoff_k = []

        # k = 0: g branch.
        coeff = np.abs(np.asarray(g[dim], dtype=complex))
        norm = np.linalg.norm(coeff)

        if norm > _FOQCS_SPIN_GLASS_TOL:
            theta_g, cutoff_g = _angles_dicke_unbalanced(coeff / norm)
        else:
            theta_g = [0.0] * L
            cutoff_g = 0

        theta_k.append(theta_g)
        cutoff_k.append(cutoff_g)

        # k = 1, ..., L-1: J branches.
        for k in range(1, L):
            coeff = np.abs(np.asarray(J[dim][k - 1], dtype=complex))
            norm = np.linalg.norm(coeff)

            if norm > _FOQCS_SPIN_GLASS_TOL:
                theta_J, cutoff_J = _angles_dicke_unbalanced(coeff / norm)
            else:
                theta_J = [0.0] * (L - k)
                cutoff_J = 0

            theta_k.append(theta_J)
            cutoff_k.append(cutoff_J)

        theta.append(theta_k)
        cutoff.append(cutoff_k)

    return theta, cutoff

def _phase_fix_dicke1_unbalanced(
    qv: QuantumVariable | Sequence[Qubit],
    coeff: Sequence[complex],
    cutoff: int | None = None
) -> None:
    """
    Helper equivalent of phase_fix_dicke1_unbalanced from foqcs-lcu
    https://github.com/QuantumComputingLab/foqcs-lcu

    Applies phase corrections for a Dicke-1 / W-state-style register whose
    amplitudes were prepared from coefficient magnitudes.

    For each coefficient ``coeff[i]``, this applies a single-qubit phase gate
    to the corresponding qubit ``qv[i]``. This converts amplitudes prepared
    as ``abs(coeff[i])`` into amplitudes with the desired complex phase
    ``coeff[i] = abs(coeff[i]) * exp(1j * angle(coeff[i]))``.

    If ``cutoff`` is given, only entries ``0`` through ``cutoff`` are
    corrected. Phases that are zero up to numerical tolerance are skipped.

    """

    coeff = np.asarray(coeff, dtype=complex)

    if cutoff is None:
        cutoff = len(coeff) - 1
    
    max_i = min(cutoff, len(coeff) - 1)

    for i in range(max_i + 1):
        angle = np.mod(np.angle(coeff[i]), 2 * np.pi)

        if angle > _FOQCS_SPIN_GLASS_TOL:
            p(angle, qv[i])

def _cgamma_opt(
    controls: Sequence[Qubit],
    qv_rot: Qubit,
    qv_x: Qubit,
    theta_list: Sequence[float],
) -> None:
    """
    Implementation of the compressed controlled-gamma gate `cgamma_opt`
    from https://github.com/QuantumComputingLab/foqcs-lcu

    This follows the upstream decomposition:

        S(rot), H(rot),
        controlled RZs on rot,
        CX(rot, x),
        controlled inverse RZs on x,
        H(rot), H(x), Sdg(x),
        CX(rot, x),
        S(rot), H(rot)
    """

    if len(theta_list) == 0:
        return

    p(np.pi / 2, qv_rot)
    h(qv_rot)

    for ctrl, theta in zip(controls, theta_list):
        ang = (np.pi - theta) / 2.0
        with control(ctrl):
            rz(ang, qv_rot)

    cx(qv_rot, qv_x)

    for ctrl, theta in zip(controls, theta_list):
        ang = (np.pi - theta) / 2.0
        with control(ctrl):
            rz(-ang, qv_x)

    h(qv_rot)
    h(qv_x)
    p(-np.pi / 2, qv_x)

    cx(qv_rot, qv_x)

    p(np.pi / 2, qv_rot)
    h(qv_rot)

def _cx_ladder(qv: QuantumVariable | Sequence[Qubit], n: int, k: int = 1) -> None:

    for i in reversed(range(0, n - k)):
        cx(qv[i], qv[i + k])

def _gamma_gate(qv_rot: Any, qv_x: Any, theta: float):
    with control(qv_x):
        ry(theta, qv_rot)
    cx(qv_rot, qv_x)

def _delta_gate(qv: QuantumVariable | Sequence[Qubit], theta_coeffs):
    if len(theta_coeffs) == 0:
        return
    if len(theta_coeffs) == 1:
        _gamma_gate(qv[0], qv[1], theta_coeffs[0])
    else:
        _gamma_gate(qv[-2], qv[-1], theta_coeffs[-1])
        _delta_gate(qv[:-1], theta_coeffs[:-1])
